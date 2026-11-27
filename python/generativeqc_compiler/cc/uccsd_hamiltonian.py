"""Shared unrestricted raw-Hamiltonian, normal-ordering and orbital pullbacks.

Both native backends consume one primal binding and generic TensorIR AD. Raw
chemists' integrals and independent spin frames are not relaxed AO densities by
themselves; runtime consumers qualify CC/Lambda, shared UHF response and physical
producer identities before exposing relaxed cotangents or nuclear derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from generativeqc_compiler.cc.uccsd_equations import build_program
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES, UCCSDLayout
from generativeqc_compiler.cc.uccsd_response import RESIDUALS
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    JVPProgram,
    Node,
    Program,
    Symmetry,
    TensorSpec,
    VJPProgram,
    add,
    constant,
    einsum,
    input_tensor,
    linearize,
    reshape,
    slice_tensor,
    transpose,
    transpose_program,
)
from generativeqc_compiler.tensor.ad_program import _rebuild

RAW_INPUTS = ("h_a", "h_b", "g_aa", "g_ab", "g_bb", "rotation_a", "rotation_b")
CANONICAL_HAMILTONIAN_INPUTS = RAW_INPUTS[:5]
INDEPENDENT_FOCK_INPUTS = ("reference_g_aa", "reference_g_ab", "reference_g_bb")


@dataclass(frozen=True)
class UCCSDHamiltonianPrograms:
    """Fixed-occupation primal, raw-input/rotation bars and Fov rotation JVP."""

    primal: Program
    pullback: VJPProgram
    orbital_jvp: JVPProgram
    parameter_names: tuple[str, ...]


def build_raw_hamiltonian(
    layout: UCCSDLayout, *, max_elements: int = 1_000_000, separate_fock: bool = False
) -> UCCSDHamiltonianPrograms:
    """Rotate raw spin matrices, build normal ordering, and bind every CC input.

    The mixed Coulomb tensor has alpha owners on its first pair and beta owners
    on its second. It is not an independent beta-alpha tensor and does not have
    fixed-owner pair-exchange symmetry. Rotation matrices need not be orthogonal
    during AD, so a later overlap-transport chain can use their symmetric bars.
    """
    primal, names, reference_inputs = _raw_hamiltonian_primal(
        layout, separate_fock=separate_fock, identity_frame=False
    )
    pullback = transpose_program(
        primal,
        (*names, "reference_electronic_energy"),
        inputs=(*RAW_INPUTS, *reference_inputs),
        max_elements=max_elements,
    )
    orbital_jvp = linearize(
        primal,
        ("rotation_a", "rotation_b"),
        outputs=("fov_aa", "fov_bb"),
        max_elements=max_elements,
    )
    return UCCSDHamiltonianPrograms(primal, pullback, orbital_jvp, names)


def _raw_hamiltonian_primal(
    layout: UCCSDLayout, *, separate_fock: bool, identity_frame: bool
) -> tuple[Program, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(layout, UCCSDLayout):
        raise TypeError("raw unrestricted Hamiltonian requires UCCSDLayout")
    counts = {"a": layout.oa + layout.va, "b": layout.ob + layout.vb}
    occupied = {"a": layout.oa, "b": layout.ob}
    spaces = {
        spin: IndexSpace(
            f"full_{spin}", "orbital", counts[spin], "alpha" if spin == "a" else "beta"
        )
        for spin in "ab"
    }

    def spec(
        spins: str,
        symmetries: tuple[Symmetry, ...] = (),
        *,
        role: Literal["parameter", "constant"] = "parameter",
    ) -> TensorSpec:
        return TensorSpec(
            tuple(
                Index(f"axis_{axis}", spaces[spin]) for axis, spin in enumerate(spins)
            ),
            symmetries=symmetries,
            representation="spin_orbital",
            role=role,
            differentiable=role == "parameter",
        )

    rotations = {
        spin: input_tensor(f"rotation_{spin}", spec(spin * 2)) for spin in "ab"
    }
    raw_h = {
        spin: input_tensor(f"h_{spin}", spec(spin * 2, (Symmetry((1, 0)),)))
        for spin in "ab"
    }
    raw_g = {
        pair: input_tensor(
            f"g_{pair}",
            spec(
                pair[0] * 2 + pair[1] * 2,
                tuple(
                    Symmetry(axes)
                    for axes in (
                        ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1))
                        if pair[0] == pair[1]
                        else ((1, 0, 2, 3), (0, 1, 3, 2))
                    )
                ),
            ),
        )
        for pair in ("aa", "ab", "bb")
    }

    def rotate_coulomb(pair: str, value: Node) -> Node:
        if identity_frame:
            return value
        owners = pair[0] * 2 + pair[1] * 2
        transformed = einsum("up,uvwx->pvwx", rotations[owners[0]], value)
        for equation, spin in zip(
            ("vq,pvwx->pqwx", "wr,pqwx->pqrx", "xs,pqrx->pqrs"),
            owners[1:],
            strict=True,
        ):
            transformed = einsum(equation, rotations[spin], transformed)
        return transformed

    rotated_h = (
        raw_h
        if identity_frame
        else {
            spin: einsum(
                "pv,vq->pq",
                einsum("up,uv->pv", rotations[spin], raw_h[spin]),
                rotations[spin],
            )
            for spin in "ab"
        }
    )
    rotated_g = {pair: rotate_coulomb(pair, value) for pair, value in raw_g.items()}
    reference_g = rotated_g
    reference_inputs = ()
    if separate_fock:
        reference_inputs = INDEPENDENT_FOCK_INPUTS
        reference_g = {
            pair: rotate_coulomb(pair, input_tensor(f"reference_g_{pair}", value.spec))
            for pair, value in raw_g.items()
        }
    densities = {
        spin: constant(
            tuple(
                1 if row == column and row < occupied[spin] else 0
                for row in range(counts[spin])
                for column in range(counts[spin])
            ),
            spec(spin * 2, role="constant"),
        )
        for spin in "ab"
    }

    def coulomb(first: str, second: str, *, reference: bool = False) -> Node:
        integrals = reference_g if reference else rotated_g
        if first + second == "ba":
            return transpose(integrals["ab"], (2, 3, 0, 1))
        return integrals[first + second]

    def coulomb_block(
        first: str,
        second: str,
        axes: tuple[int, ...],
        ranges: tuple[tuple[int, int], ...],
        *,
        reference: bool = False,
    ) -> Node:
        """Slice before gathering axes, avoiding spurious full-frame copies."""
        integrals = reference_g if reference else rotated_g
        pair = first + second
        if pair == "ba":
            pair = "ab"
            axes = tuple((axis + 2) % 4 for axis in axes)
        source_ranges = tuple(ranges[axes.index(axis)] for axis in range(4))
        sliced = slice_tensor(integrals[pair], source_ranges)
        return transpose(sliced, axes) if axes != (0, 1, 2, 3) else sliced

    def direct_fock(spin: str, source: str) -> Node:
        if not identity_frame:
            return einsum(
                "pqrs,rs->pq", coulomb(spin, source, reference=True), densities[source]
            )
        full, occ = (0, counts[spin]), (0, occupied[source])
        block = coulomb_block(
            spin, source, (0, 1, 2, 3), (full, full, occ, occ), reference=True
        )
        return einsum("pqrr->pq", block)

    def exchange_fock(spin: str) -> Node:
        source = reference_g[spin * 2]
        if not identity_frame:
            return einsum("prqs,rs->pq", source, densities[spin], coefficient=-1)
        full, occ = (0, counts[spin]), (0, occupied[spin])
        return einsum(
            "prqr->pq", slice_tensor(source, (full, occ, full, occ)), coefficient=-1
        )

    fock = {
        spin: add(
            rotated_h[spin], *(direct_fock(spin, s) for s in "ab"), exchange_fock(spin)
        )
        for spin in "ab"
    }
    reference_energy = add(
        *(
            einsum(
                "pq,pq->",
                densities[spin],
                add(rotated_h[spin], fock[spin]),
                coefficient=Fraction(1, 2),
            )
            for spin in "ab"
        )
    )
    cc = build_program(layout.oa, layout.ob, layout.va, layout.vb)
    names = tuple(
        sorted(
            {
                node.attrs["name"]
                for node in cc.nodes
                if node.op == "input" and node.attrs["name"] not in BLOCK_NAMES
            }
        )
    )
    outputs = {}
    for name in names:
        domains, owners = name[1:].split("_")
        ranges = tuple(
            (0, occupied[spin]) if domain == "o" else (occupied[spin], counts[spin])
            for domain, spin in zip(domains, owners, strict=True)
        )
        if name.startswith("f"):
            outputs[name] = slice_tensor(fock[owners[0]], ranges)
        else:
            chemists = coulomb(owners[0], owners[1]) if not identity_frame else None
            terms, coefficients = [], []
            if owners[0] == owners[2] and owners[1] == owners[3]:
                axes = (0, 2, 1, 3)
                terms.append(
                    transpose(chemists, axes)
                    if chemists is not None
                    else coulomb_block(owners[0], owners[1], axes, ranges)
                )
                coefficients.append(1)
            if owners[0] == owners[3] and owners[1] == owners[2]:
                axes = (0, 2, 3, 1)
                terms.append(
                    transpose(chemists, axes)
                    if chemists is not None
                    else coulomb_block(owners[0], owners[1], axes, ranges)
                )
                coefficients.append(-1)
            if not terms:
                raise ValueError("canonical CC input violates spin conservation")
            signed = add(*terms, coefficients=coefficients)
            outputs[name] = signed if identity_frame else slice_tensor(signed, ranges)
    outputs["reference_electronic_energy"] = reference_energy
    primal = Program(
        outputs,
        provenance={
            "model": "unrestricted raw chemists Hamiltonian to canonical CC inputs",
            "occupied": occupied,
            "counts": counts,
            "normal_ordering": "Fock after independent spin rotations; unit occupations",
            "metric": "full Frobenius raw-input and general rotation cotangents",
            **({"frame": "exact identity"} if identity_frame else {}),
        },
    )
    return primal, names, reference_inputs


@dataclass(frozen=True)
class UCCSDHamiltonianResponsePrograms:
    """One composed CC/HF/UHF-constraint Lagrangian and its raw-input bars.

    Seed correlation energy with one, physical CC residuals with U lambda/w,
    reference electronic energy with zero or one, and Fov with the shared UHF
    adjoint. The overlap bars use general-frame metric transport dU=-dS_MO/2.
    The UHF response convention is C'=C exp(-K), K_ia=x and K_ai=-x, so its
    orbital RHS is +antisym(G_U)_ia, not RCCSD's assumed sign or spin factors.
    """

    primal: Program
    pullback: VJPProgram
    weights: Program
    equation_hash: str
    binding_hash: str


def build_lagrangian_programs(
    layout: UCCSDLayout,
    hamiltonian: UCCSDHamiltonianPrograms,
    *,
    max_elements: int = 1_000_000,
) -> UCCSDHamiltonianResponsePrograms:
    """Compose the existing CC DAG with physical inputs before one shared VJP.

    Same-shape typed views identify full-frame slices with the CC graph's
    occupied/virtual index spaces. Generic DAG rebuilding and AD accumulate
    all aliases; no CC equation, normal-ordering derivative, or dense amplitude
    Jacobian is reimplemented. This avoids a separate complete CC VJP/replay
    for each canonical independent graph input.
    """
    if not isinstance(layout, UCCSDLayout) or not isinstance(
        hamiltonian, UCCSDHamiltonianPrograms
    ):
        raise TypeError("UCCSD physical response requires a layout and Hamiltonian")
    primal, equation = _bound_lagrangian(layout, hamiltonian.primal)
    pullback = _lagrangian_pullback(
        primal, hamiltonian.pullback.input_names, max_elements
    )
    weights = _physical_weights(layout, pullback)
    return UCCSDHamiltonianResponsePrograms(
        primal, pullback, weights, equation, hamiltonian.primal.logical_hash
    )


def _bound_lagrangian(layout: UCCSDLayout, hamiltonian: Program) -> tuple[Program, str]:
    cc = build_program(layout.oa, layout.ob, layout.va, layout.vb)
    replacements = {}
    for node in cc.nodes:
        if node.op != "input" or node.attrs["name"] in BLOCK_NAMES:
            continue
        binding = hamiltonian.outputs[node.attrs["name"]]
        if binding.spec.shape != node.spec.shape:
            raise ValueError("UCCSD physical binding populations disagree")
        replacements[node] = reshape(binding, node.spec.indices)
    bound = _rebuild(cc, replacements)
    primal = Program(
        {
            **bound.outputs,
            **{
                name: hamiltonian.outputs[name]
                for name in ("reference_electronic_energy", "fov_aa", "fov_bb")
            },
        },
        provenance={
            "model": "physical unrestricted CC/HF/UHF-constraint Lagrangian",
            "cc_equation": cc.logical_hash,
            "hamiltonian_binding": hamiltonian.logical_hash,
        },
    )
    return primal, cc.logical_hash


def _lagrangian_pullback(
    primal: Program, parameters: tuple[str, ...], max_elements: int
) -> VJPProgram:
    return transpose_program(
        primal,
        (
            "correlation_energy",
            *RESIDUALS,
            "reference_electronic_energy",
            "fov_aa",
            "fov_bb",
        ),
        inputs=parameters,
        max_elements=max_elements,
    )


def build_hamiltonian_cotangent_program(
    layout: UCCSDLayout, hamiltonian: UCCSDHamiltonianPrograms
) -> Program:
    """Transport externally supplied canonical bars through the existing raw VJP.

    General rotation bars, including their symmetric part, are retained. A
    correction-only consumer seeds reference electronic energy with zero;
    adding its oo/vv Fock bars must replace the diagonal denominator chain.
    """
    return _physical_weights(layout, hamiltonian.pullback)


def _canonical_hamiltonian(
    layout: UCCSDLayout, *, separate_fock: bool, max_elements: int
) -> tuple[Program, VJPProgram, dict[str, Node]]:
    raw, names, references = _raw_hamiltonian_primal(
        layout, separate_fock=separate_fock, identity_frame=True
    )
    parameters = (*CANONICAL_HAMILTONIAN_INPUTS, *references)
    pullback = transpose_program(
        raw,
        (*names, "reference_electronic_energy"),
        inputs=parameters,
        max_elements=max_elements,
    )
    inputs = {
        node.attrs["name"]: node
        for node in raw.live_nodes
        if node.op == "input" and node.attrs["name"] in parameters
    }
    return raw, pullback, inputs


def canonical_cc_cotangents(
    primal: Program, *, max_elements: int = 1_000_000
) -> Program:
    """Retain one complete canonical CC cotangent across the orbital solve."""
    sources = tuple(
        sorted(
            node.attrs["name"]
            for node in primal.live_nodes
            if node.op == "input" and node.attrs["name"] not in BLOCK_NAMES
        )
    )
    return transpose_program(
        primal,
        ("correlation_energy", *RESIDUALS),
        inputs=sources,
        max_elements=max_elements,
    ).program


def canonical_source_programs(
    layout: UCCSDLayout, *, separate_fock: bool = False, max_elements: int = 1_000_000
) -> tuple[Program, dict[str, Program]]:
    """Partition the raw adjoint by its physical integral or one-electron owner.

    The canonical force owner evaluates an exact identity spin frame. Dense
    rotations at that point are aliases, while their complete general bars
    remain necessary for overlap transport. Each action contracts one native
    raw cotangent with its integral or one-electron field once per index.
    Only that family's full tensor and partial rotation weights are returned.
    """
    raw, pullback, inputs = _canonical_hamiltonian(
        layout, separate_fock=separate_fock, max_elements=max_elements
    )
    result = {}
    for name in pullback.input_names:
        rotations = _canonical_rotation_weights(pullback, {name: inputs[name]})
        result[name] = Program(
            {
                name: pullback.program.outputs[f"bar_{name}"],
                **{f"rotation_{spin}": value for spin, value in rotations.items()},
            },
            provenance={
                "canonical_raw_cotangent": name,
                "pullback": pullback.derivative_hash,
            },
        )
    return raw, result


def _canonical_rotation_weights(
    pullback: VJPProgram, inputs: dict[str, Node]
) -> dict[str, Node]:
    terms: dict[str, list[Node]] = {spin: [] for spin in "ab"}
    for name, source in inputs.items():
        bar = pullback.program.outputs[f"bar_{name}"]
        owners = name.rsplit("_", maxsplit=1)[1]
        axes = owners * 2 if len(owners) == 1 else owners[0] * 2 + owners[1] * 2
        indices = "pqrs"[: len(axes)]
        for slot, spin in enumerate(axes):
            source_indices = indices[:slot] + "u" + indices[slot + 1 :]
            terms[spin].append(
                einsum(f"{source_indices},{indices}->u{indices[slot]}", source, bar)
            )
    return {spin: add(*values) for spin, values in terms.items() if values}


def _physical_weights(layout: UCCSDLayout, pullback: VJPProgram) -> Program:
    """Share full-frame metric and UHF sign conventions across both pullbacks."""
    outputs = {
        name: pullback.program.outputs[f"bar_{name}"]
        for name in (
            *RAW_INPUTS,
            *(name for name in pullback.input_names if name not in RAW_INPUTS),
        )
        if not name.startswith("rotation")
    }
    rotation = {spin: pullback.program.outputs[f"bar_rotation_{spin}"] for spin in "ab"}
    outputs.update(_rotation_outputs(layout, rotation))
    return Program(
        outputs,
        provenance={
            "physical_lagrangian_pullback": pullback.derivative_hash,
            "overlap": "-sym(G_U)/2 for each spin, then accumulate physical AO binding",
            "orbital_rhs": "+antisym(G_U)_ia for shared UHF C exp(-K) convention",
        },
    )


def _rotation_outputs(
    layout: UCCSDLayout, rotation: dict[str, Node]
) -> dict[str, Node]:
    occupied = {"a": layout.oa, "b": layout.ob}
    counts = {"a": layout.oa + layout.va, "b": layout.ob + layout.vb}
    outputs = {}
    for spin in "ab":
        gradient = rotation[spin]
        stationarity = add(gradient, transpose(gradient, (1, 0)), coefficients=(1, -1))
        outputs[f"rotation_{spin}"] = gradient
        outputs[f"stationarity_{spin}"] = stationarity
        outputs[f"overlap_{spin}"] = add(
            gradient,
            transpose(gradient, (1, 0)),
            coefficients=(Fraction(-1, 4), Fraction(-1, 4)),
        )
        outputs[f"orbital_rhs_{spin}"] = slice_tensor(
            stationarity, ((0, occupied[spin]), (occupied[spin], counts[spin]))
        )
    return outputs


def canonical_rotation_transport(layout: UCCSDLayout) -> Program:
    """Project accumulated raw rotation weights to the existing overlap/RHS convention."""
    rotation = {}
    for spin, count, label in (
        ("a", layout.oa + layout.va, "alpha"),
        ("b", layout.ob + layout.vb, "beta"),
    ):
        space = IndexSpace(f"full_{spin}", "orbital", count, label)
        spec = TensorSpec(
            (Index("u", space), Index("p", space)),
            representation="spin_orbital",
            role="input",
        )
        rotation[spin] = input_tensor(f"rotation_{spin}", spec)
    outputs = _rotation_outputs(layout, rotation)
    return Program(
        {
            name: value
            for name, value in outputs.items()
            if not name.startswith("rotation_")
        },
        provenance={
            "model": "canonical unrestricted overlap and orbital RHS transport"
        },
    )
