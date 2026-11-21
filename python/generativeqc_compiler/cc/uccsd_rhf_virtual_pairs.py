"""Reapply the exact restricted virtual Coulomb ladder with packed pair matrices."""

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cache

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    JVPProgram,
    Node,
    Program,
    TensorSpec,
    VJPProgram,
    add,
    einsum,
    indexed_gather,
    input_tensor,
    linearize,
    reshape,
    scatter_add,
    transpose_program,
)

from .uccsd_equations import _canonical_tau
from .uccsd_layout import BLOCK_NAMES, UCCSDLayout


@cache
def virtual_pair_coordinates(
    extent: int, *, diagonal: bool
) -> tuple[tuple[int, int], ...]:
    """Ordered representatives shared by Coulomb preparation and repeated actions."""
    offset = 0 if diagonal else 1
    return tuple((i, j) for i in range(extent) for j in range(i + offset, extent))


@dataclass(frozen=True)
class _Pairs:
    index: Index
    full_index: Index
    direct: tuple[int, ...]
    exchange: tuple[int, ...]
    reflection: tuple[int, ...]

    @classmethod
    def create(cls, extent: int, axis: str, *, diagonal: bool) -> "_Pairs":
        pairs = virtual_pair_coordinates(extent, diagonal=diagonal)
        direct = tuple(i * extent + j for i, j in pairs)
        exchange = tuple(j * extent + i for i, j in pairs)
        reflection = ()
        if diagonal:
            locations = {pair: k for k, pair in enumerate(pairs)}
            reflection = tuple(
                locations[min(i, j), max(i, j)]
                for i in range(extent)
                for j in range(extent)
            )
        return cls(
            Index(axis, IndexSpace(axis, "pair", len(pairs))),
            Index(axis, IndexSpace("ordered_" + axis, "pair", extent * extent)),
            direct,
            exchange,
            reflection,
        )

    def pack(self, tensor: Node, axis: int) -> Node:
        direct = indexed_gather(tensor, axis, self.direct, self.index)
        reflection = indexed_gather(tensor, axis, self.exchange, self.index)
        half = Fraction(1, 2)
        return add(
            direct, reflection, coefficients=(half, half if self.reflection else -half)
        )

    def expand(self, tensor: Node, axis: int) -> Node:
        if self.reflection:
            return indexed_gather(tensor, axis, self.reflection, self.full_index)
        direct = scatter_add(tensor, axis, self.direct, self.full_index)
        reflection = scatter_add(tensor, axis, self.exchange, self.full_index)
        return add(direct, reflection, coefficients=(1, -1))

    def operator(self, name: str) -> Node:
        # Orient source pair first. The action is a rectangular GEMM
        # across all spectator occupied pairs.
        return input_tensor(
            name,
            TensorSpec(
                (
                    replace(self.index, name="source"),
                    replace(self.index, name="target"),
                ),
                representation="spin_orbital",
                role="input",
            ),
        )


@dataclass(frozen=True)
class UCCRHFVirtualPairPrograms:
    """A repeated bare ladder with fixed spatial operators for physical T actions."""

    primal: Program
    residual_jvp: JVPProgram
    residual_vjp: VJPProgram


def build_uccsd_rhf_virtual_pair_programs(
    layout: UCCSDLayout, *, max_elements: int = 1_000_000
) -> UCCRHFVirtualPairPrograms:
    """Evaluate the RHF bare ladder with symmetric and antisymmetric virtual pairs.

    The caller supplies the same canonical RHF spatial operator to all spin
    sectors. In the nonorthonormal pair representation the operator columns
    fold the ordered source pairs onto their representatives. The TensorIR
    packing and expansion provide the correct transpose under the physical
    amplitude maps. The independent DF factor reverse supplies source weights
    once after stationarity, where the metric and nuclear derivative are active.
    """
    if not isinstance(layout, UCCSDLayout):
        raise TypeError("RHF virtual pairs require UCCSDLayout")
    if (layout.oa, layout.va) != (layout.ob, layout.vb):
        raise ValueError("RHF virtual pairs require equal spin populations")
    if layout.oa < 1 or layout.va < 2:
        raise ValueError(
            "RHF virtual pairs require nonempty occupied and virtual pairs"
        )
    occupied = _Pairs.create(layout.oa, "occupied_pair", diagonal=False)
    symmetric = _Pairs.create(layout.va, "virtual_symmetric", diagonal=True)
    antisymmetric = _Pairs.create(layout.va, "virtual_antisymmetric", diagonal=False)
    plus = symmetric.operator("spatial_coulomb_symmetric")
    minus = antisymmetric.operator("spatial_coulomb_antisymmetric")
    tau_aa, tau_ab, tau_bb = _canonical_tau(layout)

    def same_spin(tau: Node) -> Node:
        matrix = reshape(tau, (occupied.full_index, antisymmetric.full_index))
        block = antisymmetric.pack(occupied.pack(matrix, 0), 1)
        result = einsum("oe,ea->oa", block, minus)
        return reshape(
            occupied.expand(antisymmetric.expand(result, 1), 0), tau.spec.indices
        )

    # Opposite spin keeps both virtual parity channels and all ordered
    # occupied spectators. Any mixed-spin pair relation is unnecessary.
    mixed = reshape(tau_ab, (occupied.full_index, symmetric.full_index))
    plus_result = symmetric.expand(
        einsum("oe,ea->oa", symmetric.pack(mixed, 1), plus), 1
    )
    mixed = reshape(tau_ab, (occupied.full_index, antisymmetric.full_index))
    minus_result = antisymmetric.expand(
        einsum("oe,ea->oa", antisymmetric.pack(mixed, 1), minus), 1
    )
    primal = Program(
        {
            "r2aa": same_spin(tau_aa),
            "r2ab": add(
                reshape(plus_result, tau_ab.spec.indices),
                reshape(minus_result, tau_ab.spec.indices),
            ),
            "r2bb": same_spin(tau_bb),
        },
        provenance={
            "method": "UCCSD",
            "term": "bare_rhf_spatial_virtual_pairs",
            "version": 1,
        },
    )
    residuals = tuple(primal.outputs)
    return UCCRHFVirtualPairPrograms(
        primal,
        linearize(primal, BLOCK_NAMES, outputs=residuals, max_elements=max_elements),
        transpose_program(
            primal, residuals, inputs=BLOCK_NAMES, max_elements=max_elements
        ),
    )
