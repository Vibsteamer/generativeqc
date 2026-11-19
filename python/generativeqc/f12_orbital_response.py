"""Coupled unrestricted orbital response for real, canonical DF-UHF F12 forces."""

import ctypes
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
from generativeqc_compiler.method.canonical_frame_response import (
    orbital_metric_cotangent,
    orbital_rotation_cotangent,
)
from generativeqc_compiler.mp2.f12_unrestricted_reference import (
    unrestricted_df_canonical_driver,
    unrestricted_df_nuclear_direction,
    unrestricted_df_nuclear_weights,
)

from . import _native
from ._canonical_frame import canonical_full_frame_seeds
from .elements import checked_integer


class UnrestrictedOrbitalForces(NamedTuple):
    rotations: np.ndarray
    orbital_gradient: np.ndarray
    forces: np.ndarray


def _block(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != shape
        or array.dtype != np.float64
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must contain finite FP64 values of shape {shape}")
    return array.ravel(order="C")


class UnrestrictedNuclearSourceTerms(NamedTuple):
    nuclear_rhs: np.ndarray
    fixed_gradients: dict[str, np.ndarray]


def _nuclear_center(
    central: Mapping[str, np.ndarray], oa: int, ob: int
) -> dict[str, np.ndarray]:
    fields = {"whitener": central["whitener"]}
    for spin, occ in (("alpha", oa), ("beta", ob)):
        for key in ("coefficients", "df", "fock_seed", "overlap_seed"):
            name = f"{key}_{spin}"
            fields[name] = central[name]
        if occ:
            name = f"occupied_energies_{spin}"
            fields[name] = central[name]
    return fields


def unrestricted_frame_overlap_weights(
    central: Mapping[str, np.ndarray],
    complement: np.ndarray,
    direct: Mapping[str, np.ndarray],
    run: Callable,
) -> dict[str, np.ndarray]:
    """Derive the two AO overlap cotangents from the matching spin-MO/CABS frames."""
    cabs = np.asarray(complement)
    if cabs.ndim != 2 or not cabs.shape[1] or cabs.dtype != np.float64 or not np.isfinite(cabs).all():
        raise ValueError("Expected finite FP64 CABS columns")
    atomic, rank = cabs.shape
    weights = {}
    for spin in ("alpha", "beta"):
        basis = np.asarray(central["coefficients_" + spin])
        if basis.ndim != 2 or basis.shape[0] != atomic or not basis.shape[1] or basis.dtype != np.float64 or not np.isfinite(basis).all():
            raise ValueError("OBS and CABS coefficients must share the same AO rows")
        obs = basis.shape[1]
        retained = obs + rank
        bar = np.asarray(direct["coefficients_" + spin])
        _block(bar, (atomic, retained), "extended coefficient cotangent")
        # Relative gauge of bar and CABS columns is fixed by the forward phase convention.
        coefficients = np.column_stack((basis, cabs))
        mask = np.full((retained, retained), -0.5)
        mask[:obs, obs:] = -1.0
        mask[obs:, :obs] = 0.0
        weight = run(orbital_metric_cotangent(atomic, retained), {
            "coefficients": coefficients, "frame_bar": bar, "metric_blocks": mask,
        }).outputs["overlap_bar"]
        _block(weight, (atomic, atomic), "frame overlap cotangent")
        weights[spin] = weight
    return weights


def unrestricted_canonical_source_seeds(
    occupations: tuple[int, int],
    coefficients: Mapping[str, np.ndarray],
    energies: Mapping[str, np.ndarray],
    frame_cotangents: Mapping[str, np.ndarray],
    run: Callable,
    *,
    gap_threshold: float = 1e-8,
    stationarity_threshold: float = 1e-9,
    maximum_bytes: int = 128 << 20,
    device_id: int = 0,
    library: ctypes.CDLL | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Build central Fock and overlap weights from the two spin frame cotangents.

    Supply matching OBS columns, canonical energies and AO coefficient cotangents
    from one reference. A cotangent with complete OBS+CABS columns is accepted;
    the separate CABS frame response consumes the complement columns. The full
    OBS seeds include occupied/virtual couplings with the relaxed DF-UHF Fock.
    The caller applies orbital response only once through its coupled equation.
    """
    result, torques = {}, {}
    for spin, occ in zip(("alpha", "beta"), occupations, strict=True):
        c = np.asarray(coefficients["coefficients_" + spin])
        e = np.asarray(energies[spin])
        bar = np.asarray(frame_cotangents["coefficients_" + spin])
        n = e.size
        if (
            c.ndim != 2
            or c.shape[1] != n
            or e.shape != (n,)
            or bar.ndim != 2
            or bar.shape[0] != c.shape[0]
            or bar.shape[1] < n
            or any(v.dtype != np.float64 for v in (c, e, bar))
            or not np.isfinite(e).all()
        ):
            raise ValueError(
                "Inconsistent canonical orbitals, energies or AO frame cotangents"
            )
        graph = orbital_rotation_cotangent(c.shape[0], n)
        m = run(
            graph, {"coefficients": c, "frame_bar": np.ascontiguousarray(bar[:, :n])}
        ).outputs["rotation_gradient"]
        f, s, t, _ = canonical_full_frame_seeds(
            m,
            e,
            occ,
            gap_threshold=gap_threshold,
            stationarity_threshold=stationarity_threshold,
            maximum_bytes=maximum_bytes,
            device_id=device_id,
            library=library,
        )
        result["fock_seed_" + spin] = f
        result["overlap_seed_" + spin] = s
        torques[spin] = t
    return result, torques


def unrestricted_nuclear_source_terms(
    space: tuple[int, int, int, int],
    central: Mapping[str, np.ndarray],
    directions: Iterable[Mapping[str, np.ndarray]],
    atoms: int,
    run: Callable,
) -> UnrestrictedNuclearSourceTerms:
    """Build DF-UHF orbital equations and six fixed terms on the supplied backend.

    Iterate over the raw AO tangents in atom-major x,y,z order. Each direction
    supplies doverlap, dhcore, dthree_center and dmetric. Both spin frames use
    the same central metric whitener and AO basis. The dense central df blocks
    must use that whitener. Fock and overlap seeds are in the corresponding
    central MO frame. No orbital derivative or preassembled nuclear RHS enters.
    """
    oa, ob, n, q = space
    n_atoms = checked_integer(atoms, "atoms", low=1)
    coeff = np.asarray(central["coefficients_alpha"])
    if coeff.ndim != 2:
        raise ValueError("Expected an AO-by-OBS alpha coefficient matrix")
    program = unrestricted_df_nuclear_direction(oa, ob, n, q, coeff.shape[0])
    keys = ("dmetric", "dthree_center", "doverlap", "dhcore")
    spin_pairs = (("alpha", oa), ("beta", ob))
    fields = _nuclear_center(central, oa, ob)
    rhs = np.zeros((n_atoms, 3, oa * (n - oa) + ob * (n - ob)))
    fixed = {
        f"{key}_{spin}": np.empty((n_atoms, 3))
        for spin, _ in spin_pairs
        for key in ("fixed_fock", "metric_density", "fixed_overlap")
    }
    for (atom, axis), tangent in zip(np.ndindex(n_atoms, 3), directions, strict=True):
        inputs = {**fields, **{key: tangent[key] for key in keys}}
        result = run(program, inputs).outputs
        rhs[atom, axis] = np.concatenate(
            [result["rhs_" + spin].ravel() for spin, occ in spin_pairs if occ]
        )
        for key, channel in fixed.items():
            channel[atom, axis] = result[key].item()
    if not np.isfinite(rhs).all() or any(
        not np.isfinite(a).all() for a in fixed.values()
    ):
        raise ValueError("Nonfinite DF-UHF nuclear equation or fixed gradient")
    return UnrestrictedNuclearSourceTerms(rhs, fixed)


def unrestricted_nuclear_source_weights(
    space: tuple[int, int, int, int],
    central: Mapping[str, np.ndarray],
    orbital_adjoint: np.ndarray,
    run: Callable,
    *,
    direct_weights: Mapping[str, np.ndarray] | None = None,
    frame_overlap: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Prepare orbital-response gradient weights for the Gaussian nuclear sinks.

    Each returned AO/metric array contracts with a derivative of the same
    operator. Add these four arrays to the direct energy weights; separately
    include the OBS/CABS frame overlap response. The adjoint and the central
    seeds must describe the same electronic observable and SCF root. This
    operation uses only central quantities and produces every atomic direction's
    weights in one reverse evaluation. To return effective weights for these four
    operators, supply their direct energy cotangents and both spin frame-overlap
    cotangents. The other Gaussian operator weights pass unchanged to their sinks.
    """
    oa, ob, n, q = space
    coeff = np.asarray(central["coefficients_alpha"])
    if coeff.ndim != 2:
        raise ValueError("Expected an AO-by-OBS alpha coefficient matrix")
    include_direct = direct_weights is not None
    if include_direct != (frame_overlap is not None):
        raise ValueError(
            "Direct Gaussian and frame-overlap weights must be supplied together"
        )
    graph = unrestricted_df_nuclear_weights(
        oa, ob, n, q, coeff.shape[0], include_direct=include_direct
    )
    z = _block(orbital_adjoint, (oa * (n - oa) + ob * (n - ob),), "orbital adjoint")
    fields = _nuclear_center(central, oa, ob)
    if direct_weights is not None and frame_overlap is not None:
        for name in ("overlap_ao", "hcore_ao", "ao_coulomb", "coulomb_metric"):
            fields["direct_" + name] = direct_weights[
                "metric/coulomb" if name == "coulomb_metric" else name
            ]
        for spin in ("alpha", "beta"):
            fields["frame_" + spin] = frame_overlap[spin]
    offset = 0
    for spin, occ in (("alpha", oa), ("beta", ob)):
        for key in ("fixed_fock", "metric_density", "fixed_overlap"):
            fields[f"bar_{key}_{spin}"] = np.array(1.0)
        if occ:
            size = (n - occ) * occ
            fields["bar_rhs_" + spin] = z[offset : offset + size].reshape(n - occ, occ)
            offset += size
    weights = dict(run(graph, fields).outputs)
    weights["metric/coulomb"] = weights.pop("coulomb_metric")
    if any(not np.isfinite(w).all() for w in weights.values()):
        raise ValueError("Nonfinite DF-UHF nuclear source weights")
    return weights


def unrestricted_orbital_forces(
    space: tuple[int, int, int, int],
    factors: Mapping[str, np.ndarray],
    electronic_driver: np.ndarray,
    nuclear_rhs: np.ndarray,
    reference_forces: np.ndarray,
    fixed_gradients: Sequence[np.ndarray] = (),
    *,
    device_id: int = 0,
    maximum_bytes: int = 1536 << 20,
    library: ctypes.CDLL | None = None,
) -> UnrestrictedOrbitalForces:
    """Solve Hᵀz=g and return F_ref − Σ fixed gradients − b_Rᵀz on CUDA.

    Space contains alpha/beta occupied counts, OBS orbitals and the shared DF
    rank. Supply vo, oo, vv and gap for each occupied spin under names such as
    vo_alpha. Use the same Coulomb whitening for both spins. Flatten the
    electronic driver in alpha-then-beta virtual-major, occupied-minor order.
    Nuclear right-hand sides have shape (atoms, 3, rotations). They solve
    H x_R=b_R. The fixed-frame energy gradients must exclude this rotation
    response. Reference forces and each fixed gradient have shape (atoms, 3).
    The returned orbital gradient uses the positive energy derivative sign.

    maximum_bytes bounds explicit CUDA arrays including pivot and LU workspaces.
    """
    if np.asarray(reference_forces).size == 0:
        raise ValueError("Expected nonempty reference forces")
    return _solve_orbital_response(space, factors, electronic_driver, nuclear_rhs, reference_forces,
                                   fixed_gradients, device_id, maximum_bytes, library)


def unrestricted_source_orbital_adjoint(
    space: tuple[int, int, int, int],
    central: Mapping[str, np.ndarray],
    canonical_energies: Mapping[str, np.ndarray],
    run: Callable,
    *,
    device_id: int = 0,
    maximum_bytes: int = 1536 << 20,
    library: ctypes.CDLL | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the canonical electronic driver and solve its transposed orbital system.

    The spin DF tensors and canonical Fock cotangents must use the same MO frame
    and shared auxiliary metric. Returns alpha-then-beta VO driver and rotations.
    This solve does not consume nuclear RHS, forces or an external orbital adjoint.
    """
    oa, ob, n, q = space
    graph = unrestricted_df_canonical_driver(*space)
    values, factors = {}, {}
    for spin, occ in (("alpha", oa), ("beta", ob)):
        if not occ:
            continue
        df = np.asarray(central["df_" + spin])
        seed = np.asarray(central["fock_seed_" + spin])
        eps = _block(canonical_energies[spin], (n,), "canonical energies")
        if df.shape != (q, n, n) or seed.shape != (n, n):
            raise ValueError("Canonical driver and fitted MO space disagree")
        values["df_" + spin] = df
        values["fock_seed_" + spin] = seed
        factors.update({key + "_" + spin: block for key, block in (
            ("vo", df[:, occ:, :occ]), ("oo", df[:, :occ, :occ]), ("vv", df[:, occ:, occ:]),
            ("gap", eps[occ:, None] - eps[None, :occ]))})
    channels = run(graph, values).outputs
    driver = np.concatenate([_block(channels[spin], (n - occ, occ), "orbital driver")
                             for spin, occ in (("alpha", oa), ("beta", ob)) if occ])
    # Zero coordinates select the rotations-only ABI mode. No nuclear arrays are uploaded.
    result = _solve_orbital_response(space, factors, driver, np.empty((0, 3, driver.size)),
                                     np.empty((0, 3)), (), device_id, maximum_bytes, library)
    return driver, result.rotations


def _solve_orbital_response(space, factors, electronic_driver, nuclear_rhs, reference_forces,
                            fixed_gradients, device_id, maximum_bytes, library):
    oa, ob, n, q = (
        checked_integer(value, name)
        for value, name in zip(
            space,
            ("alpha occupation", "beta occupation", "OBS orbitals", "DF rank"),
            strict=True,
        )
    )
    if not oa + ob or n <= max(oa, ob) or q < 1:
        raise ValueError(
            "Expected occupied electrons, OBS virtuals and a nonempty fitting basis"
        )
    device = checked_integer(device_id, "CUDA device")
    budget = checked_integer(
        maximum_bytes, "workspace bytes", low=1, high=np.iinfo(np.uintp).max
    )
    rank = checked_integer(oa * (n - oa) + ob * (n - ob), "orbital rotations", low=1)
    ref = np.asarray(reference_forces)
    if ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError("Reference forces must have three Cartesian axes")
    directions = checked_integer(ref.size, "Cartesian directions")
    terms = checked_integer(len(fixed_gradients), "fixed gradient terms")
    if 8 * rank * rank > budget:
        raise ValueError(
            "The dense orbital Hessian exceeds the device workspace budget"
        )
    packed = []
    for spin, o in (("alpha", oa), ("beta", ob)):
        if not o:
            continue
        v = n - o
        for key, shape in (
            ("vo", (q, v, o)),
            ("oo", (q, o, o)),
            ("vv", (q, v, v)),
            ("gap", (v, o)),
        ):
            name = f"{key}_{spin}"
            packed.append(_block(factors[name], shape, name))
    packed.append(_block(electronic_driver, (rank,), "electronic driver"))
    packed.append(_block(nuclear_rhs, (*ref.shape, rank), "nuclear RHS"))
    packed.append(_block(ref, ref.shape, "reference forces"))
    packed.extend(_block(g, ref.shape, "fixed gradient") for g in fixed_gradients)
    if (sum(a.size for a in packed) + rank * rank + directions) * 8 + (
        rank + 1
    ) * 4 > budget:
        raise ValueError(
            "Molecular adjoint inputs and Hessian exceed the device workspace budget"
        )
    data = np.concatenate(packed)
    if library is None:
        library = _native.load_library(device="cuda", device_id=device)
    routine = library.generativeqc_uhf_orbital_adjoint_cuda_v1
    pointer = ctypes.POINTER(ctypes.c_double)
    routine.argtypes = [ctypes.c_int] * 7 + [
        pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        pointer,
        pointer,
        pointer,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    routine.restype = ctypes.c_int
    z, gradient, forces = np.empty(rank), np.empty(ref.shape), np.empty(ref.shape)
    error = ctypes.create_string_buffer(512)
    status = routine(
        device,
        oa,
        ob,
        n,
        q,
        directions,
        terms,
        data.ctypes.data_as(pointer),
        data.size,
        budget,
        z.ctypes.data_as(pointer),
        gradient.ctypes.data_as(pointer),
        forces.ctypes.data_as(pointer),
        error,
        len(error),
    )
    if status:
        raise RuntimeError(error.value.decode(errors="replace"))
    return UnrestrictedOrbitalForces(z, gradient, forces)
