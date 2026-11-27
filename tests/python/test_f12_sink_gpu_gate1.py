"""Gate 1 verification for GPU-fused spherical F12 sink, fused fixed forces, and SlaterFitNormalization.

Verifies:
1. Direct spherical forward & weighted-derivative kernels match legacy Cartesian
   ShellProjection pack/unpack + Cartesian kernel ABI across all 5 F12 operators
   (metric & 3-center AO, including asymmetric cotangent weights) to < 1e-13.
2. Bounded auxiliary-slab chunking produces identical results across slab sizes
   and raises explicit budget errors when the memory cap is infeasible. Admission
   is monotone in the budget: a rejection reports the single-shell minimum, and
   every budget at or above it is admitted.
3. Configurable SlaterFitNormalization:
   - Default ``negative_exponential`` preserves legacy baseline bit-for-bit (< 1e-14).
   - ``negative_exponential_over_beta`` scales linear geminal channels by 1/beta and
     quadratic channels by 1/beta^2 (< 1e-13), equals ``negative_exponential`` at beta=1.0
     (< 1e-14), and passes full-3N analytic vs two-step FD nuclear force checks (< 2e-8).
   - Independent Psi4 1.11 oracle on H2O (cc-pCVDZ-F12) and HCl (cc-pVDZ-F12) at
     beta = 0.9, 1.0, 1.4 confirms default ``negative_exponential`` matches Psi4 1.11
     and ``negative_exponential_over_beta`` eliminates the beta = 1.4 core-valence
     sign inversion (+0.94 mEh on H2O, +101 mEh on HCl).
4. Fused + streamed ``fixed_ucc_source_forces`` matches unfused standalone
   ``unrestricted_cc_{scalar,w,c}_ao_weights`` to < 1e-12 Eh/bohr for RHF & UHF
   (both all-active and frozen-core).
"""

from __future__ import annotations

import ctypes as ct
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from conftest import f12_host_shell

if TYPE_CHECKING:
    from collections.abc import Callable

from generativeqc import Atom, BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc._f12_focal_integral_kernel import (
    _OPCODES,
    bind,
    bind_spherical_forward,
    bind_spherical_weighted,
    channel_name,
    contract_geometry,
    slater_normalization_code,
    spherical_layout_descriptor,
)
from generativeqc._f12_focal_integral_sink import (
    ShellProjection,
    contract_live_weights,
    contract_single_channel,
)
from generativeqc._f12_focal_integrals import (
    bind_forward,
    focal_shell_layout,
    forward_integrals,
    forward_single_ao_operator,
)
from generativeqc._f12_focal_shells import spherical_projections
from generativeqc.elements import atomic_number
from generativeqc.experimental.f12_focal import (
    DEFAULT_SLATER_FIT_NORMALIZATION,
    FocalBasis,
    FocalSeries,
    plan_singlepoint,
)


@pytest.fixture(scope="module")
def host_sink() -> Path:
    return f12_host_shell()


def _compact_water_basis(beta: float = 1.4) -> FocalBasis:
    """Compact spherical orbital/CABS/JKFit/F12Fit basis with s, p, d, f shells."""
    def make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
        prov = BasisProvenance(source=name, version="1", license="MIT", checksum="0" * 64)
        return BasisSet(name, elems, prov, representation="spherical")

    def sh(l: int, exps: tuple[float, ...], coeffs: tuple[float, ...]) -> BasisShell:
        return BasisShell(l, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),))

    o_orb = ElementBasis(
        8,
        (
            sh(0, (130.70932, 23.808861, 6.4436083), (0.15432897, 0.53532814, 0.44463454)),
            sh(0, (5.0331513, 1.1695961), (-0.09996723, 0.39951283)),
            sh(0, (0.3803890,), (1.0,)),
            sh(1, (5.0331513, 1.1695961), (0.15591627, 0.60768372)),
            sh(1, (0.3803890,), (1.0,)),
        ),
    )
    h_orb = ElementBasis(
        1,
        (
            sh(0, (3.42525091, 0.62391373), (0.15432897, 0.53532814)),
            sh(0, (0.16885540,), (1.0,)),
        ),
    )
    o_cabs = ElementBasis(8, (sh(1, (1.4,), (1.0,)), sh(2, (1.2,), (1.0,))))
    h_cabs = ElementBasis(1, (sh(1, (0.9,), (1.0,)),))
    o_jk = ElementBasis(
        8,
        (
            sh(0, (50.0,), (1.0,)),
            sh(0, (12.0,), (1.0,)),
            sh(0, (3.0,), (1.0,)),
            sh(0, (0.8,), (1.0,)),
            sh(1, (6.0,), (1.0,)),
            sh(1, (1.5,), (1.0,)),
            sh(2, (2.2,), (1.0,)),
        ),
    )
    h_jk = ElementBasis(
        1,
        (
            sh(0, (4.5,), (1.0,)),
            sh(0, (1.1,), (1.0,)),
            sh(1, (1.2,), (1.0,)),
        ),
    )
    o_f12 = ElementBasis(
        8,
        (
            sh(0, (36.0,), (1.0,)),
            sh(0, (8.0,), (1.0,)),
            sh(0, (1.8,), (1.0,)),
            sh(1, (4.5,), (1.0,)),
            sh(1, (1.1,), (1.0,)),
            sh(2, (1.9,), (1.0,)),
            sh(3, (1.3,), (1.0,)),
        ),
    )
    h_f12 = ElementBasis(
        1,
        (
            sh(0, (3.8,), (1.0,)),
            sh(0, (0.9,), (1.0,)),
            sh(1, (1.1,), (1.0,)),
            sh(2, (0.8,), (1.0,)),
        ),
    )
    return FocalBasis(
        2,
        "gate1-f12sink-water",
        make_basis("orb", (o_orb, h_orb)),
        jk_fit=make_basis("jk", (o_jk, h_jk)),
        mp2_fit=make_basis("f12", (o_f12, h_f12)),
        f12_fit=make_basis("f12", (o_f12, h_f12)),
        cabs=make_basis("cabs", (o_cabs, h_cabs)),
        beta=beta,
    )


def _water_atoms(scale: float = 1.0) -> tuple[Atom, ...]:
    return (
        Atom.from_value(("O", (0.0, 0.0, 0.0))),
        Atom.from_value(("H", (0.0, 1.43042809 * scale, -1.10715266))),
        Atom.from_value(("H", (0.0, -1.43042809, -1.10715266 * scale))),
    )


def _pair_request(
    atoms: tuple[Atom, ...],
    basis: FocalBasis,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    root_id: str = "rhf",
    correlation_space_id: str = "frozen-core",
    slater_normalization: str = DEFAULT_SLATER_FIT_NORMALIZATION,
):
    series = FocalSeries(basis)
    plan = plan_singlepoint(
        atoms,
        series,
        series,
        None,
        charge=charge,
        multiplicity=multiplicity,
        root_id=root_id,
        correlation_space_id=correlation_space_id,
        protocol_id="gate1-f12sink",
        backend="cpu_reference",
        slater_normalization=slater_normalization,
    )
    return next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")


def _legacy_cartesian_forward(layout: dict, library: Path, beta: float) -> dict[str, np.ndarray]:
    """Evaluate all 10 forward tensors strictly via the legacy Cartesian ABI + CPU ShellProjection."""
    funcs = {k: bind_forward(library, k, True, extended=False) for k in ("coulomb", "geminal")}
    out = {}
    budget = 512 << 20
    for group in ("metric", "ao"):
        proj = ShellProjection(layout, group)
        blocks = np.ascontiguousarray(proj.blocks, dtype=np.int32)
        offsets = np.ascontiguousarray(proj.offsets, dtype=np.int32)
        nblocks, ncenters = blocks.shape
        ncart = int(offsets[-1])
        for op, opcode in _OPCODES.items():
            label = channel_name(group, op)
            raw = np.full(ncart, np.nan)
            ws = ct.c_uint64()
            rc = funcs["coulomb" if op == "coulomb" else "geminal"](
                ncenters,
                opcode,
                0.0 if op == "coulomb" else beta,
                len(layout["angular"]),
                len(layout["exponents"]),
                nblocks,
                ncart,
                layout["angular"],
                layout["prim_offsets"],
                layout["exponents"],
                layout["radial"],
                layout["centers"],
                blocks,
                offsets,
                raw,
                budget,
                ct.byref(ws),
            )
            assert rc == 0
            out[label] = proj.spherical_values(raw)
    return out


def _legacy_cartesian_weighted(
    layout: dict, weights: dict[str, np.ndarray], library: Path
) -> dict[str, tuple[float, np.ndarray]]:
    """Evaluate all 10 nuclear gradients strictly via the legacy Cartesian ABI + CPU ShellProjection."""
    import json

    manifest = json.loads(str(layout["manifest"]))
    funcs = {k: bind(library, k, True, extended=False) for k in ("coulomb", "geminal")}
    budget = 512 << 20
    out = {}
    for group in ("metric", "ao"):
        proj = ShellProjection(layout, group)
        for op in _OPCODES:
            label = channel_name(group, op)
            packed = proj.pack(weights[label])
            _, row = contract_geometry(
                {
                    **layout,
                    group + "_offsets": np.ascontiguousarray(proj.offsets, dtype=np.int32),
                    "weights_" + label: packed,
                },
                manifest,
                funcs,
                group,
                op,
                budget,
                True,
            )
            out[label] = (
                float(row["weighted_value_Eh"]),
                np.asarray(row["gradient_Eh_per_bohr"], dtype=np.float64),
            )
    return out


def test_spherical_direct_matches_cartesian_legacy_forward_and_weighted(host_sink: Path) -> None:
    """Direct spherical forward/weighted ABI matches legacy Cartesian ABI to < 1e-13."""
    basis = _compact_water_basis(beta=1.2)
    req = _pair_request(_water_atoms(scale=1.03), basis)
    layout = focal_shell_layout(req, host_sink, host_reference=True, budget_mib=512)

    # Confirm the new direct spherical symbols are bound and active.
    assert bind_spherical_forward(host_sink, "coulomb", True) is not None
    assert bind_spherical_forward(host_sink, "geminal", True) is not None
    assert bind_spherical_weighted(host_sink, "coulomb", True) is not None
    assert bind_spherical_weighted(host_sink, "geminal", True) is not None
    assert spherical_layout_descriptor(layout, "metric") is not None
    assert spherical_layout_descriptor(layout, "ao") is not None

    sph_fwd, resources = forward_integrals(req, host_sink, host_reference=True, budget_mib=512)
    leg_fwd = _legacy_cartesian_forward(layout, host_sink, beta=1.2)
    assert set(sph_fwd) == set(leg_fwd)
    for label, ref in leg_fwd.items():
        np.testing.assert_allclose(sph_fwd[label], ref, atol=1e-13, rtol=1e-13, err_msg=label)
        assert resources[label]["shell_blocks"] > 0

    # Asymmetric random weights verify diagonal sa==sb vs off-diagonal sa>sb folding exactly.
    rng = np.random.default_rng(20261123)
    weights = {k: rng.normal(scale=0.1, size=v.shape) for k, v in sph_fwd.items()}
    sph_res = contract_live_weights(layout, weights, host_sink, host_reference=True, budget_mib=512)
    assert sph_res["passes_contraction_checks"]
    leg_res = _legacy_cartesian_weighted(layout, weights, host_sink)
    for label, (ref_val, ref_grad) in leg_res.items():
        row = sph_res["rows"][label]
        assert abs(row["weighted_value_Eh"] - ref_val) < 1e-12, label
        np.testing.assert_allclose(
            row["gradient_Eh_per_bohr"], ref_grad, atol=1e-12, rtol=1e-12, err_msg=label
        )
        assert row["host_packing_seconds"] == 0.0
        # Also verify contract_single_channel matches identically
        group = "metric" if label.startswith("metric") else "ao"
        op = "coulomb" if label == "metric" else label.removeprefix(group + "_")
        g_single = contract_single_channel(
            layout, group, op, weights[label], host_sink, host_reference=True, budget_mib=512
        )
        np.testing.assert_allclose(g_single, ref_grad, atol=1e-12, rtol=1e-12, err_msg=label)


def test_spherical_slab_budget_chunking_parity(host_sink: Path) -> None:
    """Tight budget forcing multi-slab execution matches single-slab execution to < 1e-13."""
    basis = _compact_water_basis(beta=1.1)
    req = _pair_request(_water_atoms(), basis)
    layout = focal_shell_layout(req, host_sink, host_reference=True, budget_mib=512)

    # Full single-slab (budget=512 MiB) vs tight multi-slab (budget=2 MiB -> multiple aux slabs)
    full_ao = forward_single_ao_operator(
        layout, host_sink, "f12_squared", host_reference=True, budget_mib=512
    )
    slab_ao = forward_single_ao_operator(
        layout, host_sink, "f12_squared", host_reference=True, budget_mib=2
    )
    np.testing.assert_allclose(slab_ao, full_ao, atol=1e-14, rtol=0)

    rng = np.random.default_rng(77)
    w = rng.normal(scale=0.05, size=full_ao.shape)
    g_full = contract_single_channel(
        layout, "ao", "f12_squared", w, host_sink, host_reference=True, budget_mib=512
    )
    g_slab = contract_single_channel(
        layout, "ao", "f12_squared", w, host_sink, host_reference=True, budget_mib=2
    )
    np.testing.assert_allclose(g_slab, g_full, atol=1e-13, rtol=1e-13)

    # Non-finite weights must fail loudly.
    bad = w.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        contract_single_channel(
            layout, "ao", "f12_squared", bad, host_sink, host_reference=True, budget_mib=512
        )


def _synthetic_slab_layout(library: Path) -> dict:
    """Three-center layout whose widest spherical and Cartesian slabs differ.

    A run of s shells widens both slab directions equally, while the following
    g shells widen the Cartesian direction 15/9 as fast as the spherical one.
    """
    centers = np.array([[0.0, 0.0, 0.0], [0.0, 1.1, 1.3]])
    orbital = [
        (0, 0, (3.0, 0.6), (0.4, 0.7)),
        (0, 1, (0.9,), (1.0,)),
        (0, 2, (0.8,), (1.0,)),
        (1, 0, (1.2,), (1.0,)),
        (1, 1, (0.7,), (1.0,)),
    ]
    fitting = [(i % 2, 0, (0.5 + 0.1 * i,), (1.0,)) for i in range(24)]
    fitting += [(i % 2, 4, (0.9 + 0.05 * i,), (1.0,)) for i in range(8)]
    shells = orbital + fitting
    angular = np.array([shell[1] for shell in shells], dtype=np.int32)
    atom = np.array([shell[0] for shell in shells], dtype=np.int32)
    rows = np.full((len(shells), 2 * int(angular.max()) + 1), -1, dtype=np.int32)
    sizes = []
    for start, end in ((0, len(orbital)), (len(orbital), len(shells))):
        offset = 0
        for index in range(start, end):
            width = 2 * int(angular[index]) + 1
            rows[index, :width] = np.arange(offset, offset + width)
            offset += width
        sizes.append(offset)
    pairs = np.stack(np.tril_indices(len(orbital)), axis=1)
    blocks = np.empty((len(fitting), len(pairs), 3), dtype=np.int32)
    blocks[:, :, :2] = pairs
    blocks[:, :, 2] = np.arange(len(orbital), len(shells))[:, None]
    blocks = blocks.reshape(-1, 3)
    counts = np.prod((angular[blocks] + 1) * (angular[blocks] + 2) // 2, axis=1)
    layout = {
        "angular": angular,
        "atom": atom,
        "centers": np.ascontiguousarray(centers[atom]),
        "prim_offsets": np.r_[0, np.cumsum([len(s[2]) for s in shells])].astype(
            np.int32
        ),
        "exponents": np.concatenate([s[2] for s in shells]).astype(np.float64),
        "radial": np.concatenate([s[3] for s in shells]).astype(np.float64),
        "spherical_rows": rows,
        "orbital_shell_count": np.int32(len(orbital)),
        "ao_sizes": np.array(sizes, dtype=np.int32),
        "ao_blocks": blocks,
        "ao_offsets": np.r_[0, np.cumsum(counts)].astype(np.int32),
    }
    for l, transform in spherical_projections(library, set(angular.tolist())).items():
        layout[f"transform_{l}"] = transform
    return layout


def _legacy_ao_coulomb(layout: dict, library: Path) -> np.ndarray:
    """Unslabbed Cartesian three-center Coulomb values projected by ShellProjection."""
    projection = ShellProjection(layout, "ao")
    blocks = np.ascontiguousarray(projection.blocks, dtype=np.int32)
    offsets = np.ascontiguousarray(projection.offsets, dtype=np.int32)
    raw = np.full(int(offsets[-1]), np.nan)
    used = ct.c_uint64()
    rc = bind_forward(library, "coulomb", True, extended=False)(
        3,
        _OPCODES["coulomb"],
        0.0,
        len(layout["angular"]),
        len(layout["exponents"]),
        len(blocks),
        len(raw),
        layout["angular"],
        layout["prim_offsets"],
        layout["exponents"],
        layout["radial"],
        layout["centers"],
        blocks,
        offsets,
        raw,
        512 << 20,
        ct.byref(used),
    )
    assert rc == 0
    return projection.spherical_values(raw)


def test_spherical_slab_admission_is_monotone_in_the_budget(host_sink: Path) -> None:
    """Every budget at or above the single-shell minimum is admitted with unchanged results."""
    layout = _synthetic_slab_layout(host_sink)
    desc = spherical_layout_descriptor(layout, "ao")
    assert desc is not None
    forward = bind_spherical_forward(host_sink, "coulomb", True)
    weighted = bind_spherical_weighted(host_sink, "coulomb", True)
    reference = _legacy_ao_coulomb(layout, host_sink)
    weights = np.random.default_rng(20261127).normal(scale=0.1, size=reference.shape)
    norm = slater_normalization_code("negative_exponential_over_beta")
    natoms = 2
    shells = (desc["nshell"], desc["orbital_shell_count"], len(layout["exponents"]))
    tables = (
        layout["prim_offsets"],
        layout["exponents"],
        layout["radial"],
        layout["centers"],
        desc["sph_offsets"],
        desc["transform_table"],
    )

    def run_forward(budget: int) -> tuple[int, int, np.ndarray]:
        values = np.full(reference.shape, np.nan)
        used = ct.c_uint64()
        rc = forward(
            3,
            0,
            0.0,
            norm,
            *shells,
            desc["norb_sph"],
            desc["naux_sph"],
            layout["angular"],
            *tables,
            values,
            budget,
            ct.byref(used),
        )
        return rc, used.value, values

    def run_weighted(budget: int) -> tuple[int, int, tuple[float, np.ndarray]]:
        value = np.full(1, np.nan)
        gradient = np.full(natoms * 3, np.nan)
        used = ct.c_uint64()
        rc = weighted(
            3,
            0,
            0.0,
            norm,
            *shells,
            natoms,
            desc["norb_sph"],
            desc["naux_sph"],
            layout["angular"],
            layout["atom"],
            *tables,
            weights,
            value,
            gradient,
            budget,
            ct.byref(used),
        )
        return rc, used.value, (float(value[0]), gradient.reshape(natoms, 3))

    def sweep(run: Callable[[int], tuple[int, int, Any]]) -> tuple[Any, list[Any]]:
        rc, minimum, _ = run(1)
        assert rc == -2
        rc, single_slab, unslabbed = run(1 << 30)
        assert rc == 0 and minimum < single_slab
        grid = np.geomspace(minimum, 2 * single_slab, 64).astype(np.int64).tolist()
        admitted = []
        for budget in sorted({minimum - 1, minimum, *grid}):
            rc, used, result = run(int(budget))
            if budget < minimum:
                assert (rc, used) == (-2, minimum), budget
            else:
                assert rc == 0 and 0 < used <= budget, (budget, rc, used)
                admitted.append(result)
        return unslabbed, admitted

    _, forward_results = sweep(run_forward)
    for values in forward_results:
        np.testing.assert_allclose(values, reference, atol=1e-13, rtol=1e-13)

    (_, unslabbed_gradient), weighted_results = sweep(run_weighted)
    expected = float(np.sum(weights * reference))
    for value, gradient in weighted_results:
        assert value == pytest.approx(expected, rel=1e-12, abs=1e-14)
        np.testing.assert_allclose(gradient, unslabbed_gradient, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(unslabbed_gradient.sum(axis=0), 0.0, atol=1e-12)








@pytest.mark.parametrize("oa,ob,frozen", [(3, 3, (1, 1)), (3, 2, (1, 0))])
def test_native_pair_pullbacks_match_tensorir_exact(
    host_sink: Path, oa: int, ob: int, frozen: tuple[int, int]
) -> None:
    """Native C++/cuBLAS W, C, and Scalar MO pullbacks match TensorIR to < 1e-12."""
    from functools import partial

    from generativeqc._f12_cc_c_nuclear import _accumulate_c_mo_pullback
    from generativeqc._f12_cc_pair_pullback_native import bind_pair_pullback
    from generativeqc._f12_cc_scalar_nuclear import _accumulate_scalar_mo_pullback
    from generativeqc._f12_cc_w_nuclear import _accumulate_w_mo_pullback
    from generativeqc_compiler.tensor import execute

    for kind in ("w", "c", "scalar"):
        assert bind_pair_pullback(host_sink, kind, True) is not None

    rng = np.random.default_rng(100 + oa * 7 + ob)
    obs, ext, aux = 6, 10, 8
    space = (oa, ob, obs, ext, aux)

    def sym(a: np.ndarray) -> np.ndarray:
        return 0.5 * (a + a.swapaxes(-1, -2))

    same = oa == ob and frozen[0] == frozen[1]
    fa = sym(rng.normal(scale=0.3, size=(ext, ext)))
    ka = sym(rng.normal(scale=0.2, size=(ext, ext)))
    fock = {
        "fock_alpha": fa,
        "fock_beta": fa.copy() if same else sym(rng.normal(scale=0.3, size=(ext, ext))),
        "exchange_alpha": ka,
        "exchange_beta": ka.copy() if same else sym(rng.normal(scale=0.2, size=(ext, ext))),
    }
    fit_a = rng.normal(scale=0.2, size=(aux, obs, ext))
    fit_obs = {
        "alpha": fit_a,
        "beta": fit_a.copy() if same else rng.normal(scale=0.2, size=(aux, obs, ext)),
    }
    rc_a = rng.normal(scale=0.2, size=(aux, obs, ext))
    co = sym(rng.normal(scale=0.2, size=(ext, ext))) if same else rng.normal(scale=0.2, size=(ext, ext))
    data = {
        "cross_overlap": co,
        "source/fit_alpha": fit_obs["alpha"][:, :oa, :],
        "source/fit_beta": fit_obs["beta"][:, :ob, :],
        "source/raw_coulomb_alpha": rc_a,
        "source/raw_coulomb_beta": rc_a.copy() if same else rng.normal(scale=0.2, size=(aux, obs, ext)),
        "metric/geminal": sym(rng.normal(scale=0.2, size=(aux, aux))),
        "metric/geminal_coulomb": sym(rng.normal(scale=0.2, size=(aux, aux))),
        "metric/squared_geminal": sym(rng.normal(scale=0.2, size=(aux, aux))),
        "metric/double_commutator": sym(rng.normal(scale=0.2, size=(aux, aux))),
    }
    for op in ("geminal", "geminal_coulomb", "squared_geminal", "double_commutator"):
        ra = rng.normal(scale=0.2, size=(aux, oa, ext))
        data[f"raw/{op}/alpha"] = ra
        data[f"raw/{op}/beta"] = ra.copy() if same else rng.normal(scale=0.2, size=(aux, ob, ext))

    cpu_run = partial(execute, max_bytes=512 << 20)
    dbar_ref = {k: np.zeros_like(v) for k, v in data.items()}
    dbar_nat = {k: np.zeros_like(v) for k, v in data.items()}
    fbar_ref = {k: np.zeros_like(v) for k, v in fock.items()}
    fbar_nat = {k: np.zeros_like(v) for k, v in fock.items()}
    fobs_ref = {s: np.zeros_like(v) for s, v in fit_obs.items()}
    fobs_nat = {s: np.zeros_like(v) for s, v in fit_obs.items()}

    e_ref = _accumulate_scalar_mo_pullback(
        space, data, fock, 1.25, cpu_run, dbar_ref, fbar_ref, frozen_occupied=frozen, native_library=None
    )
    e_nat = _accumulate_scalar_mo_pullback(
        space, data, fock, 1.25, cpu_run, dbar_nat, fbar_nat, frozen_occupied=frozen, native_library=host_sink, host_reference=True
    )
    np.testing.assert_allclose(e_nat, e_ref, atol=1e-13, rtol=1e-13)

    w_aa = rng.normal(scale=0.1, size=(oa - frozen[0], oa - frozen[0], obs - oa, oa - frozen[0]))
    w_weights = {
        "w_vo_aaaa": w_aa,
        "w_vo_bbbb": w_aa.copy() if same else rng.normal(scale=0.1, size=(ob - frozen[1], ob - frozen[1], obs - ob, ob - frozen[1])),
        "w_vv_abab": rng.normal(scale=0.1, size=(oa - frozen[0], ob - frozen[1], obs - oa, obs - ob)),
    }
    _accumulate_w_mo_pullback(
        space, data, fit_obs, w_weights, cpu_run, dbar_ref, fobs_ref, frozen_occupied=frozen, native_library=None
    )
    _accumulate_w_mo_pullback(
        space, data, fit_obs, w_weights, cpu_run, dbar_nat, fobs_nat, frozen_occupied=frozen, native_library=host_sink, host_reference=True
    )

    c_aa = rng.normal(scale=0.1, size=(oa - frozen[0], oa - frozen[0], obs - oa, obs - oa))
    c_weights = {
        "c_f12_aaaa": c_aa,
        "c_f12_bbbb": c_aa.copy() if same else rng.normal(scale=0.1, size=(ob - frozen[1], ob - frozen[1], obs - ob, obs - ob)),
        "c_f12_abab": rng.normal(scale=0.1, size=(oa - frozen[0], ob - frozen[1], obs - oa, obs - ob)),
    }
    _accumulate_c_mo_pullback(
        space, data, fock, c_weights, cpu_run, dbar_ref, fbar_ref, frozen_occupied=frozen, native_library=None
    )
    _accumulate_c_mo_pullback(
        space, data, fock, c_weights, cpu_run, dbar_nat, fbar_nat, frozen_occupied=frozen, native_library=host_sink, host_reference=True
    )

    for k in dbar_ref:
        np.testing.assert_allclose(dbar_nat[k], dbar_ref[k], atol=1e-12, rtol=1e-12, err_msg=k)
    for k in fbar_ref:
        np.testing.assert_allclose(fbar_nat[k], fbar_ref[k], atol=1e-12, rtol=1e-12, err_msg=k)
    for k in fobs_ref:
        np.testing.assert_allclose(fobs_nat[k], fobs_ref[k], atol=1e-12, rtol=1e-12, err_msg=k)
