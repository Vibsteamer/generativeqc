"""Gate 1 unit and CPU-reference parity tests for F12 (T+) host-RAM compression."""

from __future__ import annotations


import numpy as np
import pytest

from conftest import f12_host_shell
from generativeqc._f12_ram_hygiene import (
    F12_TOTAL_PANEL_STREAM_BYTES,
    SINGLE_PANEL_STREAM_BYTES,
    capture_ram_trace,
    host_memory_mib,
    release_host_memory,
    should_stream_panel,
)
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalSeries,
    MolecularFocalProvider,
    plan_singlepoint,
)
from generativeqc.profiles import canonical_hash


def _mk_basis(
    name: str,
    exps_h: tuple[float, ...],
    exps_o: tuple[float, ...] | None = None,
) -> BasisSet:
    elements = [
        ElementBasis(1, tuple(BasisShell(0, (e,), ((1.0,),)) for e in exps_h))
    ]
    if exps_o is not None:
        elements.append(
            ElementBasis(8, tuple(BasisShell(0, (e,), ((1.0,),)) for e in exps_o))
        )
    return BasisSet(
        name,
        tuple(elements),
        BasisProvenance("test", name, "uncontracted test basis", canonical_hash((exps_h, exps_o))),
        representation="spherical",
    )


def test_streaming_threshold_triggers_on_large_omol_and_jk_panels() -> None:
    # C4H6Br2O ha-D-F12 dimensions: n_ext=1237, q_f12=1162, q_jk=1017
    assert should_stream_panel(
        allow_streaming=True,
        ordinary=False,
        coulomb_only=False,
        n_ext=1237,
        n_aux=1162,
        ao_cart_total=10_000,
        env_stream="",
    )
    assert should_stream_panel(
        allow_streaming=True,
        ordinary=False,
        coulomb_only=True,
        n_ext=1237,
        n_aux=1017,
        ao_cart_total=10_000,
        env_stream="",
    )
    # C5H12O2 ha-D-F12 dimensions: n_ext=1079, q_f12=1294, q_jk=1280
    assert should_stream_panel(
        allow_streaming=True,
        ordinary=False,
        coulomb_only=False,
        n_ext=1079,
        n_aux=1294,
        ao_cart_total=10_000,
        env_stream="",
    )
    assert should_stream_panel(
        allow_streaming=True,
        ordinary=False,
        coulomb_only=True,
        n_ext=1079,
        n_aux=1280,
        ao_cart_total=10_000,
        env_stream="",
    )
    # 5-panel F12 >= 1.5 GiB triggers even when a single panel (~360 MiB) is < 512 MiB
    n_ext, n_aux = 300, 520  # 1 panel = 374.4 MiB, 5 panels = 1.83 GiB
    assert n_aux * n_ext * n_ext * 8 < SINGLE_PANEL_STREAM_BYTES
    assert 5 * n_aux * n_ext * n_ext * 8 >= F12_TOTAL_PANEL_STREAM_BYTES
    assert should_stream_panel(
        allow_streaming=True,
        ordinary=False,
        coulomb_only=False,
        n_ext=n_ext,
        n_aux=n_aux,
        ao_cart_total=10_000,
        env_stream="",
    )
    # Single-panel JK >= 512 MiB triggers streaming
    assert should_stream_panel(
        allow_streaming=True,
        ordinary=False,
        coulomb_only=True,
        n_ext=400,
        n_aux=430,  # 550.4 MiB
        ao_cart_total=10_000,
        env_stream="",
    )
    # Small H2O panel stays resident by default
    assert not should_stream_panel(
        allow_streaming=True,
        ordinary=False,
        coulomb_only=False,
        n_ext=60,
        n_aux=150,
        ao_cart_total=100_000,
        env_stream="",
    )
    # Overrides and ordinary/disallowed guards
    assert should_stream_panel(
        allow_streaming=True,
        ordinary=False,
        coulomb_only=False,
        n_ext=10,
        n_aux=10,
        ao_cart_total=10,
        env_stream="1",
    )
    assert not should_stream_panel(
        allow_streaming=True,
        ordinary=False,
        coulomb_only=False,
        n_ext=1237,
        n_aux=1162,
        ao_cart_total=2_000_000_000,
        env_stream="0",
    )
    assert not should_stream_panel(
        allow_streaming=True,
        ordinary=True,
        coulomb_only=True,
        n_ext=1237,
        n_aux=1162,
        ao_cart_total=2_000_000_000,
        env_stream="1",
    )
    assert not should_stream_panel(
        allow_streaming=False,
        ordinary=False,
        coulomb_only=False,
        n_ext=1237,
        n_aux=1162,
        ao_cart_total=2_000_000_000,
        env_stream="1",
    )


def test_release_host_memory_and_diagnostics() -> None:
    mem_before = host_memory_mib()
    assert mem_before["vm_rss_mib"] >= 0.0
    assert mem_before["vm_hwm_mib"] >= mem_before["vm_rss_mib"]
    buf = np.ones((2048, 2048), dtype=np.float64)
    assert float(buf[0, 0]) == 1.0
    del buf
    trimmed = release_host_memory()
    assert isinstance(trimmed, bool)
    mem_after = host_memory_mib()
    assert mem_after["vm_hwm_mib"] >= mem_after["vm_rss_mib"] >= 0.0


@pytest.mark.parametrize(
    ("root_id", "charge", "mult", "corr_space", "mp2_mode", "asym_weights"),
    [
        ("rhf", 0, 1, "all-active", "f12", False),
        ("rhf", 0, 1, "frozen-core", "f12", False),
        ("rhf", 0, 1, "all-active", "jk", False),
        ("rhf", 0, 1, "all-active", "separate", False),
        ("rhf", 0, 1, "all-active", "coincident_jk_f12", False),
        ("rhf", 0, 1, "all-active", "f12", True),
        ("uhf", 1, 2, "all-active", "f12", True),
        ("uhf", 1, 2, "frozen-core", "f12", True),
        ("uhf", 1, 2, "all-active", "separate", True),
    ],
)
def test_h2o_tplus_scaling_forces_streaming_parity(
    monkeypatch: pytest.MonkeyPatch,
    root_id: str,
    charge: int,
    mult: int,
    corr_space: str,
    mp2_mode: str,
    asym_weights: bool,
) -> None:
    shell = f12_host_shell()
    obs = _mk_basis("obs", (1.5, 0.32), (18.0, 2.8, 0.55))
    cabs = _mk_basis("cabs", (0.11,), (0.15,))
    jk = _mk_basis("jk", (4.0, 0.9, 0.2), (24.0, 5.0, 1.2, 0.3))
    f12 = jk if mp2_mode == "coincident_jk_f12" else _mk_basis("f12", (5.0, 1.1, 0.25), (30.0, 6.0, 1.5, 0.35))
    mp2_map = {
        "f12": f12,
        "jk": jk,
        "separate": _mk_basis("mp2", (4.5, 1.0, 0.22), (27.0, 5.5, 1.3, 0.32)),
        "coincident_jk_f12": f12,
    }
    card = FocalBasis(2, "m", obs, jk_fit=jk, mp2_fit=mp2_map[mp2_mode], f12_fit=f12, cabs=cabs, beta=0.9)
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.43, 1.10)), ("H", (0.15, -1.43, 1.10))]
    steps = plan_singlepoint(
        atoms,
        FocalSeries(card),
        FocalSeries(card),
        card,
        charge=charge,
        multiplicity=mult,
        root_id=root_id,
        correlation_space_id=corr_space,
        backend="cpu_reference",
    ).steps
    req = next(r for r, _ in steps if r.role == "mp2_f12_correlation")

    monkeypatch.setenv("GENERATIVEQC_F12_STREAM_PANELS", "0")
    p_ref = MolecularFocalProvider(shell_library=shell)
    low = p_ref.low_order_energy(req, include_hf_cabs=False, include_occupied_partitions=True)
    assert low.occupied_mp2_f12 is not None
    oa = len(low.occupied_mp2_f12.mp2[0])
    ob = len(low.occupied_mp2_f12.mp2[1])
    wa = np.linspace(1.1, 1.4, oa)
    wb = np.linspace(1.15, 1.5, ob) if asym_weights else wa.copy()
    f12_w = (wa * 0.8, wb * 0.75 if asym_weights else wa * 0.8)
    forces_ref = p_ref.tplus_scaling_forces(
        req, low.reference, (wa, wb), f12_w, expected_cabs_rank=low.cabs_rank
    )

    monkeypatch.setenv("GENERATIVEQC_F12_STREAM_PANELS", "1")
    p_stream = MolecularFocalProvider(shell_library=shell)
    with capture_ram_trace() as trace:
        forces_stream = p_stream.tplus_scaling_forces(
            req, low.reference, (wa, wb), f12_w, expected_cabs_rank=low.cabs_rank
        )
    np.testing.assert_allclose(forces_stream, forces_ref, atol=1e-13, rtol=0)
    assert "entry" not in p_stream._pair_source_cache
    phases = [str(row["phase"]) for row in trace]
    assert "after_cached_pair_sources" in phases
    assert "after_ordinary_mp2" in phases
    assert "after_f12_increment" in phases
    assert "after_1e_grad" in phases
    if mp2_mode != "coincident_jk_f12":
        assert "after_jk_forward" in phases
        assert "after_jk_reverse" in phases


def test_empty_beta_uhf_and_low_order_forces_streaming_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = f12_host_shell()
    obs = _mk_basis("obs", (1.5, 0.32, 0.08))
    cabs = _mk_basis("cabs", (0.04,))
    jk = _mk_basis("jk", (4.0, 0.9, 0.2, 0.05))
    f12 = _mk_basis("f12", (5.0, 1.1, 0.25, 0.06))
    card = FocalBasis(2, "m", obs, jk_fit=jk, mp2_fit=f12, f12_fit=f12, cabs=cabs, beta=0.9)
    # H3+ triplet: 2 alpha electrons, 0 beta electrons -> exercises ob == 0 branch
    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (1.6, 0.1, -0.2)), ("H", (0.5, 1.5, 0.2))]
    steps = plan_singlepoint(
        atoms,
        FocalSeries(card),
        FocalSeries(card),
        card,
        charge=1,
        multiplicity=3,
        root_id="uhf",
        correlation_space_id="all-active",
        backend="cpu_reference",
    ).steps
    req = next(r for r, _ in steps if r.role == "mp2_f12_correlation")

    monkeypatch.setenv("GENERATIVEQC_F12_STREAM_PANELS", "0")
    p_ref = MolecularFocalProvider(shell_library=shell)
    res_ref = p_ref.low_order_energy_and_forces(req, include_hf_cabs=False)

    monkeypatch.setenv("GENERATIVEQC_F12_STREAM_PANELS", "1")
    p_stream = MolecularFocalProvider(shell_library=shell)
    res_stream = p_stream.low_order_energy_and_forces(req, include_hf_cabs=False)

    assert res_stream.energies.mp2_f12_correlation_energy == pytest.approx(
        res_ref.energies.mp2_f12_correlation_energy, abs=1e-13, rel=0
    )
    np.testing.assert_allclose(
        res_stream.mp2_f12_correlation_forces,
        res_ref.mp2_f12_correlation_forces,
        atol=1e-13,
        rtol=0,
    )


def test_h2o_low_order_to_tplus_scaling_upgrade_and_fd_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = f12_host_shell()
    obs = _mk_basis("obs", (1.5, 0.32), (18.0, 2.8, 0.55))
    cabs = _mk_basis("cabs", (0.11,), (0.15,))
    jk = _mk_basis("jk", (4.0, 0.9, 0.2), (24.0, 5.0, 1.2, 0.3))
    f12 = _mk_basis("f12", (5.0, 1.1, 0.25), (30.0, 6.0, 1.5, 0.35))
    card = FocalBasis(2, "m", obs, jk_fit=jk, mp2_fit=f12, f12_fit=f12, cabs=cabs, beta=0.9)
    base_coords = np.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.10], [0.15, -1.43, 1.10]], dtype=np.float64)
    syms = ("O", "H", "H")

    def make_req(coords: np.ndarray):
        atoms = [(s, tuple(float(v) for v in p)) for s, p in zip(syms, coords, strict=True)]
        steps = plan_singlepoint(
            atoms,
            FocalSeries(card),
            FocalSeries(card),
            card,
            charge=0,
            multiplicity=1,
            root_id="rhf",
            correlation_space_id="frozen-core",
            backend="cpu_reference",
        ).steps
        return next(r for r, _ in steps if r.role == "mp2_f12_correlation")

    req0 = make_req(base_coords)
    monkeypatch.setenv("GENERATIVEQC_F12_STREAM_PANELS", "1")
    prov = MolecularFocalProvider(shell_library=shell)
    # Reuse the same provider across low_order_energy (derivatives=False) and tplus_scaling_forces (derivatives=True)
    low0 = prov.low_order_energy(req0, include_hf_cabs=False, include_occupied_partitions=True)
    assert low0.occupied_mp2_f12 is not None
    o_act = len(low0.occupied_mp2_f12.mp2[0])
    w_mp2 = np.linspace(1.1, 1.4, o_act)
    w_f12 = np.linspace(0.8, 1.2, o_act)
    analytic = prov.tplus_scaling_forces(
        req0,
        low0.reference,
        (w_mp2, w_mp2),
        (w_f12, w_f12),
        expected_cabs_rank=low0.cabs_rank,
    )
    # Translation invariance
    assert float(np.max(np.abs(np.sum(analytic, axis=0)))) < 1e-12

    # Independent central finite-difference oracle on one displaced coordinate
    h = 5e-5
    atom_idx, axis = 1, 1

    def weighted_energy(coords: np.ndarray) -> float:
        low = prov.low_order_energy(
            make_req(coords),
            expected_cabs_rank=low0.cabs_rank,
            include_hf_cabs=False,
            include_occupied_partitions=True,
        )
        assert low.occupied_mp2_f12 is not None
        e_mp2 = float(np.dot(w_mp2, low.occupied_mp2_f12.mp2[0] + low.occupied_mp2_f12.mp2[1]))
        e_f12 = float(
            np.dot(
                w_f12,
                low.occupied_mp2_f12.f12_increment[0] + low.occupied_mp2_f12.f12_increment[1],
            )
        )
        return e_mp2 + e_f12

    cp = base_coords.copy()
    cm = base_coords.copy()
    cp[atom_idx, axis] += h
    cm[atom_idx, axis] -= h
    fd_force = -(weighted_energy(cp) - weighted_energy(cm)) / (2.0 * h)
    assert abs(analytic[atom_idx, axis] - fd_force) < 1e-8
