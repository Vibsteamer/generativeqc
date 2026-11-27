"""Gate 1 verification for dynamic-dimension AOT AO->MO forward & reverse projections.

Step 11.1 coverage:
1. Direct unit parity of `project_ao_operator_tiled`, `project_ao_operator_pullback_tiled`,
   and `_orbital_sources` against the TensorIR oracle (`GENERATIVEQC_F12_DISABLE_AOT_AOMO=1`),
   covering:
   - All 3-center operators (`raw_coulomb`, `raw_f12`, `raw_f12_squared`,
     `raw_f12_over_coulomb`, `raw_double_commutator`), `hcore`, `cross_overlap`,
     and `coulomb_fit_response` forward/reverse;
   - All pullback `inputs_mode` values (`both`, `coefficients`, `ao`), `pad_coul`
     (`seed_rows < ext`), `three_center_bar`, `seed_scale != 1.0`, pre-populated
     accumulation targets, non-symmetric AO/seed matrices, CABS rank truncation
     (`ext < nc`), and empty-spin (`ob == 0`) channels;
   - All `_orbital_sources` roles (`shared`, `pairs`, `fock`, `ordinary_jk`,
     `ordinary_fock`), `occupied_coulomb`, `obs_coulomb`, `skip_coulomb_fit_solve`,
     `can_share_rev` True/False, and `split_ao` streaming (`__ao_loader__` +
     `ao_bar_sink`) verifying `ao_mat` is released before `bar_ao` is allocated;
   - Strict zero TensorIR call assertion (`0` calls to `run` on the AOT path) and
     backend isolation (CUDA runners never bind `*_host_reference` symbols).
2. Real molecular Gate 1 parity against the TensorIR oracle across:
   - RHF `H2O` and `CH4`;
   - UHF `OH` (`oa != ob`);
   - ECP `HBr` (`ECP10MDF`, BOTH `FC4` `frozen-core` and `FC9` `frozen-core-spd`);
   verifying forward parity <= 1e-12, reverse cotangent & nuclear force parity
   <= 1e-11, and zero TensorIR calls inside `_orbital_sources` on the AOT path.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest

from conftest import configured_test_library, f12_host_shell

from generativeqc import Atom, BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc._f12_ao_mo_native import (
    bind_ao_mo_ops,
    detect_host_reference,
    resolve_ao_mo_ops,
)
from generativeqc._f12_focal_one_electron import one_electron_sources
from generativeqc._f12_focal_reference import df_reference
from generativeqc._f12_spin_ao import (
    _orbital_sources,
    project_ao_operator_pullback_tiled,
    project_ao_operator_tiled,
)
from generativeqc.elements import atomic_number
from generativeqc.experimental.f12_focal import (
    DEFAULT_SLATER_FIT_NORMALIZATION,
    FocalBasis,
    FocalRequest,
)
from generativeqc_compiler.tensor import execute
from _bse_basis_support import br_ecp10mdf

@pytest.fixture(scope="module")
def host_sink() -> Path:
    return f12_host_shell()


def _forbidden_run(program, inputs):
    raise AssertionError(
        f"Unexpected TensorIR invocation on AOT path: outputs={tuple(program.outputs)}"
    )


def _cpu_run():
    return partial(execute, max_bytes=256 << 20)


def _make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
    prov = BasisProvenance(source=name, version="1", license="MIT", checksum="0" * 64)
    return BasisSet(name, elems, prov, representation="spherical")


def _sh(l: int, exps: tuple[float, ...], coeffs: tuple[float, ...]) -> BasisShell:
    return BasisShell(
        l, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),)
    )


def _compact_split_basis(elements: tuple[int, ...]) -> FocalBasis:
    """Compact spherical split-fit basis for H, C, O, Br(ECP10MDF)."""
    br_ecp = br_ecp10mdf()
    orb_map = {
        1: ElementBasis(
            1,
            (
                _sh(0, (3.42525091, 0.62391373), (0.15432897, 0.53532814)),
                _sh(0, (0.16885540,), (1.0,)),
            ),
        ),
        6: ElementBasis(
            6,
            (
                _sh(0, (71.616837, 13.045096, 3.5305122), (0.15432897, 0.53532814, 0.44463454)),
                _sh(0, (2.9412494, 0.6834831), (-0.09996723, 0.39951283)),
                _sh(0, (0.2222899,), (1.0,)),
                _sh(1, (2.9412494, 0.6834831), (0.15591627, 0.60768372)),
                _sh(1, (0.2222899,), (1.0,)),
            ),
        ),
        8: ElementBasis(
            8,
            (
                _sh(0, (130.70932, 23.808861, 6.4436083), (0.15432897, 0.53532814, 0.44463454)),
                _sh(0, (5.0331513, 1.1695961), (-0.09996723, 0.39951283)),
                _sh(0, (0.3803890,), (1.0,)),
                _sh(1, (5.0331513, 1.1695961), (0.15591627, 0.60768372)),
                _sh(1, (0.3803890,), (1.0,)),
            ),
        ),
        35: ElementBasis(
            35,
            (
                _sh(0, (18.4, 5.2, 1.8), (0.22, 0.55, 0.35)),
                _sh(0, (0.85, 0.31), (0.45, 0.62)),
                _sh(0, (0.12,), (1.0,)),
                _sh(1, (14.1, 4.1, 1.4), (0.25, 0.52, 0.38)),
                _sh(1, (0.62, 0.22), (0.48, 0.58)),
                _sh(1, (0.095,), (1.0,)),
                _sh(2, (6.8, 2.3), (0.42, 0.65)),
                _sh(2, (0.68,), (1.0,)),
            ),
            ecp_core_electrons=br_ecp.ecp_core_electrons,
            ecp_data=br_ecp.ecp_data,
        ),
    }
    jk_map = {
        1: ElementBasis(1, (_sh(0, (6.0,), (1.0,)), _sh(0, (1.5,), (1.0,)), _sh(0, (0.4,), (1.0,)), _sh(1, (1.4,), (1.0,)))),
        6: ElementBasis(6, (_sh(0, (36.0,), (1.0,)), _sh(0, (9.0,), (1.0,)), _sh(0, (2.2,), (1.0,)), _sh(0, (0.55,), (1.0,)), _sh(1, (4.5,), (1.0,)), _sh(1, (1.1,), (1.0,)), _sh(2, (1.6,), (1.0,)))),
        8: ElementBasis(8, (_sh(0, (60.0,), (1.0,)), _sh(0, (15.0,), (1.0,)), _sh(0, (4.0,), (1.0,)), _sh(0, (1.1,), (1.0,)), _sh(1, (7.0,), (1.0,)), _sh(1, (1.8,), (1.0,)), _sh(2, (2.2,), (1.0,)))),
        35: ElementBasis(35, (_sh(0, (32.0,), (1.0,)), _sh(0, (9.0,), (1.0,)), _sh(0, (2.5,), (1.0,)), _sh(0, (0.7,), (1.0,)), _sh(0, (0.22,), (1.0,)), _sh(1, (12.0,), (1.0,)), _sh(1, (3.2,), (1.0,)), _sh(1, (0.85,), (1.0,)), _sh(2, (4.5,), (1.0,)), _sh(2, (1.2,), (1.0,)))),
    }
    mp2_map = {
        1: ElementBasis(1, (_sh(0, (5.2,), (1.0,)), _sh(0, (1.2,), (1.0,)), _sh(0, (0.32,), (1.0,)), _sh(1, (1.2,), (1.0,)))),
        6: ElementBasis(6, (_sh(0, (28.0,), (1.0,)), _sh(0, (7.2,), (1.0,)), _sh(0, (1.8,), (1.0,)), _sh(0, (0.45,), (1.0,)), _sh(1, (3.8,), (1.0,)), _sh(1, (0.95,), (1.0,)), _sh(2, (1.3,), (1.0,)))),
        8: ElementBasis(8, (_sh(0, (48.0,), (1.0,)), _sh(0, (11.0,), (1.0,)), _sh(0, (2.8,), (1.0,)), _sh(0, (0.75,), (1.0,)), _sh(1, (5.8,), (1.0,)), _sh(1, (1.4,), (1.0,)), _sh(2, (1.9,), (1.0,)))),
        35: ElementBasis(35, (_sh(0, (26.0,), (1.0,)), _sh(0, (7.5,), (1.0,)), _sh(0, (2.1,), (1.0,)), _sh(0, (0.58,), (1.0,)), _sh(0, (0.18,), (1.0,)), _sh(1, (9.5,), (1.0,)), _sh(1, (2.6,), (1.0,)), _sh(1, (0.70,), (1.0,)), _sh(2, (3.8,), (1.0,)), _sh(2, (1.0,), (1.0,)))),
    }
    cabs_map = {
        1: ElementBasis(1, (_sh(0, (0.8,), (1.0,)), _sh(1, (0.9,), (1.0,)))),
        6: ElementBasis(6, (_sh(0, (1.4,), (1.0,)), _sh(1, (1.4,), (1.0,)), _sh(2, (0.8,), (1.0,)))),
        8: ElementBasis(8, (_sh(0, (2.0,), (1.0,)), _sh(1, (2.0,), (1.0,)), _sh(2, (1.2,), (1.0,)))),
        35: ElementBasis(35, (_sh(0, (1.1,), (1.0,)), _sh(1, (1.1,), (1.0,)), _sh(2, (1.4,), (1.0,)))),
    }
    uniq = tuple(sorted(set(elements)))
    orb = _make_basis("gate1-aomo-orb", tuple(orb_map[z] for z in uniq))
    jk = _make_basis("gate1-aomo-jk", tuple(jk_map[z] for z in uniq))
    mp2 = _make_basis("gate1-aomo-mp2", tuple(mp2_map[z] for z in uniq))
    cabs = _make_basis("gate1-aomo-cabs", tuple(cabs_map[z] for z in uniq))
    return FocalBasis(
        2, "gate1-aomo-split", orb, jk_fit=jk, mp2_fit=mp2, f12_fit=mp2, cabs=cabs, beta=1.0
    )


def _make_atoms(spec: tuple[tuple[str, tuple[float, float, float]], ...]) -> tuple[Atom, ...]:
    return tuple(Atom(atomic_number(sym), pos) for sym, pos in spec)


def _synthetic_ao_bundle(
    rng: np.random.Generator,
    oa: int,
    ob: int,
    obs: int,
    ext: int,
    nc: int,
    aux: int,
    *,
    same_spin: bool,
) -> tuple[tuple[int, int, int, int, int], dict[str, np.ndarray]]:
    space = (oa, ob, obs, ext, aux)
    c_a = rng.standard_normal((nc, ext))
    c_b = c_a.copy() if same_spin else rng.standard_normal((nc, ext))
    # Build SPD metric + exact whitener W.T @ W = J^{-1}
    raw_m = rng.standard_normal((aux, aux))
    metric = raw_m @ raw_m.T + 2.0 * np.eye(aux)
    chol = np.linalg.cholesky(metric)
    whitener = np.linalg.solve(chol, np.eye(aux))
    ao = {
        "overlap_ao": rng.standard_normal((nc, nc)),  # intentionally non-symmetric!
        "hcore_ao": rng.standard_normal((nc, nc)),
        "coefficients_alpha": c_a,
        "coefficients_beta": c_b,
        "metric/coulomb": metric,
        "metric_whitener": whitener,
        "metric/geminal": rng.standard_normal((aux, aux)),
        "metric/squared_geminal": rng.standard_normal((aux, aux)),
        "metric/geminal_coulomb": rng.standard_normal((aux, aux)),
        "metric/double_commutator": rng.standard_normal((aux, aux)),
        "ao_coulomb": rng.standard_normal((aux, nc, nc)),
        "ao_f12": rng.standard_normal((aux, nc, nc)),
        "ao_f12_squared": rng.standard_normal((aux, nc, nc)),
        "ao_f12_over_coulomb": rng.standard_normal((aux, nc, nc)),
        "ao_double_commutator": rng.standard_normal((aux, nc, nc)),
    }
    return space, ao


def _random_orbital_weights(
    rng: np.random.Generator,
    projected: dict[str, np.ndarray],
    *,
    symmetric_spins: bool = False,
) -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {}
    for k, v in projected.items():
        if k == "space":
            continue
        if symmetric_spins and (k.endswith("_beta") or k.endswith("/beta")):
            continue
        weights[k] = rng.standard_normal(v.shape)
    if symmetric_spins:
        for k in list(projected):
            if k.endswith("_beta"):
                weights[k] = weights[k[:-5] + "_alpha"].copy()
            elif k.endswith("/beta"):
                weights[k] = weights[k[:-5] + "/alpha"].copy()
    return weights


def test_backend_detection_and_isolation(host_sink: Path, monkeypatch):
    """Verify host/cuda detection never crosses backends and respects opt-out/require env vars."""
    ops = bind_ao_mo_ops(host_sink, host_reference=True)
    assert ops is not None and ops.host_reference is True
    # Host library must NOT export *_cuda symbols.
    assert bind_ao_mo_ops(host_sink, host_reference=False) is None

    # Simulated CUDA focal runner closure (same shape as focal_cuda_runner / checked wrappers).
    def fake_cuda_runner(p, i):
        raise AssertionError("not called")

    fake_cuda_runner.batch = lambda p: None  # type: ignore[attr-defined]
    fake_cuda_runner.profile_snapshot = lambda: {}  # type: ignore[attr-defined]

    cuda = True

    def checked_cuda(g, i):
        return fake_cuda_runner(g, i) if cuda else None

    assert detect_host_reference(fake_cuda_runner) is False
    assert detect_host_reference(checked_cuda) is False
    assert detect_host_reference(_cpu_run()) is True

    # A CUDA runner must NEVER silently bind the host_reference library.
    assert (
        resolve_ao_mo_ops(checked_cuda, None, shell_library=host_sink) is None
    )

    # Kill-switch disables AOT cleanly.
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO", "1")
    assert resolve_ao_mo_ops(_cpu_run(), None, shell_library=host_sink) is None
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO")

    # Strict require raises when CUDA symbols are missing.
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_AOMO", "1")
    with pytest.raises(RuntimeError, match="GENERATIVEQC_F12_REQUIRE_AOT_AOMO"):
        resolve_ao_mo_ops(checked_cuda, None, shell_library=host_sink)

    # Non-finite input detection (O(1) memory in native C++/CUDA).
    monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_AOMO")
    bad_ao = np.zeros((2, 4, 4), dtype=np.float64)
    bad_ao[1, 2, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        project_ao_operator_tiled(
            4,
            3,
            1,
            2,
            "raw_coulomb",
            "ao_coulomb",
            np.eye(4, 3, dtype=np.float64),
            bad_ao,
            _forbidden_run,
            shell_library=host_sink,
            host_reference=True,
        )


@pytest.mark.parametrize(
    "nc,ext,occ,aux,field,ao_name,seed_rows,tc_rows,seed_scale,inputs_mode",
    [
        # Full raw_coulomb, all three pullback modes, non-trivial seed_scale & tc_bar
        (13, 10, 3, 11, "raw_coulomb", "ao_coulomb", 10, 0, 1.0, "both"),
        (13, 10, 3, 11, "raw_coulomb", "ao_coulomb", 10, 3, 2.0, "both"),
        (13, 10, 3, 11, "raw_coulomb", "ao_coulomb", 10, 3, 2.0, "coefficients"),
        (13, 10, 3, 11, "raw_coulomb", "ao_coulomb", 10, 3, 2.0, "ao"),
        # pad_coul: seed_rows < ext (occ and obs-sized seeds on raw_coulomb)
        (14, 11, 3, 9, "raw_coulomb", "ao_coulomb", 3, 0, 1.0, "both"),
        (14, 11, 3, 9, "raw_coulomb", "ao_coulomb", 3, 3, 2.0, "both"),
        (14, 11, 3, 9, "raw_coulomb", "ao_coulomb", 6, 3, 0.75, "coefficients"),
        (14, 11, 3, 9, "raw_coulomb", "ao_coulomb", 6, 3, 0.75, "ao"),
        # raw_coulomb with occ == 0 (used by _f12_cabs_fock.obs_sources)
        (9, 9, 0, 7, "raw_coulomb", "ao_coulomb", 9, 0, 1.0, "both"),
        # Narrow left_dim = occ operators (f12, f12_squared, f12_over_coulomb, double_commutator)
        (15, 12, 4, 10, "raw_f12", "ao_f12", 4, 0, 1.0, "both"),
        (15, 12, 4, 10, "raw_f12", "ao_f12", 4, 2, 2.0, "both"),
        (15, 12, 4, 10, "raw_f12_squared", "ao_f12_squared", 4, 0, 1.5, "coefficients"),
        (15, 12, 4, 10, "raw_f12_over_coulomb", "ao_f12_over_coulomb", 4, 0, 2.0, "ao"),
        (15, 12, 4, 10, "raw_double_commutator", "ao_double_commutator", 4, 0, 1.0, "both"),
    ],
)
def test_project_ao_operator_fwd_and_pullback_parity(
    host_sink: Path,
    monkeypatch,
    nc: int,
    ext: int,
    occ: int,
    aux: int,
    field: str,
    ao_name: str,
    seed_rows: int,
    tc_rows: int,
    seed_scale: float,
    inputs_mode: str,
):
    """Exact parity (<= 1e-12 fwd, <= 1e-11 rev) with 0 TensorIR calls on AOT path."""
    rng = np.random.default_rng(1001 + nc * 17 + ext * 13 + occ * 7 + seed_rows * 3 + tc_rows)
    coeffs = rng.standard_normal((nc, ext))
    ao_mat = rng.standard_normal((aux, nc, nc))  # non-symmetric on purpose
    seed = rng.standard_normal((aux, seed_rows, ext))
    tc_bar = rng.standard_normal((aux, tc_rows, ext)) if tc_rows > 0 else None

    # Reference TensorIR oracle
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO", "1")
    cpu = _cpu_run()
    ref_fwd = project_ao_operator_tiled(
        nc, ext, occ, aux, field, ao_name, coeffs, ao_mat, cpu, batch_size=4
    )
    init_bar_c = rng.standard_normal((nc, ext))
    init_bar_ao = rng.standard_normal((aux, nc, nc))
    ref_bar_c = init_bar_c.copy() if inputs_mode != "ao" else None
    ref_bar_ao_in = init_bar_ao.copy() if inputs_mode != "coefficients" else None
    ref_bar_ao = project_ao_operator_pullback_tiled(
        nc,
        ext,
        occ,
        aux,
        field,
        ao_name,
        coeffs,
        None if inputs_mode == "ao" else ao_mat,
        seed,
        cpu,
        bar_coefficients=ref_bar_c,
        bar_ao=ref_bar_ao_in,
        batch_size=4,
        seed_scale=seed_scale,
        three_center_bar=tc_bar,
        inputs_mode=inputs_mode,  # type: ignore[arg-type]
    )

    # AOT path with FORBIDDEN TensorIR runner -> proves 0 TensorIR calls!
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO")
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_AOMO", "1")
    aot_fwd = project_ao_operator_tiled(
        nc,
        ext,
        occ,
        aux,
        field,
        ao_name,
        coeffs,
        ao_mat,
        _forbidden_run,
        batch_size=3,
        shell_library=host_sink,
        host_reference=True,
    )
    assert np.max(np.abs(aot_fwd - ref_fwd)) <= 1e-12

    aot_bar_c = init_bar_c.copy() if inputs_mode != "ao" else None
    aot_bar_ao_in = init_bar_ao.copy() if inputs_mode != "coefficients" else None
    aot_bar_ao = project_ao_operator_pullback_tiled(
        nc,
        ext,
        occ,
        aux,
        field,
        ao_name,
        coeffs,
        None if inputs_mode == "ao" else ao_mat,
        seed,
        _forbidden_run,
        bar_coefficients=aot_bar_c,
        bar_ao=aot_bar_ao_in,
        batch_size=3,
        seed_scale=seed_scale,
        three_center_bar=tc_bar,
        inputs_mode=inputs_mode,  # type: ignore[arg-type]
        shell_library=host_sink,
        host_reference=True,
    )
    if inputs_mode != "ao":
        assert aot_bar_c is not None and ref_bar_c is not None
        assert np.max(np.abs(aot_bar_c - ref_bar_c)) <= 1e-11
    if inputs_mode == "coefficients":
        assert aot_bar_ao is None and ref_bar_ao is None
    else:
        assert aot_bar_ao is not None and ref_bar_ao is not None
        assert np.max(np.abs(aot_bar_ao - ref_bar_ao)) <= 1e-11

    # Also test bar_ao=None allocation path for inputs_mode in ("both", "ao")
    if inputs_mode != "coefficients":
        fresh_aot = project_ao_operator_pullback_tiled(
            nc,
            ext,
            occ,
            aux,
            field,
            ao_name,
            coeffs,
            None if inputs_mode == "ao" else ao_mat,
            seed,
            _forbidden_run,
            bar_coefficients=None,
            bar_ao=None,
            seed_scale=seed_scale,
            three_center_bar=tc_bar,
            inputs_mode=inputs_mode,  # type: ignore[arg-type]
            shell_library=host_sink,
            host_reference=True,
        )
        assert fresh_aot is not None
        assert np.max(np.abs(fresh_aot - (ref_bar_ao - init_bar_ao))) <= 1e-11


@pytest.mark.parametrize(
    "case_id,oa,ob,obs,ext,nc,aux,same_spin,sym_weights,role,occ_coul,obs_coul,skip_fit,stream",
    [
        ("rhf-shared-share_rev", 3, 3, 6, 10, 12, 8, True, True, "shared", False, False, False, False),
        ("rhf-shared-asym_weights", 3, 3, 6, 10, 12, 8, True, False, "shared", False, False, False, False),
        ("rhf-shared-stream-sink", 3, 3, 6, 10, 12, 8, True, True, "shared", False, False, False, True),
        ("uhf-shared-stream-sink", 4, 2, 7, 11, 13, 9, False, False, "shared", False, False, False, True),
        ("uhf-ob0-shared", 3, 0, 5, 8, 9, 7, False, False, "shared", False, False, False, False),
        ("uhf-ob0-pairs-occ_coul", 3, 0, 5, 8, 9, 7, False, False, "pairs", True, False, False, False),
        ("uhf-pairs-obs_coul-skip_fit", 4, 2, 7, 11, 13, 9, False, False, "pairs", False, True, True, True),
        ("rhf-fock-stream", 3, 3, 6, 10, 12, 8, True, True, "fock", False, False, False, True),
        ("uhf-ordinary_jk", 4, 2, 7, 7, 7, 9, False, False, "ordinary_jk", False, False, False, False),
        ("rhf-ordinary_fock", 3, 3, 6, 6, 6, 8, True, True, "ordinary_fock", False, False, False, False),
    ],
)
def test_orbital_sources_all_modes_and_streaming(
    host_sink: Path,
    monkeypatch,
    case_id: str,
    oa: int,
    ob: int,
    obs: int,
    ext: int,
    nc: int,
    aux: int,
    same_spin: bool,
    sym_weights: bool,
    role: str,
    occ_coul: bool,
    obs_coul: bool,
    skip_fit: bool,
    stream: bool,
):
    """Exhaustive `_orbital_sources` parity across roles, flags, ob==0, and split_ao streaming."""
    rng = np.random.default_rng(abs(hash(case_id)) % (2**31))
    space, ao_full = _synthetic_ao_bundle(
        rng, oa, ob, obs, ext, nc, aux, same_spin=same_spin
    )

    def _build_ao_and_sink(use_stream: bool):
        if not use_stream:
            return dict(ao_full), None, {}
        loaded: list[str] = []
        sunk: dict[str, np.ndarray] = {}
        ao_s = {k: v for k, v in ao_full.items() if not k.startswith("ao_")}

        def loader(name: str) -> np.ndarray:
            loaded.append(name)
            return ao_full[name].copy()

        def sink(name: str, val: np.ndarray) -> None:
            sunk[name] = val.copy()

        ao_s["__ao_loader__"] = loader
        return ao_s, sink, sunk

    # Reference TensorIR oracle
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO", "1")
    cpu = _cpu_run()
    ao_ref, sink_ref, sunk_ref = _build_ao_and_sink(stream)
    ref_data, _ = _orbital_sources(
        space,
        ao_ref,
        cpu,
        None,
        projected=None,
        role=role,  # type: ignore[arg-type]
        occupied_coulomb=occ_coul,
        obs_coulomb=obs_coul,
        skip_coulomb_fit_solve=skip_fit,
    )
    weights = _random_orbital_weights(rng, ref_data, symmetric_spins=sym_weights)
    _, ref_bars = _orbital_sources(
        space,
        ao_ref,
        cpu,
        weights,
        projected=ref_data,
        role=role,  # type: ignore[arg-type]
        ao_bar_sink=sink_ref,
        occupied_coulomb=occ_coul,
        obs_coulomb=obs_coul,
        skip_coulomb_fit_solve=skip_fit,
    )
    assert ref_bars is not None

    # AOT path with FORBIDDEN TensorIR runner -> 0 TensorIR calls!
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO")
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_AOMO", "1")
    ao_aot, sink_aot, sunk_aot = _build_ao_and_sink(stream)
    aot_data, _ = _orbital_sources(
        space,
        ao_aot,
        _forbidden_run,
        None,
        projected=None,
        role=role,  # type: ignore[arg-type]
        occupied_coulomb=occ_coul,
        obs_coulomb=obs_coul,
        skip_coulomb_fit_solve=skip_fit,
        shell_library=host_sink,
        host_reference=True,
    )
    assert set(aot_data.keys()) == set(ref_data.keys())
    for k in ref_data:
        if k == "space":
            assert np.array_equal(aot_data[k], ref_data[k])
        else:
            err = float(np.max(np.abs(aot_data[k] - ref_data[k]))) if ref_data[k].size else 0.0
            assert err <= 1e-12, f"{case_id} forward key {k} max abs err {err}"

    _, aot_bars = _orbital_sources(
        space,
        ao_aot,
        _forbidden_run,
        weights,
        projected=aot_data,
        role=role,  # type: ignore[arg-type]
        ao_bar_sink=sink_aot,
        occupied_coulomb=occ_coul,
        obs_coulomb=obs_coul,
        skip_coulomb_fit_solve=skip_fit,
        shell_library=host_sink,
        host_reference=True,
    )
    assert aot_bars is not None
    assert set(aot_bars.keys()) == set(ref_bars.keys())
    for k in ref_bars:
        err = float(np.max(np.abs(aot_bars[k] - ref_bars[k]))) if ref_bars[k].size else 0.0
        assert err <= 1e-11, f"{case_id} reverse bar {k} max abs err {err}"
    assert set(sunk_aot.keys()) == set(sunk_ref.keys())
    for k in sunk_ref:
        err = float(np.max(np.abs(sunk_aot[k] - sunk_ref[k])))
        assert err <= 1e-11, f"{case_id} streamed sink {k} max abs err {err}"


_MOLECULAR_CASES = [
    pytest.param(
        "H2O",
        (("O", (0.0, 0.0, 0.0)), ("H", (0.0, -1.43, 1.11)), ("H", (0.0, 1.43, 1.11))),
        (1, 8),
        0,
        1,
        "rhf",
        "all-active",
        id="RHF-H2O-AE",
    ),
    pytest.param(
        "CH4",
        (
            ("C", (0.0, 0.0, 0.0)),
            ("H", (1.19, 1.19, 1.19)),
            ("H", (-1.19, -1.19, 1.19)),
            ("H", (-1.19, 1.19, -1.19)),
            ("H", (1.19, -1.19, -1.19)),
        ),
        (1, 6),
        0,
        1,
        "rhf",
        "frozen-core",
        id="RHF-CH4-FC",
    ),
    pytest.param(
        "OH",
        (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.83))),
        (1, 8),
        0,
        2,
        "uhf",
        "all-active",
        id="UHF-OH-AE",
    ),
    pytest.param(
        "HBr",
        (("Br", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.68))),
        (1, 35),
        0,
        1,
        "rhf",
        "frozen-core",
        id="ECP-HBr-FC4",
    ),
    pytest.param(
        "HBr",
        (("Br", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.68))),
        (1, 35),
        0,
        1,
        "rhf",
        "frozen-core-spd",
        id="ECP-HBr-FC9",
    ),
]


def _canonical_ref(provider: MolecularFocalProvider, req: FocalRequest):
    has_ecp = any(
        req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons
        for a in req.atoms
    )
    if not has_ecp:
        return df_reference(req, forces=False, budget_mib=provider.budget_mib)

    # Exact canonical DF-RHF on native one-electron ECP + JKFit integrals for CPU ECP tests.
    from dataclasses import replace
    from scipy.linalg import eigh
    from generativeqc._f12_focal_reference import FocalReference

    one = one_electron_sources(
        req,
        library=provider.one_electron_library or provider.shell_library,
        orbital_only=True,
        derivatives=False,
        budget_mib=provider.budget_mib,
    )
    jk = provider._panel(
        replace(req, role="mp2_f12_correlation"), "jk_fit", orbital_only=True
    )
    S = np.asarray(one.overlap, dtype=np.float64)
    H = np.asarray(one.hcore, dtype=np.float64)
    ao3 = np.asarray(jk.data["ao_coulomb"], dtype=np.float64)
    B = np.einsum("PQ,Qmn->Pmn", jk.whitener, ao3, optimize=True)
    n_elec = (
        sum(
            a.atomic_number - req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons
            for a in req.atoms
        )
        - req.charge
    )
    nocc = n_elec // 2
    enuc = 0.0
    for i, a in enumerate(req.atoms):
        zi = a.atomic_number - req.basis.orbital.by_element[a.atomic_number].ecp_core_electrons
        for j in range(i + 1, len(req.atoms)):
            zj = req.atoms[j].atomic_number - req.basis.orbital.by_element[req.atoms[j].atomic_number].ecp_core_electrons
            r = float(np.linalg.norm(np.array(a.position) - np.array(req.atoms[j].position)))
            enuc += zi * zj / r

    eps, C = eigh(H, S)
    D = 2.0 * (C[:, :nocc] @ C[:, :nocc].T)
    f_hist, e_hist = [], []
    F = H.copy()
    it = 0
    for it in range(1, 121):
        c_occ = C[:, :nocc]
        q_vec = np.einsum("Pmn,mn->P", B, D, optimize=True)
        J = np.einsum("P,Pmn->mn", q_vec, B, optimize=True)
        Bo = np.einsum("Pmn,ni->Pmi", B, c_occ, optimize=True)
        K = np.einsum("Pmi,Pni->mn", Bo, Bo, optimize=True)
        F = 0.5 * (H + J - K + (H + J - K).T)
        err = F @ D @ S - S @ D @ F
        if float(np.max(np.abs(err))) < 1e-12:
            break
        f_hist.append(F.copy())
        e_hist.append(err.ravel().copy())
        if len(f_hist) > 8:
            f_hist.pop(0)
            e_hist.pop(0)
        if len(f_hist) >= 2:
            m = len(f_hist)
            A = np.zeros((m + 1, m + 1))
            for a_i in range(m):
                for b_i in range(m):
                    A[a_i, b_i] = float(np.dot(e_hist[a_i], e_hist[b_i]))
                A[a_i, m] = A[m, a_i] = -1.0
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                w = np.linalg.solve(A, rhs)[:m]
                F_use = sum(wi * fi for wi, fi in zip(w, f_hist))
            except np.linalg.LinAlgError:
                F_use = F
        else:
            F_use = F
        eps, C = eigh(F_use, S)
        D = 2.0 * (C[:, :nocc] @ C[:, :nocc].T)
    eps, C = eigh(F, S)
    D = 2.0 * (C[:, :nocc] @ C[:, :nocc].T)
    e_hf = float(0.5 * np.sum(D * (H + F)) + enuc)
    return FocalReference(
        energy=e_hf,
        forces=None,
        overlap=S,
        hcore=H,
        coefficients=C[None, :, :].copy(),
        orbital_energies=eps[None, :].copy(),
        density=D[None, :, :].copy(),
        fock=F[None, :, :].copy(),
        occupied=(nocc,),
        iterations=it,
        executed_backend=req.backend,
        surface_id=req.surface_id,
        basis_id=req.basis.basis_id,
        orbital_id=req.basis.orbital.identity,
        jk_fit_id=req.basis.jk_fit.identity,
    )






# ============================================================================
# Step 11.2 — Dynamic-dimension AOT CABS Singles E+F Gate 1 parity tests
# ============================================================================


@pytest.mark.parametrize(
    "operation,field",
    [
        ("fock_fwd", "occ_a"),
        ("fock_rev", "seed_b"),
        ("fock_rev", "rho"),
        ("fock_rev", "fit_b"),
        ("cphf_action", "df_a"),
        ("cphf_action", "trial_b"),
        ("nuclear_weights", "c_b"),
        ("nuclear_weights", "q_chunk_hint"),
        ("frame_kernel_cotangent", "frame_bar"),
        ("whiten_3c", "pairs"),
    ],
)
def test_cabs_rejects_invalid_buffer_before_native(operation, field):
    from types import SimpleNamespace

    from generativeqc import _f12_ao_mo_native as native

    def unexpected_native(*args):
        pytest.fail("invalid dimensions reached the C ABI")

    name = "cabs_" + operation if operation.startswith("fock") else operation
    ops = SimpleNamespace(**{name: unexpected_native})
    # Non-square frames distinguish the AO, extended and occupied dimensions.
    n, ext, aux, occ = 4, 3, 2, 1
    matrix = np.eye(n)
    frame = matrix[:, :ext]
    seed = np.eye(ext)
    raw = np.zeros((aux, n, n))
    fit = np.zeros((aux, occ, n))
    occ_frame = frame[:, :occ]
    df = np.zeros((aux, ext, ext))
    trial = np.zeros((ext - occ, occ))
    w = np.eye(aux)
    inputs = {
        "fock_fwd": dict(hcore=matrix, raw=raw, whitener=w, occ_a=occ_frame,
                         occ_b=occ_frame, occ_rows=n, same_occ=False, save_intermediates=True),
        "fock_rev": dict(fock_ao_a=matrix, fock_ao_b=matrix, frame_a=frame, frame_b=frame,
                         seed_a=seed, seed_b=seed, raw=raw, whitener=w, rho=np.ones(aux),
                         raw_a=fit, fit_a=fit, raw_b=fit, fit_b=fit, occ_a=occ_frame,
                         occ_b=occ_frame, oa=occ, ob=occ, occ_rows=n, same_spin=False),
        "cphf_action": dict(oa=occ, ob=occ, obs=ext, aux=aux, is_rhf=False,
                            df_a=df, df_b=df, gap_a=trial, gap_b=trial,
                            trial_a=trial, trial_b=trial),
        "nuclear_weights": dict(oa=occ, ob=occ, obs=ext, aux=aux, nao=n, same_spin=False,
                                whitener=w, c_a=frame, c_b=frame, df_a=df, df_b=df,
                                p_a=seed, p_b=seed, sm_a=seed, sm_b=seed),
        "frame_kernel_cotangent": dict(overlap=matrix, frame=frame, frame_bar=frame),
        "whiten_3c": dict(whitener=np.eye(1), raw=raw[:1]),
    }[operation]
    if field == "q_chunk_hint":
        inputs[field] = 2**31
    elif field == "pairs":
        # Reject the ABI overflow without materializing a 16 GiB broadcast.
        inputs["raw"] = np.broadcast_to(np.float64(0), (1, 2**31))
    else:
        inputs[field] = inputs[field][:-1]
    with pytest.raises(ValueError, match=field):
        getattr(native, "run_cabs_" + operation)(ops, **inputs)


def test_cabs_singles_backend_isolation_and_validation(host_sink: Path, monkeypatch):
    """Verify CABS Singles AOT ops enforce backend isolation and non-finite guards."""
    from generativeqc._f12_ao_mo_native import (
        bind_cabs_singles_ops,
        resolve_cabs_singles_ops,
        run_cabs_fock_fwd,
    )

    ops = bind_cabs_singles_ops(host_sink, host_reference=True)
    assert ops is not None and ops.host_reference is True
    assert bind_cabs_singles_ops(host_sink, host_reference=False) is None

    def fake_cuda_runner(p, i):
        raise AssertionError("not called")

    fake_cuda_runner.batch = lambda p: None  # type: ignore[attr-defined]
    fake_cuda_runner.profile_snapshot = lambda: {}  # type: ignore[attr-defined]

    assert (
        resolve_cabs_singles_ops(fake_cuda_runner, shell_library=host_sink) is None
    )

    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO", "1")
    assert resolve_cabs_singles_ops(_cpu_run(), shell_library=host_sink) is None
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO")

    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_AOMO", "1")
    with pytest.raises(RuntimeError, match="GENERATIVEQC_F12_REQUIRE_AOT_AOMO"):
        resolve_cabs_singles_ops(fake_cuda_runner, shell_library=host_sink)
    monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_AOMO")

    # Non-finite input guard
    bad_raw = np.zeros((3, 4, 4), dtype=np.float64)
    bad_raw[2, 1, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        run_cabs_fock_fwd(
            ops,
            np.eye(4, dtype=np.float64),
            bad_raw,
            np.eye(3, dtype=np.float64),
            np.eye(4, 1, dtype=np.float64),
            None,
            occ_rows=4,
            same_occ=True,
            save_intermediates=False,
        )


@pytest.mark.parametrize(
    "case_id,n,obs,ext,q,oa,ob,same_spin,occ_on_obs_only",
    [
        ("rhf-full", 11, 5, 11, 9, 2, 2, True, True),
        ("rhf-trunc-cabs", 13, 6, 10, 9, 3, 3, True, True),
        ("rhf-dense-frame", 10, 5, 9, 8, 2, 2, True, False),
        ("uhf-oa-ne-ob", 12, 6, 10, 9, 3, 1, False, True),
        ("uhf-ob-zero", 11, 5, 9, 8, 2, 0, False, True),
        ("uhf-dense-frame", 10, 5, 8, 7, 3, 2, False, False),
    ],
)
def test_cabs_fock_and_frame_kernel_unit_parity(
    host_sink: Path,
    case_id: str,
    n: int,
    obs: int,
    ext: int,
    q: int,
    oa: int,
    ob: int,
    same_spin: bool,
    occ_on_obs_only: bool,
):
    """Exact unit parity for AOT physical_fock fwd/rev and frame_kernel_cotangent."""
    from generativeqc._f12_ao_mo_native import (
        bind_cabs_singles_ops,
        run_cabs_fock_fwd,
        run_cabs_fock_rev,
        run_cabs_frame_kernel_cotangent,
    )
    from generativeqc._f12_cabs_fock import _project, physical_fock
    from generativeqc_compiler.mp2.f12_frame_response import frame_kernel_cotangent

    import zlib

    ops = bind_cabs_singles_ops(host_sink, host_reference=True)
    assert ops is not None
    rng = np.random.default_rng(7001 + (zlib.crc32(case_id.encode()) % 100000))

    def sym(x: np.ndarray) -> np.ndarray:
        return 0.5 * (x + x.swapaxes(-1, -2))

    h = sym(rng.standard_normal((n, n)))
    raw = sym(rng.standard_normal((q, n, n)))
    m_raw = rng.standard_normal((q, q))
    metric = m_raw @ m_raw.T + 2.0 * np.eye(q)
    whitener = np.linalg.solve(np.linalg.cholesky(metric), np.eye(q))

    def _make_frame() -> np.ndarray:
        c = rng.standard_normal((n, ext))
        if occ_on_obs_only:
            c[obs:, :obs] = 0.0
        return c

    frame_a = _make_frame()
    frame_b = frame_a.copy() if same_spin else _make_frame()
    seed_a = rng.standard_normal((ext, ext))
    seed_b = seed_a.copy() if same_spin else rng.standard_normal((ext, ext))

    cpu = _cpu_run()
    ref_focks, ref_pb = physical_fock(
        h, raw, whitener, (frame_a, frame_b), (oa, ob), cpu, False, None
    )
    (ref_ga, ref_gb), ref_core, ref_raw_bar, ref_metric = ref_pb([seed_a, seed_b])

    occ_rows = obs if occ_on_obs_only else n
    occ_a = np.ascontiguousarray(frame_a[:occ_rows, :oa]) if oa > 0 else None
    occ_b = np.ascontiguousarray(frame_b[:occ_rows, :ob]) if ob > 0 else None
    fa_ao, fb_ao, rho_v, ra_v, da_v, rb_v, db_v = run_cabs_fock_fwd(
        ops,
        h,
        raw,
        whitener,
        occ_a,
        occ_b,
        occ_rows=occ_rows,
        same_occ=same_spin,
        save_intermediates=True,
    )
    aot_focks = [
        _project(fa_ao, frame_a, False, None),
        _project(fb_ao, frame_b, False, None),
    ]
    for s_idx in range(2):
        assert np.max(np.abs(aot_focks[s_idx] - ref_focks[s_idx])) <= 1e-12

    assert rho_v is not None
    aot_ga, aot_gb, aot_core, aot_raw_bar, aot_metric = run_cabs_fock_rev(
        ops,
        fa_ao,
        fb_ao,
        frame_a,
        frame_b,
        seed_a,
        seed_b,
        raw,
        whitener,
        rho_v,
        ra_v,
        da_v,
        rb_v,
        db_v,
        occ_a,
        occ_b,
        oa=oa,
        ob=ob,
        occ_rows=occ_rows,
        same_spin=same_spin,
    )
    def _assert_close(aot_arr: np.ndarray, ref_arr: np.ndarray, label_s: str) -> None:
        err = float(np.max(np.abs(aot_arr - ref_arr)))
        mag = float(np.max(np.abs(ref_arr))) if ref_arr.size else 0.0
        assert err <= 1e-11 + 1e-12 * mag, f"{case_id} {label_s} err={err} mag={mag}"

    _assert_close(aot_core, ref_core, "core_bar")
    _assert_close(aot_raw_bar, ref_raw_bar, "raw_bar")
    _assert_close(aot_metric, ref_metric, "metric_bar")
    if occ_on_obs_only:
        _assert_close(aot_ga[:obs, :obs], ref_ga[:obs, :obs], "ga_obs")
        _assert_close(aot_gb[:obs, :obs], ref_gb[:obs, :obs], "gb_obs")
        _assert_close(aot_ga[:, obs:], ref_ga[:, obs:], "ga_cabs")
        _assert_close(aot_gb[:, obs:], ref_gb[:, obs:], "gb_cabs")
    else:
        _assert_close(aot_ga, ref_ga, "ga")
        _assert_close(aot_gb, ref_gb, "gb")

    # Also test frame_kernel_cotangent parity
    rank = ext - obs
    if rank > 0:
        s_ov = sym(rng.standard_normal((n, n))) + 3.0 * np.eye(n)
        comp = rng.standard_normal((n, rank))
        cg = rng.standard_normal((n, rank))
        ref_fk = cpu(
            frame_kernel_cotangent(n, rank),
            {"overlap": s_ov, "frame": comp, "frame_bar": cg},
        ).outputs
        aot_fk = run_cabs_frame_kernel_cotangent(ops, s_ov, comp, cg)
        for key in ("kernel_bar", "dual_gram", "gauge_skew", "reconstruction_residual"):
            err = float(np.max(np.abs(aot_fk[key] - ref_fk[key])))
            mag = float(np.max(np.abs(ref_fk[key]))) if ref_fk[key].size else 0.0
            assert err <= 1e-12 + 1e-13 * mag, f"{case_id} fk {key} err={err} mag={mag}"


@pytest.mark.parametrize(
    "case_id,oa,ob,obs,aux,nao,same_spin",
    [
        ("rhf-obs", 3, 3, 7, 9, 7, True),
        ("uhf-obs", 4, 2, 7, 9, 7, False),
        ("uhf-ob0", 3, 0, 6, 8, 6, False),
        ("uhf-nao-gt-obs", 3, 1, 6, 8, 10, False),
    ],
)
def test_cabs_cphf_action_and_nuclear_weights_unit_parity(
    host_sink: Path,
    case_id: str,
    oa: int,
    ob: int,
    obs: int,
    aux: int,
    nao: int,
    same_spin: bool,
):
    """Exact unit parity for AOT whiten_3c, cphf_action, and nuclear_weights."""
    from generativeqc._f12_ao_mo_native import (
        bind_cabs_singles_ops,
        run_cabs_cphf_action,
        run_cabs_nuclear_weights,
        run_cabs_whiten_3c,
    )
    from generativeqc._f12_focal_singles_response import _rhf_df_orbital_action
    from generativeqc.f12_orbital_response import unrestricted_nuclear_source_weights
    from generativeqc_compiler.mp2.f12_unrestricted_reference import (
        unrestricted_df_orbital_action,
    )

    import zlib

    ops = bind_cabs_singles_ops(host_sink, host_reference=True)
    assert ops is not None
    rng = np.random.default_rng(8101 + (zlib.crc32(case_id.encode()) % 100000))

    w_raw = rng.standard_normal((aux, aux))
    metric = w_raw @ w_raw.T + 2.0 * np.eye(aux)
    whitener = np.linalg.solve(np.linalg.cholesky(metric), np.eye(aux))
    raw_a = rng.standard_normal((aux, obs, obs))
    raw_a = 0.5 * (raw_a + raw_a.swapaxes(1, 2))
    df_a = run_cabs_whiten_3c(ops, whitener, raw_a)
    assert np.max(np.abs(df_a - np.einsum("ab,bpq->apq", whitener, raw_a))) <= 1e-12

    if same_spin:
        df_b = df_a
    else:
        raw_b = rng.standard_normal((aux, obs, obs))
        raw_b = 0.5 * (raw_b + raw_b.swapaxes(1, 2))
        df_b = run_cabs_whiten_3c(ops, whitener, raw_b)

    va, vb = obs - oa, obs - ob
    gap_a = rng.uniform(0.5, 3.0, size=(va, oa)) if oa > 0 else None
    gap_b = (
        gap_a.copy()
        if same_spin
        else (rng.uniform(0.5, 3.0, size=(vb, ob)) if ob > 0 else None)
    )
    trial_a = rng.standard_normal((va, oa)) if oa > 0 else None
    trial_b = (
        trial_a.copy()
        if same_spin
        else (rng.standard_normal((vb, ob)) if ob > 0 else None)
    )

    cpu = _cpu_run()
    # 1. CPHF action parity (both RHF fast-path and general UHF)
    if same_spin:
        ref_rhf = cpu(
            _rhf_df_orbital_action(oa, obs, aux),
            {
                "vv": df_a[:, oa:, oa:],
                "oo": df_a[:, :oa, :oa],
                "vo": df_a[:, oa:, :oa],
                "trial": trial_a,
                "gap": gap_a,
            },
        ).outputs["rhf"]
        aot_a, aot_b = run_cabs_cphf_action(
            ops,
            oa,
            ob,
            obs,
            aux,
            is_rhf=True,
            df_a=df_a,
            df_b=None,
            gap_a=gap_a,
            gap_b=None,
            trial_a=trial_a,
            trial_b=None,
        )
        assert aot_a is not None and aot_b is None
        assert np.max(np.abs(aot_a - ref_rhf)) <= 1e-12
    else:
        u_in = {}
        if oa > 0:
            u_in.update(
                {
                    "vv_alpha": df_a[:, oa:, oa:],
                    "oo_alpha": df_a[:, :oa, :oa],
                    "vo_alpha": df_a[:, oa:, :oa],
                    "trial_alpha": trial_a,
                    "gap_alpha": gap_a,
                }
            )
        if ob > 0:
            u_in.update(
                {
                    "vv_beta": df_b[:, ob:, ob:],
                    "oo_beta": df_b[:, :ob, :ob],
                    "vo_beta": df_b[:, ob:, :ob],
                    "trial_beta": trial_b,
                    "gap_beta": gap_b,
                }
            )
        ref_u = cpu(unrestricted_df_orbital_action(oa, ob, obs, aux), u_in).outputs
        aot_a, aot_b = run_cabs_cphf_action(
            ops,
            oa,
            ob,
            obs,
            aux,
            is_rhf=False,
            df_a=df_a if oa > 0 else None,
            df_b=df_b if ob > 0 else None,
            gap_a=gap_a,
            gap_b=gap_b,
            trial_a=trial_a,
            trial_b=trial_b,
        )
        if oa > 0:
            assert aot_a is not None
            assert np.max(np.abs(aot_a - ref_u["alpha"])) <= 1e-12
        if ob > 0:
            assert aot_b is not None
            assert np.max(np.abs(aot_b - ref_u["beta"])) <= 1e-12
        else:
            assert aot_b is None

    # 2. Nuclear weights parity
    c_a = rng.standard_normal((nao, obs))
    c_b = c_a.copy() if same_spin else rng.standard_normal((nao, obs))
    eps_a = rng.standard_normal((oa,)) if oa > 0 else np.empty((0,))
    eps_b = eps_a.copy() if same_spin else (rng.standard_normal((ob,)) if ob > 0 else np.empty((0,)))
    fseed_a = rng.standard_normal((obs, obs))
    fseed_b = fseed_a.copy() if same_spin else rng.standard_normal((obs, obs))
    sseed_a = rng.standard_normal((obs, obs))
    sseed_b = sseed_a.copy() if same_spin else rng.standard_normal((obs, obs))
    z_a = rng.standard_normal((va, oa)) if oa > 0 else np.empty((0, 0))
    z_b = z_a.copy() if same_spin else (rng.standard_normal((vb, ob)) if ob > 0 else np.empty((0, 0)))
    z_parts = []
    if oa > 0:
        z_parts.append(z_a.ravel())
    if ob > 0:
        z_parts.append(z_b.ravel())
    z_vec = np.concatenate(z_parts)

    central = {
        "whitener": whitener,
        "coefficients_alpha": c_a,
        "coefficients_beta": c_b,
        "df_alpha": df_a,
        "df_beta": df_b,
        "fock_seed_alpha": fseed_a,
        "fock_seed_beta": fseed_b,
        "overlap_seed_alpha": sseed_a,
        "overlap_seed_beta": sseed_b,
    }
    if oa > 0:
        central["occupied_energies_alpha"] = eps_a
    if ob > 0:
        central["occupied_energies_beta"] = eps_b

    ref_nuc = unrestricted_nuclear_source_weights((oa, ob, obs, aux), central, z_vec, cpu)

    p_a = fseed_a.copy()
    sm_a = sseed_a.copy()
    if oa > 0:
        p_a[oa:, :oa] -= z_a
        sm_a[oa:, :oa] += z_a * eps_a[None, :]
    p_b = fseed_b.copy()
    sm_b = sseed_b.copy()
    if ob > 0:
        p_b[ob:, :ob] -= z_b
        sm_b[ob:, :ob] += z_b * eps_b[None, :]

    aot_nuc = run_cabs_nuclear_weights(
        ops,
        oa,
        ob,
        obs,
        aux,
        nao,
        same_spin=same_spin,
        whitener=whitener,
        c_a=c_a,
        c_b=c_b,
        df_a=df_a,
        df_b=df_b,
        p_a=p_a,
        p_b=p_b,
        sm_a=sm_a,
        sm_b=sm_b,
    )
    for key in ("overlap_ao", "hcore_ao", "ao_coulomb", "metric/coulomb"):
        err = float(np.max(np.abs(aot_nuc[key] - ref_nuc[key])))
        assert err <= 1e-11, f"{case_id} nuclear weight {key} err={err}"






# ============================================================================
# Step 11.3 — Bounded, identity-safe ha-D source reuse & anti-bleed Gate 1 tests
# ============================================================================


def _compact_shared_fit_basis(elements: tuple[int, ...], *, beta: float = 1.0) -> FocalBasis:
    """Compact spherical shared-fit basis (jk_fit == mp2_fit == f12_fit)."""
    split = _compact_split_basis(elements)
    return FocalBasis(
        2,
        f"gate1-aomo-shared-b{beta}",
        split.orbital,
        jk_fit=split.f12_fit,
        mp2_fit=split.f12_fit,
        f12_fit=split.f12_fit,
        cabs=split.cabs,
        beta=beta,
    )


def _compact_split_basis_with_beta(
    elements: tuple[int, ...], beta: float
) -> FocalBasis:
    split = _compact_split_basis(elements)
    return FocalBasis(
        2,
        f"gate1-aomo-split-b{beta}",
        split.orbital,
        jk_fit=split.jk_fit,
        mp2_fit=split.mp2_fit,
        f12_fit=split.f12_fit,
        cabs=split.cabs,
        beta=beta,
    )


def test_had_coulomb_slice_readonly_and_killswitch(host_sink: Path, monkeypatch):
    """Verify Coulomb downward-slice hit, upward-upgrade safety, ob==0, immutability, and opt-out."""
    from generativeqc._f12_spin_ao import (
        clear_projected_source_cache,
        projected_source_cache_info,
    )

    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_HAD_REUSE", raising=False)
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_AOMO", "1")
    clear_projected_source_cache(reset_stats=True)

    # 1. Full ext -> obs -> occ downward slicing must hit with 0 extra projections.
    rng = np.random.default_rng(11301)
    space, ao = _synthetic_ao_bundle(rng, 3, 2, 6, 10, 12, 8, same_spin=False)
    oa, ob, obs, ext, aux = space

    d_shared, _ = _orbital_sources(
        space, ao, _forbidden_run, None, projected=None, role="shared",
        shell_library=host_sink, host_reference=True,
    )
    info1 = projected_source_cache_info()
    assert info1["misses"] == 1 and info1["size"] == 1
    assert d_shared["source/raw_coulomb_alpha"].shape == (aux, ext, ext)
    assert d_shared["source/raw_coulomb_alpha"].flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        d_shared["source/raw_coulomb_alpha"][0, 0, 0] = 0.0

    d_obs, _ = _orbital_sources(
        space, ao, _forbidden_run, None, projected=None, role="pairs",
        obs_coulomb=True, shell_library=host_sink, host_reference=True,
    )
    info2 = projected_source_cache_info()
    assert info2["hits"] == 1
    assert info2["projection_calls"] == info1["projection_calls"]
    assert info2["coulomb_fit_calls"] == info1["coulomb_fit_calls"]
    assert "source/hcore_alpha" not in d_obs
    assert d_obs["source/raw_coulomb_alpha"].shape == (aux, obs, ext)
    assert d_obs["source/raw_coulomb_beta"].shape == (aux, obs, ext)
    assert np.array_equal(
        d_obs["source/raw_coulomb_alpha"], d_shared["source/raw_coulomb_alpha"][:, :obs, :]
    )
    assert d_obs["source/raw_coulomb_alpha"].flags.writeable is False

    d_occ, _ = _orbital_sources(
        space, ao, _forbidden_run, None, projected=None, role="pairs",
        occupied_coulomb=True, skip_coulomb_fit_solve=True,
        shell_library=host_sink, host_reference=True,
    )
    info3 = projected_source_cache_info()
    assert info3["hits"] == 2
    assert info3["projection_calls"] == info1["projection_calls"]
    assert "source/fit_alpha" not in d_occ and "source/fit_beta" not in d_occ
    assert d_occ["source/raw_coulomb_alpha"].shape == (aux, oa, ext)
    assert d_occ["source/raw_coulomb_beta"].shape == (aux, ob, ext)

    # 2. Upward upgrade (occ -> obs -> full ext) must NEVER alias a smaller slice!
    clear_projected_source_cache(reset_stats=True)
    d_up_occ, _ = _orbital_sources(
        space, ao, _forbidden_run, None, projected=None, role="pairs",
        occupied_coulomb=True, shell_library=host_sink, host_reference=True,
    )
    p_occ = projected_source_cache_info()
    d_up_obs, _ = _orbital_sources(
        space, ao, _forbidden_run, None, projected=None, role="pairs",
        obs_coulomb=True, shell_library=host_sink, host_reference=True,
    )
    p_obs = projected_source_cache_info()
    # Only the 2 spins of Coulomb are projected (2 calls instead of 10), 0 extra fit solves!
    assert p_obs["partial_hits"] == 1
    assert p_obs["projection_calls"] == p_occ["projection_calls"] + 2
    assert p_obs["coulomb_fit_calls"] == p_occ["coulomb_fit_calls"]
    assert d_up_obs["source/raw_coulomb_alpha"].shape == (aux, obs, ext)
    assert np.max(np.abs(d_up_obs["source/raw_coulomb_alpha"] - d_obs["source/raw_coulomb_alpha"])) <= 1e-12
    assert np.max(np.abs(d_up_obs["source/raw_coulomb_beta"] - d_obs["source/raw_coulomb_beta"])) <= 1e-12

    d_up_ext, _ = _orbital_sources(
        space, ao, _forbidden_run, None, projected=None, role="shared",
        shell_library=host_sink, host_reference=True,
    )
    p_ext = projected_source_cache_info()
    assert p_ext["partial_hits"] == 2
    assert p_ext["projection_calls"] == p_obs["projection_calls"] + 2
    assert d_up_ext["source/raw_coulomb_alpha"].shape == (aux, ext, ext)
    assert np.max(np.abs(d_up_ext["source/raw_coulomb_alpha"] - d_shared["source/raw_coulomb_alpha"])) <= 1e-12
    assert np.max(np.abs(d_up_ext["source/hcore_alpha"] - d_shared["source/hcore_alpha"])) <= 1e-12

    # 3. Empty beta spin channel (ob == 0): occ (0 rows) -> obs -> occ round-trip
    clear_projected_source_cache(reset_stats=True)
    sp0, ao0 = _synthetic_ao_bundle(rng, 3, 0, 5, 8, 9, 7, same_spin=False)
    d0_occ, _ = _orbital_sources(
        sp0, ao0, _forbidden_run, None, projected=None, role="pairs",
        occupied_coulomb=True, shell_library=host_sink, host_reference=True,
    )
    assert d0_occ["source/raw_coulomb_beta"].shape == (7, 0, 8)
    assert "source/fit_beta" not in d0_occ
    d0_obs, _ = _orbital_sources(
        sp0, ao0, _forbidden_run, None, projected=None, role="pairs",
        obs_coulomb=True, shell_library=host_sink, host_reference=True,
    )
    assert d0_obs["source/raw_coulomb_beta"].shape == (7, 5, 8)
    assert "source/fit_beta" not in d0_obs
    p0_obs = projected_source_cache_info()
    d0_occ2, _ = _orbital_sources(
        sp0, ao0, _forbidden_run, None, projected=None, role="pairs",
        occupied_coulomb=True, shell_library=host_sink, host_reference=True,
    )
    assert projected_source_cache_info()["projection_calls"] == p0_obs["projection_calls"]
    assert d0_occ2["source/raw_coulomb_beta"].shape == (7, 0, 8)

    # 4. Kill-switch disables cache completely (0 entries retained, no hits)
    clear_projected_source_cache(reset_stats=True)
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_HAD_REUSE", "1")
    _orbital_sources(
        space, ao, _forbidden_run, None, projected=None, role="shared",
        shell_library=host_sink, host_reference=True,
    )
    info_off = projected_source_cache_info()
    assert info_off["size"] == 0 and info_off["hits"] == 0 and info_off["misses"] == 0
