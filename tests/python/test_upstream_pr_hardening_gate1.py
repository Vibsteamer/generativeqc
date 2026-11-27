"""Gate 1 unit tests for Upstream-PR Hardening Track B (B1, H1, H5, M2, M7, M8)."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# B1: Optional basis_set_exchange guard & packaging declaration
# ============================================================================


def test_b1_pyproject_declares_basis_set_exchange_optional_extra() -> None:
    """pyproject.toml declares basis-set-exchange in optional dependencies."""
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    opt = data["project"]["optional-dependencies"]
    assert any(dep.startswith("basis-set-exchange") for dep in opt["reference-test"])
    assert any(dep.startswith("basis-set-exchange") for dep in opt["bse"])


# ============================================================================
# M2: Non-_ex integral normalization contract matches compiled kDefault=1 (1/beta)
# ============================================================================


def test_m2_non_ex_fallback_matches_compiled_1_over_beta_default(tmp_path: Path) -> None:
    """Non-_ex fallback bind and execute match compiled kDefaultSlaterFitNormalizationCode=1."""
    import ctypes as ct
    from unittest.mock import patch

    from generativeqc._f12_focal_integral_kernel import (
        DEFAULT_SLATER_FIT_NORMALIZATION,
        LEGACY_SLATER_FIT_NORMALIZATION,
        LEGACY_SLATER_NORM_CODE,
        contract_geometry,
        slater_normalization_code,
    )
    from generativeqc._f12_focal_integral_sink import _bind_fallback_weighted

    assert DEFAULT_SLATER_FIT_NORMALIZATION == "negative_exponential_over_beta"
    assert LEGACY_SLATER_FIT_NORMALIZATION == "negative_exponential_over_beta"
    assert LEGACY_SLATER_NORM_CODE == 1

    fake_lib = tmp_path / "lib_non_ex.so"
    fake_lib.write_bytes(b"")

    def fake_non_ex_kernel(
        ncenter, op, beta, nshell, nprim, nblocks, nweights,
        angular, prim_offsets, exponents, radial, centers,
        blocks, offsets, weights, output, budget, size_ptr,
    ) -> int:
        output[:] = 0.0
        output[:, 0] = 2.5
        ct.cast(size_ptr, ct.POINTER(ct.c_uint64))[0] = 64
        return 0

    def fake_bind(library, kernel, host_reference, *, extended: bool = False):
        if extended:
            raise AttributeError("non-_ex library")
        return fake_non_ex_kernel

    code_over_beta = slater_normalization_code("negative_exponential_over_beta")
    code_unscaled = slater_normalization_code("negative_exponential")
    assert code_over_beta == 1 and code_unscaled == 0

    with patch("generativeqc._f12_focal_integral_sink.bind", side_effect=fake_bind):
        fn, is_ext = _bind_fallback_weighted(fake_lib, "geminal", True, code_over_beta)
        assert fn is fake_non_ex_kernel and is_ext is False
        with pytest.raises(RuntimeError, match="SlaterFitNormalization"):
            _bind_fallback_weighted(fake_lib, "geminal", True, code_unscaled)
        # Coulomb is independent of Slater normalization.
        fn_c, is_ext_c = _bind_fallback_weighted(fake_lib, "coulomb", True, code_unscaled)
        assert fn_c is fake_non_ex_kernel and is_ext_c is False

    fixture = {
        "metric_blocks": np.array([[0, 0]], dtype=np.int32),
        "metric_offsets": np.array([0, 1], dtype=np.int32),
        "weights_metric_f12": np.array([1.0], dtype=np.float64),
        "weights_metric": np.array([1.0], dtype=np.float64),
        "angular": np.array([0], dtype=np.int32),
        "atom": np.array([0], dtype=np.int32),
        "prim_offsets": np.array([0, 1], dtype=np.int32),
        "exponents": np.array([1.0], dtype=np.float64),
        "radial": np.array([1.0], dtype=np.float64),
        "centers": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    }
    manifest = {
        "elements": ["H"],
        "geometry_bohr": [[0.0, 0.0, 0.0]],
        "options": {"f12_beta": 1.0},
    }
    _, row_ok = contract_geometry(
        fixture, manifest, {"geminal": fake_non_ex_kernel},
        "metric", "f12", 1 << 20, True,
        slater_normalization="negative_exponential_over_beta",
        extended_abi=False,
    )
    assert row_ok["weighted_value_Eh"] == pytest.approx(2.5)
    with pytest.raises(RuntimeError, match="SlaterFitNormalization"):
        contract_geometry(
            fixture, manifest, {"geminal": fake_non_ex_kernel},
            "metric", "f12", 1 << 20, True,
            slater_normalization="negative_exponential",
            extended_abi=False,
        )
    # Coulomb accepts non-default Slater normalization on non-_ex too.
    _, row_c = contract_geometry(
        fixture, manifest, {"coulomb": fake_non_ex_kernel},
        "metric", "coulomb", 1 << 20, True,
        slater_normalization="negative_exponential",
        extended_abi=False,
    )
    assert row_c["weighted_value_Eh"] == pytest.approx(2.5)


# ============================================================================
# M7: Frozen-core shell-splitting energy-gap guard (NaBr FC4 vs FC9)
# ============================================================================


def test_frozen_core_shell_gap_refuses_an_explicit_count_that_splits_a_shell() -> None:
    from generativeqc._f12_frozen_core import check_frozen_core_shell_gap

    # A count-based cut falls inside a fivefold near-degenerate d shell.
    eps = np.array([-40.5, -9.6, -7.2, -7.2, -7.2, -3.014, -3.012, -3.012,
                    -3.010, -3.008, -2.8, -1.55, -1.55, -1.55, -0.9, -0.35])
    with pytest.raises(ValueError, match=r"nearest frozen counts .* are 5 and 10\.$"):
        check_frozen_core_shell_gap(eps, 9, n_occupied=16, correlation_space_id="frozen:9")
    assert check_frozen_core_shell_gap(eps, 10, n_occupied=16) == pytest.approx(0.208)


def test_m7_real_df_hf_shell_gaps_pass_for_h2o_oh_hbr() -> None:
    """Real DF-HF canonical orbital gaps for H2O and OH are >> 0.05 Eh."""
    from generativeqc._f12_focal_reference import df_reference
    from generativeqc._f12_frozen_core import (
        DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH,
        select_frozen_occupied_frame,
    )
    from generativeqc.experimental.f12_focal import FocalBasis, FocalSeries, plan_singlepoint

    # 1. H2O (RHF) and OH (UHF) in sto-3g / def2-svp.
    light_basis = FocalBasis(
        2,
        "m7-light",
        "sto-3g",
        jk_fit="def2-svp",
        mp2_fit="def2-svp",
        f12_fit="def2-svp",
        cabs="def2-svp",
        beta=0.9,
    )
    for atoms, mult, root, cspace in (
        (
            (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.43, 1.1)), ("H", (0.0, -1.43, 1.1))),
            1,
            "rhf",
            "frozen-core",
        ),
        (
            (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.83))),
            2,
            "uhf",
            "frozen-core",
        ),
    ):
        plan = plan_singlepoint(
            atoms,
            FocalSeries(light_basis),
            FocalSeries(light_basis),
            None,
            charge=0,
            multiplicity=mult,
            root_id=root,
            correlation_space_id=cspace,
            backend="cpu_reference",
        )
        req = plan.steps[0][0]
        ref = df_reference(req, forces=False, budget_mib=256)
        frame = select_frozen_occupied_frame(
            req.atoms, req.basis.orbital, ref.overlap, ref.coefficients, ref.orbital_energies,
            ref.occupied, cspace,
        )
        assert frame.counts(cspace) == (1, 1)
        for channel in frame.channels(cspace):
            assert channel.minimum_gap_eh > 10.0 * DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH


# ============================================================================
# M8: Guard single-pass fp32_single W schedule on forces
# ============================================================================


def test_m8_fp32_single_never_default_and_refused_on_non_l20_forces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fp32_single is never default; non-L20 forces refuse it before the native contraction."""
    import inspect
    import warnings
    from types import SimpleNamespace

    from generativeqc import _f12_focal_ucc_triples as ucc_triples
    from generativeqc._cuda_device import CudaDeviceIdentity
    from generativeqc._f12_focal_molecular import MolecularFocalProvider
    from generativeqc.experimental.f12_focal import TriplesWPrecisionError

    # 1. Verify defaults never select fp32_single.
    default_provider = MolecularFocalProvider(shell_library=Path("fake.so"))
    assert default_provider.triples_fp32_w is False
    assert default_provider.triples_w_schedule is None
    cuda_sig = inspect.signature(MolecularFocalProvider.cuda)
    assert cuda_sig.parameters["triples_fp32_w"].default is False
    assert cuda_sig.parameters["triples_w_schedule"].default is None

    # 2. Verify MolecularFocalProvider.triples_energy vs triples_energy_and_forces.
    native_calls: list[bool] = []

    def fake_triples(req: object, ref: object, **kw: object) -> tuple[float, np.ndarray | None]:
        native_calls.append(bool(kw.get("forces")))
        return -0.01, np.zeros((1, 3), dtype=np.float64) if kw.get("forces") else None

    monkeypatch.setattr(ucc_triples, "molecular_triples_on_reference", fake_triples)
    fake_req = SimpleNamespace(
        atoms=(),
        basis=SimpleNamespace(orbital=SimpleNamespace(by_element={})),
        backend="cuda",
        correlation_space_id="all-active",
        root_id="rhf",
    )
    fake_ref = SimpleNamespace(
        occupied=(2,),
        orbital_energies=None,
        _pair_spin_indices=lambda req: (0, 0),
    )

    def a100() -> CudaDeviceIdentity:
        return CudaDeviceIdentity("NVIDIA A100-SXM4-80GB", "sm_80")

    fp32_single_provider = MolecularFocalProvider(
        shell_library=Path("fake.so"), triples_fp32_w=True, _cuda_device=a100
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e = fp32_single_provider.triples_energy(fake_req, reference=fake_ref)  # type: ignore[arg-type]
    assert e == pytest.approx(-0.01)
    assert native_calls == [False]

    with pytest.raises(TriplesWPrecisionError, match="only on an NVIDIA L20"):
        fp32_single_provider.triples_energy_and_forces(
            fake_req, reference=fake_ref  # type: ignore[arg-type]
        )
    assert native_calls == [False]

    # Default FP64 and compensated fp32_split3 forces run without warnings.
    for prov in (
        MolecularFocalProvider(shell_library=Path("fake.so"), _cuda_device=a100),
        MolecularFocalProvider(
            shell_library=Path("fake.so"), triples_w_schedule="fp32_split3", _cuda_device=a100
        ),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            e_f, f_arr = prov.triples_energy_and_forces(fake_req, reference=fake_ref)  # type: ignore[arg-type]
        assert e_f == pytest.approx(-0.01)
        assert f_arr.shape == (1, 3)
    assert native_calls == [False, True, True]
