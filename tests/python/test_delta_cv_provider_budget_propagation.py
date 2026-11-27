"""Endpoint device budgets reach every CUDA conventional DF-MP2 Delta-CV kernel.

The production provider is driven end to end while the native library is a
fake C ABI that applies the CUDA admission rule: a positive ``budget_bytes`` is
the device cap and ``0`` selects the fixed per-kernel defaults (1 GiB chunk,
2.5 GiB pair, 2.8 GiB J/K) of ``f12_dfmp2_cv_cuda.cu``. A tight provider budget
must reject the first kernel, and a provider budget above every old fixed cap
must admit a workload the fixed caps would refuse.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from generativeqc._f12_focal_molecular import MolecularFocalProvider
    from generativeqc.experimental.f12_focal import FocalRequest

_MIB = 1 << 20
_KERR_OOM = 4
_OLD_FIXED_CAP = {
    "df_jk_fock": 1024 * _MIB,
    "mp2_factors_fwd": 1024 * _MIB,
    "mp2_factors_rev": 1024 * _MIB,
    "cphf_matvec": 1024 * _MIB,
    "mp2_pairs": 2560 * _MIB,
    "jk_mo_fock_fwd": 2800 * _MIB,
    "jk_fock_pullback": 2800 * _MIB,
    "nuclear_source_weights": 2800 * _MIB,
}

# Leading C-int count and {pointer offset: output shape} for each CUDA entry.
_LAYOUT = {
    "df_jk_fock": (6, lambda d: {4: (d[1], d[1]), 5: (d[1], d[1]), 6: (d[1], d[1])}),
    "mp2_factors_fwd": (5, lambda d: {4: (d[0], d[2], d[3])}),
    "mp2_factors_rev": (
        5,
        lambda d: {
            6: (d[1], d[2]),
            7: (d[1], d[3]),
            8: (d[0], d[1], d[1]),
            9: (d[0], d[0]),
        },
    ),
    "mp2_pairs": (
        10,
        lambda d: {
            6: (3,),
            7: (3,),
            8: (d[2], d[3], d[5]),
            9: (d[2], d[4], d[6]),
            10: (d[3], d[3]),
            11: (d[4], d[4]),
            12: (d[5], d[5]),
            13: (d[6], d[6]),
        },
    ),
    "jk_mo_fock_fwd": (
        6,
        lambda d: {
            5: (d[0], d[1], d[1]),
            6: (d[0], d[1], d[1]),
            7: (d[1], d[1]),
            8: (d[1], d[1]),
        },
    ),
    "jk_fock_pullback": (
        6,
        lambda d: {
            9: (d[1], d[1]),
            10: (d[1], d[1]),
            11: (d[1], d[1]),
            12: (d[0], d[1], d[1]),
            13: (d[0], d[0]),
        },
    ),
    "cphf_matvec": (6, lambda d: {6: (d[1] - d[2], d[2]), 7: (d[1] - d[3], d[3])}),
    "nuclear_source_weights": (
        6,
        lambda d: {
            9: (d[1], d[1]),
            10: (d[1], d[1]),
            11: (d[0], d[1], d[1]),
            12: (d[0], d[0]),
        },
    ),
}

# Spin-summed (same, opposite, same) pair energies written by the fake pair kernel.
_CV_PAIRS = (-0.125, -0.25, -0.125)
_FC_PAIRS = (-0.0625, -0.125, -0.0625)
_EPS = np.array([-20.5, -1.3, -0.7, -0.55, -0.5, 0.6, 0.75])
_N, _Q, _OCC = 7, 4, 5


class _FakeKernel:
    def __init__(self, library: _FakeDeltaCVLibrary, key: str) -> None:
        self.library = library
        self.key = key
        self.argtypes = None
        self.restype = None

    def __call__(self, *args: Any) -> int:
        n_int, layout = _LAYOUT[self.key]
        dims = [int(a) for a in args[:n_int]]
        budget = int(args[n_int])
        pointers = args[n_int + 1 :]
        self.library.calls.append((self.key, budget))
        cap = budget if budget > 0 else _OLD_FIXED_CAP[self.key]
        if self.library.required_bytes > cap:
            return _KERR_OOM
        for offset, shape in layout(dims).items():
            if pointers[offset] and all(shape):
                np.ctypeslib.as_array(pointers[offset], shape=shape)[...] = 0.0
        if self.key == "mp2_pairs":
            np.ctypeslib.as_array(pointers[6], shape=(3,))[...] = _CV_PAIRS
            np.ctypeslib.as_array(pointers[7], shape=(3,))[...] = _FC_PAIRS
        elif self.key == "jk_mo_fock_fwd":
            n = dims[1]
            np.ctypeslib.as_array(pointers[7], shape=(n, n))[...] = np.diag(_EPS)
        return 0


class _FakeDeltaCVLibrary:
    """CUDA C ABI double that records the budget each kernel receives."""

    def __init__(self, required_bytes: int) -> None:
        self.required_bytes = required_bytes
        self.calls: list[tuple[str, int]] = []
        for key in _LAYOUT:
            setattr(self, f"gqc_f12_cv_{key}_cuda", _FakeKernel(self, key))

    def kernels(self) -> set[str]:
        return {key for key, _ in self.calls}

    def budgets(self) -> set[int]:
        return {budget for _, budget in self.calls}


def _water_requests() -> tuple[FocalRequest, FocalRequest]:
    from generativeqc import Atom
    from generativeqc._f12_focal_cv import conventional_mp2_request
    from generativeqc.experimental.f12_focal import FocalBasis

    basis = FocalBasis(
        2,
        "delta-cv-budget",
        "sto-3g",
        jk_fit="def2-svp",
        mp2_fit="def2-svp",
        f12_fit="def2-svp",
        cabs="def2-svp",
        beta=0.9,
    )
    atoms = (
        Atom(8, (0.0, 0.0, 0.0)),
        Atom(1, (0.0, 1.43, 1.1)),
        Atom(1, (0.0, -1.43, 1.1)),
    )
    requests = [
        conventional_mp2_request(
            atoms, basis, root_id="rhf", correlation_space_id=space, backend="cuda"
        )
        for space in ("cv-active", "frozen-core")
    ]
    return requests[0], requests[1]


def _panel(fit_id: str) -> SimpleNamespace:
    rng = np.random.default_rng(7)
    three_center = rng.standard_normal((_Q, _N, _N))
    three_center = 0.5 * (three_center + three_center.transpose(0, 2, 1))
    source = SimpleNamespace(
        three_center=three_center,
        whitener=np.eye(_Q),
        metric=np.eye(_Q),
    )
    return SimpleNamespace(
        fit_id=fit_id,
        data={"ao_coulomb": three_center},
        whitener=np.eye(_Q),
        obs=lambda request, n: source,
    )


@pytest.fixture
def cuda_delta_cv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Install the fake CUDA Delta-CV library and stub the non-Delta-CV device stages."""
    from generativeqc import _canonical_frame
    from generativeqc import _f12_cv_dfmp2_native as native
    from generativeqc import _f12_focal_one_electron as one_electron
    from generativeqc import _f12_focal_reference as reference
    from generativeqc._f12_focal_molecular import MolecularFocalProvider

    for key in (
        "GENERATIVEQC_F12_REQUIRE_AOT_CV",
        "GENERATIVEQC_F12_REQUIRE_AOT_DELTACV",
        "GENERATIVEQC_F12_DISABLE_AOT_CV",
        "GENERATIVEQC_F12_DISABLE_AOT_DELTACV",
    ):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "libfake_delta_cv_cuda.so"
    canonical_budgets: list[int] = []

    def install(
        required_bytes: int, budget_mib: int
    ) -> tuple[_FakeDeltaCVLibrary, MolecularFocalProvider]:
        library = _FakeDeltaCVLibrary(required_bytes)
        monkeypatch.setitem(native._CDLL_CACHE, path.resolve(), library)
        provider = MolecularFocalProvider(
            shell_library=path,
            one_electron_library=path,
            metric_library=path,
            response_library=object(),
            canonical_library=object(),
            run=lambda graph, inputs: pytest.fail(
                "Delta-CV AOT path must not run TensorIR"
            ),
            budget_mib=budget_mib,
        )
        return library, provider

    def fake_reference(
        request: FocalRequest, **_options: object
    ) -> reference.FocalReference:
        hcore = np.diag(_EPS)
        density = np.zeros((1, _N, _N))
        density[0, :_OCC, :_OCC] = 2.0 * np.eye(_OCC)
        basis = request.basis
        assert basis.jk_fit is not None
        return reference.FocalReference(
            energy=0.0,
            forces=None,
            overlap=np.eye(_N),
            hcore=hcore,
            coefficients=np.eye(_N)[None, :, :],
            orbital_energies=_EPS[None, :],
            density=density,
            fock=hcore[None, :, :],
            occupied=(_OCC,),
            iterations=1,
            executed_backend=request.backend,
            surface_id=request.surface_id,
            basis_id=basis.basis_id,
            orbital_id=basis.orbital.identity,
            jk_fit_id=basis.jk_fit.identity,
        )

    monkeypatch.setattr(reference, "df_reference", fake_reference)
    monkeypatch.setattr(
        one_electron,
        "one_electron_sources",
        lambda request, **k: SimpleNamespace(
            obs_ao_count=_N,
            overlap=np.eye(_N),
            hcore=np.diag(_EPS),
            d_overlap=np.zeros((3 * len(request.atoms), _N, _N)),
            d_hcore=np.zeros((3 * len(request.atoms), _N, _N)),
        ),
    )
    monkeypatch.setattr(
        MolecularFocalProvider, "_panel", lambda self, request, role, **k: _panel(role)
    )
    monkeypatch.setattr(
        MolecularFocalProvider,
        "_gradient",
        lambda self, request, panel, weights, **k: np.zeros((len(request.atoms), 3)),
    )

    def canonical_frame_seed(
        rotation: np.ndarray,
        eps: np.ndarray,
        occupied: int,
        *,
        maximum_bytes: int,
        library: object,
    ) -> tuple[np.ndarray]:
        canonical_budgets.append(maximum_bytes)
        return (np.zeros_like(rotation),)

    monkeypatch.setattr(_canonical_frame, "canonical_frame_seed", canonical_frame_seed)
    return SimpleNamespace(install=install, canonical_budgets=canonical_budgets)


def _force_focal_source_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    from generativeqc import _f12_focal_cv as focal_cv
    from generativeqc import _f12_focal_reference as reference

    def unavailable(*args: object, **kwargs: object) -> NoReturn:
        raise NotImplementedError("ECP execution supports orbital s/p/d/f shells")

    monkeypatch.setattr(reference, "df_reference", unavailable)
    monkeypatch.setattr(
        focal_cv,
        "_is_eligible_for_focal_source_df_hf_fallback",
        lambda request, exc: True,
    )


_ENTRIES = {
    "energy": ("delta_cv_mp2_energy", False),
    "energy_and_forces": ("delta_cv_mp2_energy_and_forces", False),
    "focal_source_reference": ("delta_cv_mp2_energy", True),
}
_ENERGY_KERNELS = {"mp2_factors_fwd", "mp2_pairs"}
_FORCE_KERNELS = {
    "mp2_factors_fwd",
    "mp2_pairs",
    "mp2_factors_rev",
    "jk_mo_fock_fwd",
    "jk_fock_pullback",
    "cphf_matvec",
    "nuclear_source_weights",
}


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
def test_tight_provider_budget_rejects_delta_cv_kernels(
    entry: str, cuda_delta_cv: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, focal_source_reference = _ENTRIES[entry]
    if focal_source_reference:
        _force_focal_source_reference(monkeypatch)
    library, provider = cuda_delta_cv.install(required_bytes=256 * _MIB, budget_mib=128)
    req_cv, req_fc = _water_requests()

    with pytest.raises(RuntimeError, match="failed with status 4"):
        getattr(provider, method)(req_cv, req_fc)

    assert library.calls, "the tight budget must reach a native kernel"
    assert library.budgets() == {128 * _MIB}
    first = "df_jk_fock" if focal_source_reference else "mp2_factors_fwd"
    assert library.calls[0][0] == first


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
def test_provider_budget_above_old_fixed_caps_is_admitted(
    entry: str, cuda_delta_cv: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, focal_source_reference = _ENTRIES[entry]
    if focal_source_reference:
        _force_focal_source_reference(monkeypatch)
    required = 3072 * _MIB
    assert required > max(_OLD_FIXED_CAP.values())
    library, provider = cuda_delta_cv.install(required_bytes=required, budget_mib=4096)
    req_cv, req_fc = _water_requests()

    result = getattr(provider, method)(req_cv, req_fc)

    assert library.budgets() == {4096 * _MIB}
    assert result.cv_frozen == (0, 0) and result.fc_frozen == (1, 1)
    assert result.delta_ordinary_mp2_Eh == pytest.approx(
        sum(_CV_PAIRS) - sum(_FC_PAIRS)
    )
    expected = _FORCE_KERNELS if method.endswith("forces") else _ENERGY_KERNELS
    if focal_source_reference:
        expected = expected | {"df_jk_fock"}
        assert result.reference_provenance == "focal_source_df_hf_fallback"
    assert library.kernels() == expected
    if method.endswith("forces"):
        assert result.delta_forces.shape == (3, 3)
        assert cuda_delta_cv.canonical_budgets
        assert set(cuda_delta_cv.canonical_budgets) == {4096 * _MIB}


def test_ordinary_mp2_factor_build_uses_the_endpoint_budget(
    cuda_delta_cv: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import _f12_mp2_pair_native as pair_native
    from generativeqc._f12_focal_energy import _ordinary_energies

    monkeypatch.setattr(pair_native, "bind_ordinary_df_mp2", lambda **k: None)
    source = _panel("mp2").obs(None, _N)
    reference = SimpleNamespace(
        occupied=(_OCC,),
        orbital_energies=_EPS[None, :],
        coefficients=np.eye(_N)[None, :, :],
        hcore=np.diag(_EPS),
        executed_backend="cuda",
    )

    def run(graph: object, inputs: object) -> SimpleNamespace:
        return SimpleNamespace(outputs={"energy": np.float64(-0.01)})

    for required, budget_mib, admitted in (
        (256 * _MIB, 128, False),
        (3072 * _MIB, 4096, True),
    ):
        library, provider = cuda_delta_cv.install(
            required_bytes=required, budget_mib=budget_mib
        )
        call = partial(
            _ordinary_energies,
            reference,
            (0, 0),
            source,
            run,
            frozen_occupied=(1, 1),
            shell_library=provider.shell_library,
            budget_mib=budget_mib,
            cuda=True,
        )
        if admitted:
            assert call() == pytest.approx((-0.01, -0.01, -0.01))
        else:
            with pytest.raises(RuntimeError, match="failed with status 4"):
                call()
        assert library.kernels() == {"mp2_factors_fwd"}
        assert library.budgets() == {budget_mib * _MIB}


def test_cuda_binding_requires_an_explicit_positive_budget(
    cuda_delta_cv: SimpleNamespace,
) -> None:
    from generativeqc._f12_cv_dfmp2_native import bind_delta_cv_native

    library, provider = cuda_delta_cv.install(required_bytes=0, budget_mib=64)
    with pytest.raises(ValueError, match="explicit positive budget_bytes"):
        bind_delta_cv_native(provider.shell_library, host_reference=False)
    with pytest.raises(ValueError, match="explicit positive budget_bytes"):
        bind_delta_cv_native(
            provider.shell_library, host_reference=False, budget_bytes=0
        )
    bound = bind_delta_cv_native(
        provider.shell_library, host_reference=False, budget_bytes=64 * _MIB
    )
    assert bound is not None and bound.budget_bytes == 64 * _MIB
    with pytest.raises(ValueError, match="explicit positive budget_bytes"):
        bound.run_mp2_factors_fwd(
            np.zeros((_Q, _N, _N)),
            np.eye(_Q),
            np.eye(_N)[:, :2],
            np.eye(_N)[:, 2:],
            budget_bytes=0,
        )
    assert library.calls == []
