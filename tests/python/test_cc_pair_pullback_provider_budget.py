"""Endpoint device budgets reach the CC-F12 V+B-X, W and C pair pullback kernels.

The CUDA pair-pullback ABI maps ``budget_bytes == 0`` to a fixed 4 GiB device
cap (``f12_pair_pullback_cuda.cu``). The CUDA doubles below apply that admission
rule and then run the host-reference kernel with the same arguments, so an
admitted call returns the real pullback.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from conftest import f12_host_shell

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from generativeqc._f12_focal_cc_sources import FocalUCCEquationSources
    from generativeqc._f12_focal_molecular import MolecularFocalProvider
    from generativeqc.experimental.f12_focal import FocalRequest

_MIB = 1 << 20
_OLD_FIXED_CAP = 4096 * _MIB
_KERR_OOM = -3
_BUDGET_ARGUMENT = {"w": 7, "c": 7, "scalar": 9}


@pytest.fixture(scope="module")
def host_sink() -> Path:
    return f12_host_shell()


def _install_cuda_double(
    monkeypatch: pytest.MonkeyPatch, host_sink: Path, required_bytes: int
) -> list[tuple[str, int]]:
    from generativeqc import _f12_cc_pair_pullback_native as native

    real_bind = native.bind_pair_pullback
    calls: list[tuple[str, int]] = []

    def bind(
        library: Path | None, kind: str, host_reference: bool
    ) -> Callable[..., int] | None:
        if library is None:
            return None
        assert not host_reference, "the CUDA double serves CUDA bindings only"
        host = real_bind(host_sink, kind, True)

        def kernel(*args: Any) -> int:
            budget = int(args[_BUDGET_ARGUMENT[kind]])
            calls.append((kind, budget))
            if required_bytes > (budget if budget > 0 else _OLD_FIXED_CAP):
                return _KERR_OOM
            return host(*args)

        return kernel

    monkeypatch.setattr(native, "bind_pair_pullback", bind)
    return calls


def _synthetic_frame(
    oa: int = 3, ob: int = 2, frozen: tuple[int, int] = (1, 0)
) -> SimpleNamespace:
    rng = np.random.default_rng(4242 + oa * 7 + ob)
    obs, ext, aux = 6, 10, 8
    space = (oa, ob, obs, ext, aux)

    def sym(a: np.ndarray) -> np.ndarray:
        return 0.5 * (a + a.swapaxes(-1, -2))

    fock = {
        "fock_alpha": sym(rng.normal(scale=0.3, size=(ext, ext))),
        "fock_beta": sym(rng.normal(scale=0.3, size=(ext, ext))),
        "exchange_alpha": sym(rng.normal(scale=0.2, size=(ext, ext))),
        "exchange_beta": sym(rng.normal(scale=0.2, size=(ext, ext))),
    }
    fit_obs = {
        "alpha": rng.normal(scale=0.2, size=(aux, obs, ext)),
        "beta": rng.normal(scale=0.2, size=(aux, obs, ext)),
    }
    data = {
        "cross_overlap": rng.normal(scale=0.2, size=(ext, ext)),
        "source/fit_alpha": fit_obs["alpha"][:, :oa, :],
        "source/fit_beta": fit_obs["beta"][:, :ob, :],
        "source/raw_coulomb_alpha": rng.normal(scale=0.2, size=(aux, obs, ext)),
        "source/raw_coulomb_beta": rng.normal(scale=0.2, size=(aux, obs, ext)),
    }
    for name in ("geminal", "geminal_coulomb", "squared_geminal", "double_commutator"):
        data["metric/" + name] = sym(rng.normal(scale=0.2, size=(aux, aux)))
        data[f"raw/{name}/alpha"] = rng.normal(scale=0.2, size=(aux, oa, ext))
        data[f"raw/{name}/beta"] = rng.normal(scale=0.2, size=(aux, ob, ext))
    na, nb = oa - frozen[0], ob - frozen[1]
    va, vb = obs - oa, obs - ob
    weights = {
        "scalar": 1.25,
        "w": {
            "w_vo_aaaa": rng.normal(scale=0.1, size=(na, na, va, na)),
            "w_vo_bbbb": rng.normal(scale=0.1, size=(nb, nb, vb, nb)),
            "w_vv_abab": rng.normal(scale=0.1, size=(na, nb, va, vb)),
        },
        "c": {
            "c_f12_aaaa": rng.normal(scale=0.1, size=(na, na, va, va)),
            "c_f12_bbbb": rng.normal(scale=0.1, size=(nb, nb, vb, vb)),
            "c_f12_abab": rng.normal(scale=0.1, size=(na, nb, va, vb)),
        },
    }
    return SimpleNamespace(
        space=space,
        frozen=frozen,
        data=data,
        fock=fock,
        fit_obs=fit_obs,
        weights=weights,
    )


def _accumulate(
    kind: str, frame: SimpleNamespace, **native: Any
) -> dict[str, np.ndarray]:
    from generativeqc._f12_cc_c_nuclear import _accumulate_c_mo_pullback
    from generativeqc._f12_cc_scalar_nuclear import _accumulate_scalar_mo_pullback
    from generativeqc._f12_cc_w_nuclear import _accumulate_w_mo_pullback
    from generativeqc_compiler.tensor import execute

    space, frozen, data, fock = frame.space, frame.frozen, frame.data, frame.fock
    fit_obs, weights = frame.fit_obs, frame.weights
    run = partial(execute, max_bytes=512 * _MIB)
    dbar = {k: np.zeros_like(v) for k, v in data.items()}
    fbar = {k: np.zeros_like(v) for k, v in fock.items()}
    fobs = {k: np.zeros_like(v) for k, v in fit_obs.items()}
    out: dict[str, np.ndarray] = {}
    if kind == "scalar":
        out["energies"] = np.asarray(
            _accumulate_scalar_mo_pullback(
                space,
                data,
                fock,
                weights["scalar"],
                run,
                dbar,
                fbar,
                frozen_occupied=frozen,
                **native,
            )
        )
    elif kind == "w":
        _accumulate_w_mo_pullback(
            space,
            data,
            fit_obs,
            weights["w"],
            run,
            dbar,
            fobs,
            frozen_occupied=frozen,
            **native,
        )
    else:
        _accumulate_c_mo_pullback(
            space,
            data,
            fock,
            weights["c"],
            run,
            dbar,
            fbar,
            frozen_occupied=frozen,
            **native,
        )
    for prefix, bars in (("dbar", dbar), ("fbar", fbar), ("fit_obs_bar", fobs)):
        out.update({f"{prefix}:{k}": v for k, v in bars.items()})
    return out


_KINDS = ("scalar", "w", "c")


@pytest.mark.parametrize("kind", _KINDS)
def test_cuda_pair_pullback_rejects_a_tight_endpoint_budget(
    kind: str, host_sink: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_cuda_double(monkeypatch, host_sink, required_bytes=2 * _MIB)
    with pytest.raises(RuntimeError, match="failed with status -3"):
        _accumulate(
            kind,
            _synthetic_frame(),
            native_library=host_sink,
            host_reference=False,
            budget_bytes=_MIB,
        )
    assert calls and {budget for _, budget in calls} == {_MIB}


@pytest.mark.parametrize("kind", _KINDS)
def test_cuda_pair_pullback_admits_budget_above_old_fixed_cap(
    kind: str, host_sink: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _synthetic_frame()
    tensorir = _accumulate(kind, frame, native_library=None)
    calls = _install_cuda_double(monkeypatch, host_sink, required_bytes=5120 * _MIB)
    budget = 6144 * _MIB
    cuda = _accumulate(
        kind, frame, native_library=host_sink, host_reference=False, budget_bytes=budget
    )
    assert {k for k, _ in calls} == {kind}
    assert {b for _, b in calls} == {budget}
    assert cuda.keys() == tensorir.keys()
    for key, value in tensorir.items():
        np.testing.assert_allclose(
            cuda[key], value, atol=1e-12, rtol=1e-12, err_msg=key
        )


@pytest.mark.parametrize("kind", _KINDS)
def test_cuda_pair_pullback_requires_an_explicit_budget(
    kind: str, host_sink: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_cuda_double(monkeypatch, host_sink, required_bytes=0)
    with pytest.raises(ValueError, match="explicit positive budget_bytes"):
        _accumulate(
            kind, _synthetic_frame(), native_library=host_sink, host_reference=False
        )
    assert calls == []


@pytest.mark.parametrize("kind", _KINDS)
def test_tensorir_and_host_pair_pullbacks_do_not_consume_a_device_budget(
    kind: str, host_sink: Path
) -> None:
    frame = _synthetic_frame()
    tensorir = _accumulate(kind, frame, native_library=None)
    host = _accumulate(kind, frame, native_library=host_sink, host_reference=True)
    for key, value in tensorir.items():
        np.testing.assert_allclose(
            host[key], value, atol=1e-12, rtol=1e-12, err_msg=key
        )


def _fixed_sources(
    host_sink: Path, root_id: str, budget_mib: int
) -> tuple[MolecularFocalProvider, FocalRequest, Any, FocalUCCEquationSources]:
    from generativeqc import Atom
    from generativeqc._f12_focal_cc_sources import FocalUCCEquationSources
    from generativeqc._f12_focal_molecular import MolecularFocalProvider
    from generativeqc.experimental.f12_focal import (
        FocalBasis,
        FocalSeries,
        plan_singlepoint,
    )

    basis = FocalBasis(
        2,
        "cc-pair-pullback-budget",
        "sto-3g",
        jk_fit="def2-svp",
        mp2_fit="def2-svp",
        f12_fit="def2-svp",
        cabs="def2-svp",
        beta=1.0,
    )
    if root_id == "rhf":
        atoms = (
            Atom("O", (0.0, 0.0, 0.0)),
            Atom("H", (1.43, 0.0, 1.11)),
            Atom("H", (-1.43, 0.0, 1.11)),
        )
        multiplicity = 1
    else:
        atoms = (Atom("O", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 1.83)))
        multiplicity = 2
    plan = plan_singlepoint(
        atoms,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=0,
        multiplicity=multiplicity,
        root_id=root_id,
        correlation_space_id="frozen-core",
        backend="cpu_reference",
    )
    request = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    provider = MolecularFocalProvider(shell_library=host_sink, budget_mib=budget_mib)
    fixed = provider.ucc_fixed_sources(request)
    return (
        provider,
        request,
        fixed,
        FocalUCCEquationSources(fixed, basis.mp2_fit.identity, {}),
    )


def _record_host_pair_pullbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, int]]:
    from generativeqc import _f12_cc_pair_pullback_native as native

    real_bind = native.bind_pair_pullback
    calls: list[tuple[str, int]] = []

    def bind(
        library: Path | None, kind: str, host_reference: bool
    ) -> Callable[..., int] | None:
        host = real_bind(library, kind, host_reference)
        if host is None:
            return None

        def kernel(*args: Any) -> int:
            calls.append((kind, int(args[_BUDGET_ARGUMENT[kind]])))
            return host(*args)

        return kernel

    monkeypatch.setattr(native, "bind_pair_pullback", bind)
    return calls


@pytest.mark.parametrize("root_id", ["rhf", "uhf"])
def test_fixed_cc_force_callers_pass_the_provider_budget(
    root_id: str, host_sink: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc._f12_focal_cc_fixed import fixed_ucc_source_forces
    from generativeqc._f12_focal_rcc_f12b import spatial_rhf_fixed_cc_forces
    from generativeqc_compiler.cc.uccsd_equations import build_f12_program
    from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout

    budget_mib = 1536
    provider, request, fixed, sources = _fixed_sources(host_sink, root_id, budget_mib)
    ref = fixed.reference
    oa, ob = (ref.occupied[i] for i in ref._pair_spin_indices(request))
    fa, fb = fixed.frozen_occupied
    n = ref.coefficients.shape[-1]
    program = build_f12_program(UCCSDLayout(oa - fa, ob - fb, n - oa, n - ob))
    rng = np.random.default_rng(31)
    weights = {
        k: (
            np.asarray(1.0)
            if k == "fixed_geminal_energy"
            else rng.normal(0.0, 0.04, size=fixed.feeds[k].shape)
        )
        for k in sorted(
            node.attrs["name"]
            for node in program.live_nodes
            if node.op == "input" and node.attrs["name"] in fixed.feeds
        )
    }

    calls = _record_host_pair_pullbacks(monkeypatch)
    forces = fixed_ucc_source_forces(provider, request, sources, weights)
    assert np.isfinite(forces).all()
    assert {kind for kind, _ in calls} == set(_KINDS)
    assert {budget for _, budget in calls} == {budget_mib << 20}

    if root_id == "rhf":
        calls.clear()
        o_act, n_vir = oa - fa, n - oa
        t2 = rng.normal(0.0, 0.03, size=(o_act, o_act, n_vir, n_vir))
        l2 = rng.normal(0.0, 0.03, size=(o_act, o_act, n_vir, n_vir))
        forces = spatial_rhf_fixed_cc_forces(
            provider,
            request,
            ref,
            cabs_rank=fixed.cabs_rank,
            fixed_geminal_energy=float(fixed.feeds["fixed_geminal_energy"]),
            t1=rng.normal(0.0, 0.04, size=(o_act, n_vir)),
            t2=0.5 * (t2 + t2.transpose(1, 0, 3, 2)),
            lambda1=rng.normal(0.0, 0.04, size=(o_act, n_vir)),
            lambda2=0.5 * (l2 + l2.transpose(1, 0, 3, 2)),
            frozen_occupied=fa,
        )
        assert np.isfinite(forces).all()
        assert {kind for kind, _ in calls} == set(_KINDS)
        assert {budget for _, budget in calls} == {budget_mib << 20}
