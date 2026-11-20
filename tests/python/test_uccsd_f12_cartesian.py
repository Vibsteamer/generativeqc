"""Cartesian assembly checked by independent reoptimization at a displaced source."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from generativeqc._f12_cc_spin_sources import unrestricted_cc_f12_fixed_source_feeds
from generativeqc_compiler.cc.uccsd_equations import build_f12_program
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_extended_df_fock
from generativeqc_compiler.tensor import execute
from scipy.optimize import root
from test_cc_native_tensor_cuda import _fake_executor
from test_f12_unrestricted_singles import _singles

from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor
from tools.generativeqc_cc.uccsd_df_direction import unrestricted_cc_df_hamiltonian
from tools.generativeqc_cc.uccsd_f12_cartesian import (
    UCCSDF12CartesianResult,
    unrestricted_cc_f12_energy_forces,
)
from tools.generativeqc_cc.uccsd_solver import _representatives


def test_cartesian_energy_forces_and_stationarity(tmp_path: Path) -> None:
    rng = np.random.default_rng(52937)
    space = (2, 1, 3, 4, 2)
    oa, ob, obs, ext, aux = space
    layout = UCCSDLayout(oa, ob, obs - oa, obs - ob)
    cc = build_f12_program(layout)
    fock_graph = unrestricted_extended_df_fock(oa, ob, ext, aux)
    amplitudes = np.zeros(layout.packed_size)
    reference, dref = -1.78, 0.015

    def symmetric(shape: tuple[int, ...]) -> np.ndarray:
        x = rng.normal(scale=0.04, size=shape)
        return (x + x.swapaxes(-1, -2)) / 2

    raw = {"cross_overlap": np.eye(ext)}
    for s, n in (("alpha", oa), ("beta", ob)):
        raw["source/raw_coulomb_" + s] = symmetric((aux, ext, ext))
        raw["source/hcore_" + s] = np.diag(np.linspace(-1.0, 2.0, ext)) + symmetric(
            (ext, ext)
        )
        for op in (
            "geminal",
            "geminal_coulomb",
            "squared_geminal",
            "double_commutator",
        ):
            raw["metric/" + op] = symmetric((aux, aux))
            raw[f"raw/{op}/{s}"] = symmetric((aux, ext, ext))[:, :n]
    draw = {k: rng.normal(scale=0.015, size=v.shape) for k, v in raw.items()}
    for k, v in raw.items():
        if k.startswith(("source/", "metric/")):
            draw[k] = symmetric(v.shape)
    J = np.array([[1.3, 0.28], [0.28, 1.1]])
    dJ = symmetric((aux, aux))

    def point(x: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
        data = {k: v + x * draw[k] for k, v in raw.items()}
        metric = J + x * dJ
        for s, n in (("alpha", oa), ("beta", ob)):
            r = data["source/raw_coulomb_" + s][:, :n].reshape(aux, -1)
            data["source/fit_" + s] = np.linalg.solve(metric, r).reshape(aux, n, ext)
        return data, np.linalg.inv(np.linalg.cholesky(metric))

    def solve_point(
        x: float,
    ) -> tuple[
        float, float, np.ndarray, Callable[[np.ndarray], tuple[float, np.ndarray]]
    ]:
        data, w = point(x)
        fock = execute(
            fock_graph, {k[7:]: v for k, v in data.items() if k.startswith("source/")}
        ).outputs
        ordinary = unrestricted_cc_df_hamiltonian(space, data, fock, w, execute)
        fixed = unrestricted_cc_f12_fixed_source_feeds(space, data, fock, w, execute)
        e_si = sum(
            _singles(
                (fock["fock_" + s] + fock["fock_" + s].T) / 2,
                n,
                spin_weight=1,
                pullback=False,
            )[0]
            for s, n in (("alpha", oa), ("beta", ob))
        )

        def evaluate(t: np.ndarray) -> tuple[float, np.ndarray]:
            out = execute(cc, {**ordinary, **fixed, **layout.unpack(t)}).outputs
            return float(out["correlation_energy"]), _representatives(
                layout, out, residuals=True
            )

        result = root(lambda t: evaluate(t)[1], amplitudes, tol=1e-11)
        assert result.success
        return reference + x * dref + evaluate(result.x)[0], e_si, result.x, evaluate

    e, si, t, evaluate = solve_point(0.0)
    jac = []
    gradient = []
    for v in np.eye(t.size):
        h = 5e-5
        plus, minus = (evaluate(t + x * v) for x in (h, -h))
        jac.append((plus[1] - minus[1]) / (2 * h))
        gradient.append((plus[0] - minus[0]) / (2 * h))
    l = np.linalg.solve(np.asarray(jac), -np.asarray(gradient))
    factors = np.array([[0.1, 0.9, -1.4], [0.6, -0.4, 0.8]])
    data, w = point(0.0)
    tangent = draw.copy()
    for s, n in (("alpha", oa), ("beta", ob)):
        dr = draw["source/raw_coulomb_" + s][:, :n].reshape(aux, -1)
        fit = data["source/fit_" + s].reshape(aux, -1)
        tangent["source/fit_" + s] = np.linalg.solve(J, dr - dJ @ fit).reshape(
            aux, n, ext
        )

    def coordinates(a: int, k: int) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
        q = factors[a, k]
        return {key: q * v for key, v in tangent.items()}, q * dJ, q * dref

    def cartesian(
        multipliers: np.ndarray,
        include: bool,
        stationary_executor: CudaCCTensorExecutor | None = None,
    ) -> UCCSDF12CartesianResult:
        return unrestricted_cc_f12_energy_forces(
            space,
            data,
            w,
            t,
            multipliers,
            execute,
            natoms=2,
            coordinate_tangent=coordinates,
            reference_energy=reference,
            include_cabs_singles=include,
            singles=_singles,
            stationary_executor=stationary_executor,
            block_elements=31,
        )

    ep, sp, _, _ = solve_point(0.0001)
    em, sm, _, _ = solve_point(-0.0001)
    cuda, _, compiled = _fake_executor(tmp_path)
    for include in (False, True):
        val = cartesian(l, include, cuda if include else None)
        assert val.energy == pytest.approx(
            e + (si if include else 0.0), abs=2e-13, rel=0
        )
        expected = -(ep - em + (sp - sm if include else 0.0)) / 0.0002
        np.testing.assert_allclose(val.forces, factors * expected, rtol=0, atol=1e-8)
        assert (val.cabs_singles_forces is not None) == include
        assert not val.forces.flags.writeable
    assert len(compiled) == 2 and compiled[0] == cc.logical_hash
    with pytest.raises(ValueError, match="stationarity"):
        cartesian(l + 0.01, False)
