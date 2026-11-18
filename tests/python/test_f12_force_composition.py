"""Exercise the sign and single-reference ownership of the full molecular force."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.f12_molecular_force_composition import endpoint_arguments, full_result


def test_only_correlation_is_added_to_the_owned_reference() -> None:
    reference = SimpleNamespace(
        reference_energy=-75.0,
        increment_energy=-0.1,
        reference_forces=np.array([[3.0, -1.0, 2.0]]),
    )
    ordinary = {
        "passed": True,
        "backend": "cuda",
        "reference_energy_Eh": -75.0,
        "correlation_energy_Eh": -0.2,
        "correlation_forces_Eh_per_bohr": [[0.5, 1.0, 1.0]],
        "mp2_energy_Eh": -75.2,
    }
    energy, forces = full_result(
        reference, [[0.1, 1.0, 0.5]], [[1.0, 2.0, 0.1]], ordinary
    )
    assert energy == pytest.approx(-75.3)
    np.testing.assert_allclose(forces, [[2.4, -3.0, 2.4]])
    with pytest.raises(ValueError, match="different RHF"):
        full_result(
            reference, forces, forces, {**ordinary, "reference_energy_Eh": -74.0}
        )
    with pytest.raises(ValueError, match="Cartesian"):
        full_result(reference, np.zeros(3), forces, ordinary)


def test_numpy_scalar_checkpoint_survives_later_invalid_report(tmp_path: Path) -> None:
    import json

    from benchmarks.f12_molecular_force_composition import write_measurement

    path = tmp_path / "receipt.json"
    partial = {
        "passed": np.bool_(True),
        "residual": np.float64(1e-12),
        "rows": np.int64(10),
    }
    write_measurement(path, partial)
    assert json.loads(path.read_text()) == {
        "passed": True,
        "residual": 1e-12,
        "rows": 10,
    }
    raw = path.read_bytes()
    for bad, exception in ((np.nan, ValueError), (np.array([1.0]), TypeError)):
        with pytest.raises(exception):
            write_measurement(path, {"value": bad})
        assert path.read_bytes() == raw


@pytest.mark.parametrize("selected", [False, True])
def test_full_rank_cabs_response_is_opt_in(
    monkeypatch: pytest.MonkeyPatch, selected: bool
) -> None:
    paths = [
        "scf-inputs",
        "component-library",
        "session-library",
        "cache",
        "output",
        "layout",
        "sink-library",
        "panel",
    ]
    argv = ["molecular-ef", "--arch", "sm_89", "--budget-mib", "16384"]
    for name in paths:
        argv.extend(["--" + name, name])
    if selected:
        argv.append("--full-rank-cabs-response")
    monkeypatch.setattr(sys, "argv", argv)
    assert endpoint_arguments().full_rank_cabs_response is selected


def test_isolated_singles_adjoint_and_pair_difference_at_two_steps() -> None:
    from generativeqc_compiler.tensor.interpreter import execute

    from benchmarks.f12_ao_response import OPERATORS, evaluate

    rng = np.random.default_rng(319)
    occ, orbitals, aux = 2, 5, 3
    data = {
        "space": [occ, 3, orbitals],
        "coefficients": np.eye(orbitals),
        "hcore_ao": np.diag([-4.0, -2.0, 1.0, 3.0, 4.0]),
        "metric": np.eye(aux),
        "metric_whitener": np.eye(aux),
    }
    data["hcore_ao"][0, 3] = data["hcore_ao"][3, 0] = 0.15
    for op in ("coulomb", *OPERATORS):
        block = rng.normal(scale=0.08, size=(aux, orbitals, orbitals))
        data["ao_" + op] = block + block.swapaxes(1, 2)
    for op in OPERATORS:
        data["metric_" + op] = np.eye(aux)

    def singles(
        fock: np.ndarray, occupied: int, *, pullback: bool
    ) -> tuple[float, np.ndarray | None]:
        # Direct Sylvester-equation solution in a noncanonical orbital frame.
        a, b = fock[:occupied, :occupied], fock[occupied:, occupied:]
        coupling = fock[:occupied, occupied:]
        op = np.kron(np.eye(len(b)), a) - np.kron(b.T, np.eye(occupied))
        x = np.linalg.solve(op, coupling.flatten("F")).reshape(
            coupling.shape, order="F"
        )
        grad = 2 * np.block([[-x @ x.T, x], [x.T, x.T @ x]]) if pullback else None
        return 2 * np.vdot(coupling, x), grad

    def reference_singles(v: dict) -> float:
        c, hc, ao = (v[name] for name in ("coefficients", "hcore_ao", "ao_coulomb"))
        raw = c.T @ ao @ c
        fitted = np.linalg.solve(v["metric"], raw[:, :occ].reshape(aux, -1)).reshape(
            aux, occ, -1
        )
        fock = c.T @ hc @ c + 2 * np.einsum("aii,apq->pq", fitted[:, :, :occ], raw)
        fock -= np.einsum("aip,aiq->pq", fitted, raw[:, :occ])
        return singles(fock, occ, pullback=False)[0]

    total, total_bars, details = evaluate(data, execute, pullback=True, singles=singles)
    single, bars, _ = evaluate(
        data, execute, pullback=True, singles=singles, singles_only=True
    )
    assert single == pytest.approx(reference_singles(data), abs=1e-13)
    assert set(bars) == {"hcore_ao", "coefficients", "metric", "ao_coulomb"}
    pair = details["energy_components"]["f12_pair_energy_Eh"]
    assert abs(pair) > 1e-6 and abs(single) > 1e-4
    assert total == pytest.approx(single + pair, abs=1e-13)
    direction = {}
    for name in total_bars:
        v = rng.normal(scale=0.08, size=data[name].shape)
        if name != "coefficients":
            v = 0.5 * (v + v.swapaxes(-1, -2))
        direction[name] = v

    ds = sum(np.vdot(bar, direction[k]) for k, bar in bars.items())
    dp = sum(np.vdot(bar, direction[k]) for k, bar in total_bars.items()) - ds
    for h in (4e-4, 2e-4):
        es, ep = [], []
        for step in (h, -h):
            displaced = {
                **data,
                **{k: data[k] + step * d for k, d in direction.items()},
            }
            displaced["metric_whitener"] = np.linalg.inv(
                np.linalg.cholesky(displaced["metric"])
            )
            es.append(reference_singles(displaced))
            ep.append(evaluate(displaced, execute, pullback=False)[0])
        assert ds == pytest.approx((es[0] - es[1]) / (2 * h), abs=2e-8)
        assert dp == pytest.approx((ep[0] - ep[1]) / (2 * h), abs=2e-8)


def test_split_force_removes_rhf_nuclei_from_increment() -> None:
    from benchmarks.f12_molecular_force_composition import component_forces

    hf = np.array([[0.2, 0.4, 0.6], [-0.2, -0.4, -0.6]])
    corr, pairs, singles = 0.1 * hf, 0.3 * hf, -0.2 * hf
    one, two = 2 * hf, -0.4 * hf
    total = hf + corr + pairs + singles
    reference = SimpleNamespace(reference_forces=hf + singles + one + two)
    values = component_forces(total, hf, corr, reference, one, two)
    np.testing.assert_allclose(values["cabs_singles"], singles)
    np.testing.assert_allclose(values["f12_pair"], pairs)
    np.testing.assert_allclose(sum(map(np.asarray, values.values())), total)
    with pytest.raises(ValueError, match="component forces"):
        component_forces(total, np.ones((2, 1)), corr, reference, one, two)
