"""Unrestricted composition against a fixed-frame physical Sylvester oracle."""

from collections.abc import Iterator

import numpy as np
import pytest
from generativeqc._f12_unrestricted import (
    UnrestrictedCabsSingles,
    unrestricted_cabs_energy,
)
from generativeqc_compiler.tensor import Execution, Program, execute
from scipy.linalg import solve_sylvester


def _singles(
    matrix: np.ndarray, occupied: int, *, spin_weight: int, pullback: bool
) -> tuple[float, np.ndarray | None]:
    oo, vv, ov = (
        matrix[:occupied, :occupied],
        matrix[occupied:, occupied:],
        matrix[:occupied, occupied:],
    )
    if not occupied:
        return 0.0, np.zeros_like(matrix) if pullback else None
    amplitudes = solve_sylvester(vv, -oo, ov.T)
    energy = -np.sum(amplitudes.T * ov)
    bar = None
    if pullback:
        bar = np.block(
            [
                [-amplitudes.T @ amplitudes, -amplitudes.T],
                [-amplitudes, amplitudes @ amplitudes.T],
            ]
        )
    return spin_weight * energy, None if bar is None else spin_weight * bar


@pytest.mark.parametrize("occs", [(2, 1), (1, 0)])
@pytest.mark.parametrize("tile", [None, 2])
def test_joint_ao_source_pullback_and_empty_spin(
    occs: tuple[int, int], tile: int | None
) -> None:
    rng = np.random.default_rng(58487)
    size, aux = 4, 5
    data = {}
    for spin, occupied in zip(("alpha", "beta"), occs, strict=True):
        h = np.diag([-1.4, -0.8, 0.1, 0.4])
        d = rng.normal(size=(size, size))
        data["hcore_" + spin] = h + 0.06 * (d + d.T)
        raw = rng.normal(size=(aux, size, size)) * 0.04
        raw = raw + raw.swapaxes(1, 2)
        data["raw_coulomb_" + spin] = raw
        if occupied:
            data["fit_" + spin] = raw[:, :occupied]
    sources = {k: rng.normal(size=v.shape) for k, v in data.items()}

    def calc(delta: float, pullback: bool) -> UnrestrictedCabsSingles:
        return unrestricted_cabs_energy(
            (*occs, size, aux),
            {k: v + delta * sources[k] for k, v in data.items()},
            execute,
            singles=_singles,
            pullback=pullback,
            auxiliary_batch_size=tile,
        )

    center = calc(0, True)
    sums = [
        _singles(m, n, spin_weight=1, pullback=False)[0]
        for m, n in zip(_direct_fock(data, occs), occs, strict=True)
    ]
    np.testing.assert_allclose(center.spin_energies, sums, atol=1e-14, rtol=0)
    if not occs[1]:
        assert center.spin_energies[1] == 0.0
    assert calc(0, False).source_cotangents is None
    slope = sum(np.vdot(v, sources[k]) for k, v in center.source_cotangents.items())
    for h in (1e-5, 3e-6):
        finite = (calc(h, False).energy - calc(-h, False).energy) / (2 * h)
        np.testing.assert_allclose(slope, finite, atol=3e-8, rtol=0)


def _direct_fock(
    data: dict[str, np.ndarray], occs: tuple[int, int]
) -> Iterator[np.ndarray]:
    spin_occupations = tuple(zip(("alpha", "beta"), occs, strict=True))
    trace = sum(
        np.trace(data["fit_" + s][:, :, :n], axis1=1, axis2=2)
        for s, n in spin_occupations
        if n
    )
    for s, n in spin_occupations:
        raw = data["raw_coulomb_" + s]
        matrix = data["hcore_" + s] + np.einsum("Qpq,Q->pq", raw, trace)
        if n:
            matrix -= np.einsum("Qip,Qiq->pq", data["fit_" + s], raw[:, :n])
        yield (matrix + matrix.T) / 2


@pytest.mark.parametrize(
    "occs,frozen", [((2, 1), (0, 0)), ((2, 0), (0, 0)), ((1, 2), (1, 2))]
)
def test_ao_energy_and_cotangents_follow_same_spin_reference(
    occs: tuple[int, int], frozen: tuple[int, int]
) -> None:
    from generativeqc._f12_spin_pairs import molecular_pair_energies
    from generativeqc._f12_unrestricted import unrestricted_mp2_f12_ao_response

    rng = np.random.default_rng(6236)
    obs, ext, aux = 3, 5, 2
    space = (*occs, obs, ext, aux)

    def symmetric(shape: tuple[int, ...]) -> np.ndarray:
        x = rng.normal(size=shape)
        return (x + x.swapaxes(-1, -2)) * 0.025

    center = {
        "overlap_ao": np.eye(ext),
        "hcore_ao": np.diag(np.linspace(-2.0, 1.0, ext)) + symmetric((ext, ext)),
        "ao_coulomb": symmetric((aux, ext, ext)),
        "metric/coulomb": np.diag([1.1, 2.0]),
    }
    operators = {
        "geminal": "f12",
        "squared_geminal": "f12_squared",
        "geminal_coulomb": "f12_over_coulomb",
        "double_commutator": "double_commutator",
    }
    for op, raw in operators.items():
        center["metric/" + op] = symmetric((aux, aux))
        center["ao_" + raw] = symmetric((aux, ext, ext))
    for spin in ("alpha", "beta"):
        frame, _ = np.linalg.qr(np.eye(ext) + 0.03 * rng.normal(size=(ext, ext)))
        center["coefficients_" + spin] = frame
    direction = {key: symmetric(x.shape) for key, x in center.items()}
    for spin in ("alpha", "beta"):
        direction["coefficients_" + spin] = rng.normal(size=(ext, ext)) * 0.025

    def forward(step: float) -> tuple[np.ndarray, int]:
        a = {k: x + step * direction[k] for k, x in center.items()}
        c, b = a["coefficients_alpha"], a["coefficients_beta"]
        data = {"cross_overlap": c.T @ a["overlap_ao"] @ b}
        sources = {}
        for spin, occ, frame in zip(("alpha", "beta"), occs, (c, b), strict=True):
            raw = frame.T @ a["ao_coulomb"] @ frame
            sources["hcore_" + spin] = frame.T @ a["hcore_ao"] @ frame
            sources["raw_coulomb_" + spin] = raw
            if occ:
                sources["fit_" + spin] = np.linalg.solve(
                    a["metric/coulomb"], raw[:, :occ].reshape(aux, -1)
                ).reshape(aux, occ, ext)
                for op, name in operators.items():
                    data["metric/" + op] = a["metric/" + op]
                    data[f"raw/{op}/{spin}"] = (
                        frame[:, :occ].T @ a["ao_" + name] @ frame
                    )
        fock = dict(
            zip(("fock_alpha", "fock_beta"), _direct_fock(sources, occs), strict=True)
        )
        for spin, occ in zip(("alpha", "beta"), occs, strict=True):
            fock["exchange_" + spin] = (
                np.einsum(
                    "Aip,Aiq->pq",
                    sources["fit_" + spin],
                    sources["raw_coulomb_" + spin][:, :occ],
                )
                if occ
                else np.zeros((ext, ext))
            )
        pairs = list(
            molecular_pair_energies(
                space, data, sources, fock, execute, frozen_occupied=frozen
            )
        )
        sc = [
            _singles(fock["fock_" + spin], occ, spin_weight=1, pullback=False)[0]
            for spin, occ in zip(("alpha", "beta"), occs, strict=True)
        ]
        return np.array(
            [sum(p.mp2 for p in pairs), sum(p.f12 for p in pairs), *sc]
        ), len(pairs)

    ao = {
        **center,
        "metric_whitener": np.linalg.inv(np.linalg.cholesky(center["metric/coulomb"])),
    }
    energy, bars = unrestricted_mp2_f12_ao_response(
        space,
        ao,
        execute,
        singles=_singles,
        frozen_occupied=frozen,
        auxiliary_batch_size=1,
    )
    expected, count = forward(0.0)
    assert count == energy.active_pairs
    assert bars.keys() == center.keys()
    np.testing.assert_allclose(
        [energy.ordinary_mp2, energy.f12_increment, *energy.cabs_singles],
        expected,
        rtol=0,
        atol=2e-13,
    )
    slope = sum(np.vdot(bars[k], direction[k]) for k in bars)
    for step in (1e-4, 3e-5):
        fd = (forward(step)[0].sum() - forward(-step)[0].sum()) / (2 * step)
        np.testing.assert_allclose(slope, fd, rtol=0, atol=1e-8)


@pytest.mark.parametrize("frozen", [(0, 0), (1, 0), (2, 1)])
def test_joint_spin_pair_and_reference_vjp(frozen: tuple[int, int]) -> None:
    from generativeqc._f12_spin_pairs import molecular_pair_energies
    from generativeqc._f12_unrestricted import unrestricted_mp2_f12_source_pullback
    from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_extended_df_fock

    rng = np.random.default_rng(71137)
    space = (2, 1, 4, 6, 4)
    na, nb, _, ext, aux = space
    values = {"cross_overlap": np.eye(ext)}
    operators = ("geminal", "squared_geminal", "geminal_coulomb", "double_commutator")
    for op in operators:
        matrix = 0.1 * rng.normal(size=(aux, aux))
        values["metric/" + op] = (matrix + matrix.T) / 2
    for spin, occ in (("alpha", na), ("beta", nb)):
        values["source/hcore_" + spin] = np.diag(np.linspace(-1.8, 1, ext))
        raw = 0.07 * rng.normal(size=(aux, ext, ext))
        raw += raw.swapaxes(1, 2).copy()
        values["source/raw_coulomb_" + spin] = raw
        values["source/fit_" + spin] = raw[:, :occ, :].copy()
        for op in operators:
            values["raw/" + op + "/" + spin] = 0.1 * rng.normal(size=(aux, occ, ext))
    # Independent fitted and raw operands perturb the unconstrained Fock
    # off its symmetric manifold; the singles energy uses its symmetric part.
    delta = {k: 0.1 * rng.normal(size=v.shape) for k, v in values.items()}

    def forward(t: float) -> float:
        data = {k: v + t * delta[k] for k, v in values.items()}
        src = {k[7:]: v for k, v in data.items() if k.startswith("source/")}
        fg = unrestricted_extended_df_fock(na, nb, ext, aux)
        fock = execute(fg, src).outputs
        en = sum(
            p.mp2 + p.f12
            for p in molecular_pair_energies(
                space, data, src, fock, execute, frozen_occupied=frozen
            )
        )
        for spin, occ in (("alpha", na), ("beta", nb)):
            ff = fock["fock_" + spin]
            en += _singles((ff + ff.T) / 2, occ, spin_weight=1, pullback=False)[0]
        return en

    bars = unrestricted_mp2_f12_source_pullback(
        space,
        values,
        execute,
        singles=_singles,
        frozen_occupied=frozen,
        auxiliary_batch_size=2,
    )
    actual = sum(np.vdot(bar, delta[k]) for k, bar in bars.items())
    for step in (1e-4, 3e-5):
        expected = (forward(step) - forward(-step)) / (2 * step)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)


@pytest.mark.parametrize("limit", [1, 4, 100])
def test_chunked_force_preserves_strides_and_reference(limit: int) -> None:
    from generativeqc._f12_spin_force import spin_source_force

    seed = np.random.default_rng(5189)
    w = {"B": seed.normal(size=(5, 3)).T, "C": seed.normal(size=4)}
    d = {"C": seed.normal(size=4), "B": seed.normal(size=(3, 5))[:, ::-1]}
    expected = -(1.329 + sum(np.sum(w[k] * d[k]) for k in w))
    sizes = []

    def checked(graph: Program, feeds: dict[str, np.ndarray]) -> Execution:
        sizes.append(feeds["weights"].size)
        return execute(graph, feeds)

    actual = spin_source_force(w, d, 1.329, checked, block_elements=limit)
    assert max(sizes) <= limit and sum(sizes) == 19
    assert abs(actual - expected) < 1e-14


def test_independent_force_stencils_each_keep_both_steps() -> None:
    from benchmarks.f12_uhf_direction import check_displacements

    force = 0.023
    steps = [0.0002, 0.0001]
    good = [
        {"step_bohr": h, "plus_Eh": -55.0 - h * force, "minus_Eh": -55.0 + h * force}
        for h in steps
    ]
    ref = {"displacements": good, "independent_displacements": {"other": good}}
    original, independent = check_displacements(force, ref)
    assert max(original) < 1e-10
    assert independent == {"other": original}
    bad = [dict(good[0]), dict(good[1], plus_Eh=good[1]["plus_Eh"] + 1e-9)]
    ref["independent_displacements"] = {"other": bad}
    with pytest.raises(ValueError, match="failed its independent displacement checks"):
        check_displacements(force, ref)
