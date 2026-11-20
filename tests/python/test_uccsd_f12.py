"""Independent fixed-geminal UCCSD-F12b equation and response gates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product

import numpy as np
import pytest
from generativeqc_compiler.cc.f12_geminal_energy import all_pair_geminal_energy
from generativeqc_compiler.cc.f12_residual import all_pair_cc_residual
from generativeqc_compiler.cc.uccsd_equations import (
    build_f12_correction_program,
    build_f12_program,
    build_program,
)
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.cc.uccsd_response import (
    build_f12_response_programs,
    build_parameter_vjp,
)
from generativeqc_compiler.tensor import execute
from test_uccsd_equations import _feeds, _random_hamiltonian, _spin_amplitudes


@dataclass(frozen=True)
class _PairSources:
    w: np.ndarray
    c: np.ndarray
    scalar: float


def _pair_sources(layout: UCCSDLayout, seed: int) -> _PairSources:
    rng = np.random.default_rng(seed)
    occupied = (0,) * layout.oa + (1,) * layout.ob
    virtual = (0,) * layout.va + (1,) * layout.vb

    def antisymmetric_pairs(free: tuple[int, ...]) -> np.ndarray:
        result = np.zeros((len(occupied), len(occupied), len(free), len(free)))
        for i, j in combinations(range(len(occupied)), 2):
            for p, q in combinations(range(len(free)), 2):
                if occupied[i] + occupied[j] == free[p] + free[q]:
                    value = rng.normal(scale=0.045)
                    result[i, j, p, q] = result[j, i, q, p] = value
                    result[j, i, p, q] = result[i, j, q, p] = -value
        return result

    return _PairSources(
        antisymmetric_pairs(occupied + virtual),
        antisymmetric_pairs(virtual),
        float(rng.normal(scale=0.045)),
    )


def _source_feeds(layout: UCCSDLayout, sources: _PairSources) -> dict[str, np.ndarray]:
    no = layout.oa + layout.ob
    oa, ob = slice(0, layout.oa), slice(layout.oa, no)
    va, vb = slice(0, layout.va), slice(layout.va, layout.va + layout.vb)
    obs_va = slice(no, no + layout.va)
    obs_vb = slice(no + layout.va, no + layout.va + layout.vb)
    occupied, virtual, obs_virtual = (oa, ob), (va, vb), (obs_va, obs_vb)
    feeds = {"fixed_geminal_energy": np.array(sources.scalar)}
    for i, j, p, q in product((0, 1), repeat=4):
        if i + j != p + q:
            continue
        spins = "".join("ab"[spin] for spin in (i, j, p, q))
        first, second = occupied[i], occupied[j]
        feeds[f"w_vv_{spins}"] = sources.w[
            first, second, obs_virtual[p], obs_virtual[q]
        ].copy()
        feeds[f"w_vo_{spins}"] = sources.w[
            first, second, obs_virtual[p], occupied[q]
        ].copy()
        feeds[f"c_f12_{spins}"] = sources.c[
            first, second, virtual[p], virtual[q]
        ].copy()
    return feeds


def _dense_correction(
    layout: UCCSDLayout, amplitudes: dict[str, np.ndarray], sources: _PairSources
) -> dict[str, np.ndarray]:
    """Sum all ordered occupied and virtual spin orbitals directly."""
    t1, t2 = _spin_amplitudes(layout, amplitudes)
    no, _ = t1.shape
    r1, r2 = np.zeros_like(t1), np.zeros_like(t2)
    energy = sources.scalar
    for i, a in np.ndindex(t1.shape):
        r1[i, a] = sum(sources.w[i, j, no + a, j] for j in range(no))
        energy += r1[i, a] * t1[i, a]
    for i, j, a, b in np.ndindex(t2.shape):
        w = sources.w[i, j, no + a, no + b]
        c = sources.c[i, j, a, b]
        ta = t2[i, j, a, b] + t1[i, a] * t1[j, b] - t1[i, b] * t1[j, a]
        energy += (w * ta + c * t2[i, j, a, b]) / 4
        r2[i, j, a, b] = (
            w
            + c
            - sum(
                sources.w[i, j, no + a, k] * t1[k, b]
                - sources.w[i, j, no + b, k] * t1[k, a]
                for k in range(no)
            )
        )
    oa, ob = slice(0, layout.oa), slice(layout.oa, None)
    va, vb = slice(0, layout.va), slice(layout.va, None)
    return {
        "r1a": r1[oa, va],
        "r1b": r1[ob, vb],
        "r2aa": r2[oa, oa, va, va],
        "r2ab": r2[oa, ob, va, vb],
        "r2bb": r2[ob, ob, vb, vb],
        "correlation_energy": np.array(energy),
    }


def _pack_residual(
    layout: UCCSDLayout, outputs: dict[str, np.ndarray], prefix: str = ""
) -> np.ndarray:
    return np.concatenate(
        [
            block.pack_dual(outputs[f"{prefix}r{block.name[1:]}"])
            / block.full_metric_weight
            for block in layout.blocks
        ]
    )


@pytest.mark.parametrize(
    "populations", [(3, 2, 2, 3), (2, 0, 2, 0), (0, 2, 2, 3), (0, 0, 0, 0)]
)
def test_fixed_cusp_correction_and_ordinary_limit(populations: tuple[int, ...]) -> None:
    layout = UCCSDLayout(*populations)
    amplitudes = layout.unpack(
        np.random.default_rng(973).normal(scale=0.09, size=layout.packed_size)
    )
    sources = _pair_sources(layout, 974)
    source_feeds = _source_feeds(layout, sources)
    fock, integrals = _random_hamiltonian(layout)
    ordinary_feeds = _feeds(layout, fock, integrals, amplitudes)
    ordinary = execute(build_program(*populations), ordinary_feeds).outputs
    correction = execute(
        build_f12_correction_program(layout), {**amplitudes, **source_feeds}
    ).outputs
    combined_program = build_f12_program(layout)
    combined = execute(combined_program, {**ordinary_feeds, **source_feeds}).outputs
    zero = execute(
        combined_program,
        {
            **ordinary_feeds,
            **{name: np.zeros_like(value) for name, value in source_feeds.items()},
        },
    ).outputs
    expected = _dense_correction(layout, amplitudes, sources)
    assert correction.keys() == combined.keys() == zero.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_allclose(
            correction[name], value, atol=3e-14, rtol=0, err_msg=name
        )
        np.testing.assert_allclose(
            combined[name], ordinary[name] + value, atol=3e-14, rtol=0, err_msg=name
        )
        np.testing.assert_array_equal(zero[name], ordinary[name], err_msg=name)


def test_fixed_cusp_reduction_to_restricted_f12b() -> None:
    no, nv = 2, 3
    layout = UCCSDLayout(no, no, nv, nv)
    rng = np.random.default_rng(975)

    def spatial_pairs(free: int, *, hermitian: bool = False) -> np.ndarray:
        pairs = rng.normal(scale=0.2, size=(no, no, free, free))
        pairs = (pairs + pairs.transpose(1, 0, 3, 2)) / 2
        if hermitian:
            pairs = (pairs + pairs.transpose(2, 3, 0, 1)) / 2
        return pairs

    occ_index = np.tile(np.arange(no), 2)
    occ_spin = np.repeat((0, 1), no)
    vir_index = np.tile(np.arange(nv), 2)
    vir_spin = np.repeat((0, 1), nv)

    def spin_pairs(
        spatial: np.ndarray,
        free_index: np.ndarray,
        free_spin: np.ndarray,
        *,
        both_cusp_legs: bool = False,
    ) -> np.ndarray:
        direct = (3 * spatial + spatial.swapaxes(0, 1)) / 8
        if both_cusp_legs:
            direct = (3 * direct + direct.swapaxes(2, 3)) / 8
        i, j, p, q = np.ix_(occ_index, occ_index, free_index, free_index)
        si, sj, sp, sq = np.ix_(occ_spin, occ_spin, free_spin, free_spin)
        return direct[i, j, p, q] * ((si == sp) & (sj == sq)) - direct[i, j, q, p] * (
            (si == sq) & (sj == sp)
        )

    w, c = spatial_pairs(no + nv), spatial_pairs(nv)
    v_occ = spatial_pairs(no)
    b, x = spatial_pairs(no, hermitian=True), spatial_pairs(no, hermitian=True)
    foo = rng.normal(scale=0.2, size=(no, no))
    foo = (foo + foo.T) / 2
    v_spin = spin_pairs(v_occ, occ_index, occ_spin)
    b_spin = spin_pairs(b, occ_index, occ_spin, both_cusp_legs=True)
    x_spin = spin_pairs(x, occ_index, occ_spin, both_cusp_legs=True)
    x_foo = np.einsum("ijkj->ik", x_spin)
    fixed = (
        np.einsum("ijij->", v_spin)
        + np.einsum("ijij->", b_spin) / 2
        - np.einsum("ik,ik->", np.kron(np.eye(2), foo), x_foo)
    )
    sources = _PairSources(
        spin_pairs(
            w,
            np.concatenate((occ_index, no + vir_index)),
            np.concatenate((occ_spin, vir_spin)),
        ),
        spin_pairs(c, vir_index, vir_spin),
        float(fixed),
    )
    t1, t2 = rng.normal(scale=0.2, size=(no, nv)), spatial_pairs(nv)
    same_spin_t2 = t2 - t2.swapaxes(2, 3)
    amplitudes = {
        "t1a": t1,
        "t1b": t1,
        "t2aa": same_spin_t2,
        "t2ab": t2,
        "t2bb": same_spin_t2,
    }
    correction = build_f12_correction_program(layout)
    spin = execute(correction, {**amplitudes, **_source_feeds(layout, sources)}).outputs
    common_sources = replace(sources, w=np.zeros_like(sources.w))
    spin_common = execute(
        correction, {**amplitudes, **_source_feeds(layout, common_sources)}
    ).outputs

    restricted_feeds = {
        "v_vv": w[:, :, no:, no:],
        "v_vo": w[:, :, no:, :no],
        "v_ov": w[:, :, :no, no:],
        "v": v_occ,
        "b": b,
        "x": x,
        "foo": foo,
        "c": c,
        "t1": t1,
        "t2": t2,
    }
    restricted_residual = execute(
        all_pair_cc_residual(no, nv), restricted_feeds
    ).outputs
    restricted_energy = execute(
        all_pair_geminal_energy(no, nv), restricted_feeds
    ).outputs
    doubles = restricted_residual["doubles"]
    same_spin_doubles = doubles - doubles.swapaxes(2, 3)
    for name in ("r1a", "r1b"):
        np.testing.assert_allclose(
            spin[name], restricted_residual["singles"], atol=3e-14, rtol=0
        )
    np.testing.assert_allclose(spin["r2ab"], doubles, atol=3e-14, rtol=0)
    for name in ("r2aa", "r2bb"):
        np.testing.assert_allclose(spin[name], same_spin_doubles, atol=3e-14, rtol=0)
    common = spin_common["correlation_energy"]
    np.testing.assert_allclose(
        common, restricted_energy["f12a_geminal_energy"], atol=3e-14, rtol=0
    )
    np.testing.assert_allclose(
        spin["correlation_energy"] - common,
        restricted_energy["f12b_extra_energy"],
        atol=3e-14,
        rtol=0,
    )


def test_packed_f12_response_and_stationary_source_gradients() -> None:
    layout = UCCSDLayout(2, 2, 2, 3)
    programs = build_f12_response_programs(layout)
    no, nv = layout.oa + layout.ob, layout.va + layout.vb
    singles_size = layout.oa * layout.va + layout.ob * layout.vb
    # With diagonal Fock and zero two-electron integrals, ordinary CC only
    # contributes its orbital energy gaps to the residual. The F12 singles
    # are constant and its doubles depend on T1, so the root is triangular.
    fock = np.diag(np.repeat((-0.8, 0.7), (no, nv)))
    integrals = np.zeros((no + nv,) * 4)
    gaps = np.repeat((1.5, 3.0), (singles_size, layout.packed_size - singles_size))

    def reference(packed: np.ndarray, source: _PairSources) -> tuple[float, np.ndarray]:
        values = _dense_correction(layout, layout.unpack(packed), source)
        return float(values["correlation_energy"]), gaps * packed + _pack_residual(
            layout, values
        )

    def stationary_amplitudes(source: _PairSources) -> np.ndarray:
        result = np.zeros(layout.packed_size)
        result[:singles_size] = (
            -reference(result, source)[1][:singles_size] / gaps[:singles_size]
        )
        result[singles_size:] = (
            -reference(result, source)[1][singles_size:] / gaps[singles_size:]
        )
        return result

    source = _pair_sources(layout, 1029)
    packed = stationary_amplitudes(source)
    feeds = {
        **_feeds(layout, fock, integrals, layout.unpack(packed)),
        **_source_feeds(layout, source),
    }
    primal = execute(programs.primal, feeds).outputs
    np.testing.assert_allclose(_pack_residual(layout, primal), 0, atol=5e-16, rtol=0)
    energy, _ = reference(packed, source)
    assert float(primal["correlation_energy"]) == pytest.approx(energy, abs=3e-14)

    step = 1e-5
    jacobian = np.empty((packed.size, packed.size))
    gradient = np.empty_like(packed)
    for column, direction in enumerate(np.eye(packed.size)):
        e_plus, r_plus = reference(packed + step * direction, source)
        e_minus, r_minus = reference(packed - step * direction, source)
        jacobian[:, column] = (r_plus - r_minus) / (2 * step)
        gradient[column] = (e_plus - e_minus) / (2 * step)

    rng = np.random.default_rng(1030)
    direction = rng.normal(size=packed.size)
    dual = rng.normal(size=packed.size)
    stationary_dual = np.linalg.solve(jacobian.T, -gradient)

    def residual_seeds(seed: np.ndarray) -> dict[str, np.ndarray]:
        return {
            f"bar_r{name[1:]}": value
            for name, value in layout.unpack_dual(seed).items()
        }

    def pack_amplitude_bars(outputs: dict[str, np.ndarray]) -> np.ndarray:
        return layout.pack_dual(
            {block.name: outputs[f"bar_{block.name}"] for block in layout.blocks}
        )

    energy_seed = {"bar_correlation_energy": np.array(1.0)}
    stationary_feeds = {**feeds, **energy_seed, **residual_seeds(stationary_dual)}
    jvp = execute(
        programs.residual_jvp.program,
        {
            **feeds,
            **{f"d_{name}": value for name, value in layout.unpack(direction).items()},
        },
    ).outputs
    vjp = execute(
        programs.residual_vjp.program, {**feeds, **residual_seeds(dual)}
    ).outputs
    energy_vjp = execute(programs.energy_vjp.program, {**feeds, **energy_seed}).outputs
    stationarity = execute(programs.stationarity_vjp.program, stationary_feeds).outputs
    np.testing.assert_allclose(
        _pack_residual(layout, jvp, "d_"), jacobian @ direction, atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        pack_amplitude_bars(vjp), jacobian.T @ dual, atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        pack_amplitude_bars(energy_vjp), gradient, atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(pack_amplitude_bars(stationarity), 0, atol=1e-10, rtol=0)

    perturbation = _pair_sources(layout, 1031)
    zero = _PairSources(np.zeros_like(source.w), np.zeros_like(source.c), 0.0)
    w_vv = np.zeros_like(source.w)
    w_vv[:, :, no:, no:] = perturbation.w[:, :, no:, no:]
    for prefix, source_direction in (
        ("w_vv_", replace(zero, w=w_vv)),
        ("w_vo_", replace(zero, w=perturbation.w - w_vv)),
        ("c_f12_", replace(zero, c=perturbation.c)),
        ("fixed_geminal_energy", replace(zero, scalar=perturbation.scalar)),
    ):
        parameters = tuple(
            name for name in programs.parameter_names if name.startswith(prefix)
        )
        assert parameters
        direction_fields = _source_feeds(layout, source_direction)
        predicted = 0.0
        for parameter in parameters:
            action = build_parameter_vjp(programs, parameter)
            bar = execute(action.program, stationary_feeds).outputs[f"bar_{parameter}"]
            predicted += np.vdot(bar, direction_fields[parameter])
        energies = []
        for sign in (1, -1):
            shifted = _PairSources(
                source.w + sign * step * source_direction.w,
                source.c + sign * step * source_direction.c,
                source.scalar + sign * step * source_direction.scalar,
            )
            energies.append(reference(stationary_amplitudes(shifted), shifted)[0])
        np.testing.assert_allclose(
            predicted,
            (energies[0] - energies[1]) / (2 * step),
            atol=1e-10,
            rtol=0,
            err_msg=prefix,
        )
