"""Independent dense FP64 checks for the auxiliary-sliced UCC virtual term."""

import numpy as np
import pytest
from generativeqc_compiler.cc.uccsd_df_virtual_ladder import (
    FACTOR_INPUTS,
    TAU_INPUTS,
    build_uccsd_df_virtual_ladder,
    build_uccsd_df_virtual_ladder_response,
)
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES, UCCSDLayout
from generativeqc_compiler.cc.uccsd_response import (
    RESIDUALS,
    build_f12_df_remainder_source_vjps,
    build_f12_df_response_programs,
    build_f12_response_programs,
)
from generativeqc_compiler.tensor import execute, transpose_program


def _fixture(layout: UCCSDLayout, auxiliary: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds = {
        name: rng.normal(scale=0.2, size=block.spec.shape)
        for name, block in zip(TAU_INPUTS, layout.blocks[2:], strict=True)
    }
    for name, v in zip(
        FACTOR_INPUTS, (layout.va, layout.vb, layout.va, layout.vb), strict=True
    ):
        feeds[name] = rng.normal(scale=0.2, size=(auxiliary, v, v))
    return feeds


def _dense_virtual_integrals(
    va: int, vb: int, feeds: dict[str, np.ndarray]
) -> np.ndarray:
    pa, pb = slice(va), slice(va, va + vb)
    coulomb = np.zeros((va + vb,) * 4)
    for left, right, name_left, name_right in (
        (pa, pa, "alpha", "alpha"),
        (pa, pb, "alpha", "beta"),
        (pb, pb, "beta", "beta"),
    ):
        coulomb[left, left, right, right] = np.einsum(
            "Pae,Pbf->aebf",
            feeds["fit_left_" + name_left],
            feeds["raw_right_" + name_right],
        )
    coulomb[pb, pb, pa, pa] = coulomb[pa, pa, pb, pb].transpose(2, 3, 0, 1)
    direct = coulomb.transpose(0, 2, 1, 3)
    return direct - direct.swapaxes(2, 3)


def _dense_spin_oracle(
    layout: UCCSDLayout, feeds: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Expand occupied and virtual spins before building chemists' integrals."""
    oa, ob, va, vb = layout.oa, layout.ob, layout.va, layout.vb
    ia, ib = slice(oa), slice(oa, oa + ob)
    pa, pb = slice(va), slice(va, va + vb)
    tau = np.zeros((oa + ob, oa + ob, va + vb, va + vb))
    tau[ia, ia, pa, pa] = feeds["tau_aa"]
    tau[ib, ib, pb, pb] = feeds["tau_bb"]
    ab = feeds["tau_ab"]
    tau[ia, ib, pa, pb] = ab
    tau[ia, ib, pb, pa] = -ab.swapaxes(2, 3)
    tau[ib, ia, pa, pb] = -ab.swapaxes(0, 1)
    tau[ib, ia, pb, pa] = ab.transpose(1, 0, 3, 2)
    residual = 0.5 * np.einsum(
        "ijef,abef->ijab", tau, _dense_virtual_integrals(va, vb, feeds)
    )
    return {
        "r2aa": residual[ia, ia, pa, pa],
        "r2ab": residual[ia, ib, pa, pb],
        "r2bb": residual[ib, ib, pb, pb],
    }


@pytest.mark.parametrize(
    "populations", [(2, 2, 4, 4), (2, 3, 4, 5), (2, 0, 4, 3), (2, 1, 4, 0)]
)
def test_auxiliary_slices_match_dense_spin_oracle(
    populations: tuple[int, int, int, int],
) -> None:
    layout = UCCSDLayout(*populations)
    feeds = _fixture(layout, 7, 55)
    expected = _dense_spin_oracle(layout, feeds)
    accumulated = {name: np.zeros_like(value) for name, value in expected.items()}
    for start in range(0, 7, 2):
        stop = min(start + 2, 7)
        slab = {**feeds, **{key: feeds[key][start:stop] for key in FACTOR_INPUTS}}
        values = execute(build_uccsd_df_virtual_ladder(layout, stop - start), slab)
        for name, array in values.outputs.items():
            accumulated[name] += array
    for name, value in expected.items():
        np.testing.assert_allclose(accumulated[name], value, rtol=0, atol=2e-14)


@pytest.mark.parametrize(
    "side,input_names", [("tau", TAU_INPUTS), ("factor", FACTOR_INPUTS)]
)
def test_ladder_directions_and_adjoints(
    side: str, input_names: tuple[str, ...]
) -> None:
    layout = UCCSDLayout(2, 3, 4, 5)
    programs = build_uccsd_df_virtual_ladder_response(layout, 3)
    feeds = _fixture(layout, 3, 93)
    direction = _fixture(layout, 3, 94)
    rng = np.random.default_rng(95)
    seeds = {
        name: rng.normal(size=node.spec.shape)
        for name, node in programs.primal.outputs.items()
    }
    tangent = execute(
        getattr(programs, side + "_jvp").program,
        {**feeds, **{"d_" + name: direction[name] for name in input_names}},
    ).outputs
    bars = execute(
        getattr(programs, side + "_vjp").program,
        {**feeds, **{"bar_" + name: seed for name, seed in seeds.items()}},
    ).outputs
    step = 0.125
    dense_directions = [
        _dense_spin_oracle(
            layout,
            {
                **feeds,
                **{
                    name: feeds[name] + sign * step * direction[name]
                    for name in input_names
                },
            },
        )
        for sign in (1, -1)
    ]
    for name in seeds:
        # The isolated term has at most second degree in the perturbed inputs.
        expected = (dense_directions[0][name] - dense_directions[1][name]) / (2 * step)
        np.testing.assert_allclose(tangent["d_" + name], expected, rtol=0, atol=2e-14)
    left = sum(np.vdot(seeds[name], tangent["d_" + name]) for name in seeds)
    right = sum(np.vdot(direction[name], bars["bar_" + name]) for name in input_names)
    np.testing.assert_allclose(left, right, rtol=0, atol=2e-14)


def test_organic_ha_t_graph_bounds_every_primal_and_response_intermediate() -> None:
    layout = UCCSDLayout(17, 17, 185, 185)
    programs = build_uccsd_df_virtual_ladder_response(layout, 8)
    for graph in (
        programs.primal,
        programs.tau_jvp.program,
        programs.tau_vjp.program,
        programs.factor_jvp.program,
        programs.factor_vjp.program,
    ):
        assert all(
            sum(idx.space.kind == "virtual" for idx in node.spec.indices) <= 2
            for node in graph.live_nodes
        )
        assert max(node.spec.size for node in graph.live_nodes) <= 8 * 17**2 * 185**2


def _dense_virtual_blocks(
    layout: UCCSDLayout, values: np.ndarray, names: tuple[str, ...]
) -> dict[str, np.ndarray]:
    bounds = {"a": slice(layout.va), "b": slice(layout.va, layout.va + layout.vb)}
    return {
        name: values[tuple(bounds[label] for label in name.split("_")[1])]
        for name in names
    }


@pytest.mark.parametrize("populations", [(2, 3, 4, 5), (2, 0, 4, 3)])
def test_complete_fixed_f12_equations_and_responses_match_dense(
    populations: tuple[int, int, int, int],
) -> None:
    layout = UCCSDLayout(*populations)
    dense = build_f12_response_programs(layout)
    rng = np.random.default_rng(293)
    inputs = {
        node.attrs["name"]: rng.normal(scale=0.2, size=node.spec.shape)
        for node in dense.primal.live_nodes
        if node.op == "input" and node.attrs["name"] not in BLOCK_NAMES
    }
    inputs.update(layout.unpack(rng.normal(scale=0.2, size=layout.packed_size)))
    names = tuple(name for name in inputs if name.startswith("gvvvv_"))
    factors = {
        name: value
        for name, value in _fixture(layout, 5, 294).items()
        if name in FACTOR_INPUTS
    }
    inputs.update(
        _dense_virtual_blocks(
            layout, _dense_virtual_integrals(layout.va, layout.vb, factors), names
        )
    )
    seeds = {
        "bar_" + name.replace("t", "r", 1): value
        for name, value in layout.unpack_dual(
            rng.normal(size=layout.packed_size)
        ).items()
    }
    seeds["bar_correlation_energy"] = np.array(0.61)
    directions = {
        "d_" + name: value
        for name, value in layout.unpack(rng.normal(size=layout.packed_size)).items()
    }
    programs = {
        width: build_f12_df_response_programs(layout, width) for width in (2, 1)
    }
    remainder = programs[2].remainder

    matched_actions = (
        (dense.primal, remainder.primal, "primal", {}),
        (
            dense.residual_jvp.program,
            remainder.residual_jvp.program,
            "residual_jvp",
            directions,
        ),
        (
            dense.residual_vjp.program,
            remainder.residual_vjp.program,
            "residual_vjp",
            seeds,
        ),
        (
            dense.stationarity_vjp.program,
            remainder.stationarity_vjp.program,
            "residual_vjp",
            seeds,
        ),
    )
    for full_graph, remainder_graph, slice_action, extra in matched_actions:
        actual = {
            name: value.copy()
            for name, value in execute(
                remainder_graph, {**inputs, **extra}
            ).outputs.items()
        }
        for start, stop in ((0, 2), (2, 4), (4, 5)):
            ladder = programs[stop - start].ladder
            action = getattr(ladder, slice_action)
            graph = action if slice_action == "primal" else action.program
            slab = {name: factor[start:stop] for name, factor in factors.items()}
            for name, value in execute(
                graph, {**inputs, **slab, **extra}
            ).outputs.items():
                actual[name] += value
        expected = execute(full_graph, {**inputs, **extra}).outputs
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name], value, rtol=0, atol=2e-13)

    all_sources = ("correlation_energy", *RESIDUALS)
    full_pullback = transpose_program(
        dense.primal, all_sources, inputs=dense.parameter_names
    ).program
    expected_bars = execute(full_pullback, {**inputs, **seeds}).outputs
    actual_bars = {
        name: value
        for group in build_f12_df_remainder_source_vjps(programs[2])
        for name, value in execute(group.program, {**inputs, **seeds}).outputs.items()
    }
    for name in remainder.parameter_names:
        np.testing.assert_allclose(
            actual_bars["bar_" + name], expected_bars["bar_" + name], rtol=0, atol=2e-13
        )

    df_directions = {
        name: rng.normal(size=factor.shape) for name, factor in factors.items()
    }
    epsilon = 0.125  # The dense integral is bilinear in the factors.
    dense_directions = [
        _dense_virtual_integrals(
            layout.va,
            layout.vb,
            {
                name: factor + sign * epsilon * df_directions[name]
                for name, factor in factors.items()
            },
        )
        for sign in (1, -1)
    ]
    g_tangent = (dense_directions[0] - dense_directions[1]) / (2 * epsilon)
    expected = sum(
        np.vdot(expected_bars["bar_" + name], value)
        for name, value in _dense_virtual_blocks(layout, g_tangent, names).items()
    )
    actual = 0.0
    for start, stop in ((0, 2), (2, 4), (4, 5)):
        graph = programs[stop - start].ladder.factor_vjp.program
        slab = {name: factor[start:stop] for name, factor in factors.items()}
        bars = execute(graph, {**inputs, **slab, **seeds}).outputs
        actual += sum(
            np.vdot(bars["bar_" + name], direction[start:stop])
            for name, direction in df_directions.items()
        )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-13)
