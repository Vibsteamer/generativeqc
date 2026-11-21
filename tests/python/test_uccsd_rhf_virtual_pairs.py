"""Independent repeated-action tests for exact RHF virtual-pair Coulomb factors."""

import numpy as np
import pytest
from generativeqc._uccsd_rhf_virtual_pairs import (
    prepare_rhf_f12_lambda_action,
    prepare_rhf_virtual_pair_operators,
)
from generativeqc_compiler.cc.uccsd_df_virtual_ladder import (
    build_uccsd_df_virtual_ladder_amplitude_programs,
)
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES, UCCSDLayout
from generativeqc_compiler.cc.uccsd_response import build_f12_response_programs
from generativeqc_compiler.cc.uccsd_rhf_virtual_pairs import (
    build_uccsd_rhf_virtual_pair_programs,
)
from generativeqc_compiler.tensor import execute
from test_uccsd_df_virtual_ladder import _dense_virtual_blocks, _dense_virtual_integrals


def _factor_fixture(auxiliary: int, virtual: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(81812)
    raw = rng.normal(scale=0.2, size=(auxiliary, virtual, virtual))
    raw = raw + raw.swapaxes(1, 2)
    factor = rng.normal(scale=0.3, size=(auxiliary, auxiliary))
    metric = factor @ factor.T + np.eye(auxiliary)
    fit = np.linalg.solve(metric, raw.reshape(auxiliary, -1)).reshape(raw.shape)
    return fit, raw


@pytest.mark.parametrize("occupied,virtual", [(1, 3), (2, 4), (3, 2)])
def test_reusable_virtual_operator_matches_physical_df_actions(
    occupied: int,
    virtual: int,
) -> None:
    layout = UCCSDLayout(occupied, occupied, virtual, virtual)
    fit, raw = _factor_fixture(5, virtual)
    prepared = prepare_rhf_virtual_pair_operators(
        fit, raw, execute, row_tile=3, max_bytes=1 << 20
    )
    reference = build_uccsd_df_virtual_ladder_amplitude_programs(layout, 5)
    paired = build_uccsd_rhf_virtual_pair_programs(layout)
    assert all(not operator.flags.writeable for operator in prepared.values())
    rng = np.random.default_rng(71234)
    factors = {
        key + "_" + spin: value
        for spin in ("alpha", "beta")
        for key, value in (("fit_left", fit), ("raw_right", raw))
    }
    for _ in range(2):
        amplitude = layout.unpack(rng.normal(scale=0.2, size=layout.packed_size))
        tangent = {
            "d_" + name: value
            for name, value in layout.unpack(
                rng.normal(size=layout.packed_size)
            ).items()
        }
        seeds = {
            "bar_" + name.replace("t", "r", 1): value
            for name, value in layout.unpack_dual(
                rng.normal(size=layout.packed_size)
            ).items()
        }
        for graph, oracle, extra in (
            (paired.primal, reference.primal, {}),
            (paired.residual_jvp.program, reference.residual_jvp.program, tangent),
        ):
            actual = execute(graph, {**amplitude, **prepared, **extra}).outputs
            expected = execute(oracle, {**amplitude, **factors, **extra}).outputs
            for name in actual:
                np.testing.assert_allclose(
                    actual[name], expected[name], rtol=0, atol=3e-14
                )
        actual = execute(
            paired.residual_vjp.program, {**amplitude, **prepared, **seeds}
        ).outputs
        expected = execute(
            reference.residual_vjp.program, {**amplitude, **factors, **seeds}
        ).outputs
        actual_dual = layout.pack_dual(
            {name: actual["bar_" + name] for name in BLOCK_NAMES}
        )
        expected_dual = layout.pack_dual(
            {name: expected["bar_" + name] for name in BLOCK_NAMES}
        )
        np.testing.assert_allclose(actual_dual, expected_dual, rtol=0, atol=5e-14)


def test_resource_or_nonphysical_factor_rejection() -> None:
    fit, raw = _factor_fixture(2, 3)

    def never_run(*args: object) -> None:
        raise AssertionError("RHF spatial operator was staged before admission")

    with pytest.raises(MemoryError, match="host numeric budget"):
        prepare_rhf_virtual_pair_operators(fit, raw, never_run, row_tile=2, max_bytes=8)
    with pytest.raises(ValueError, match="equal spin populations"):
        build_uccsd_rhf_virtual_pair_programs(UCCSDLayout(2, 3, 4, 4))
    fit[0, 0, 1] += 0.2
    with pytest.raises(ValueError, match="within-pair symmetry"):
        prepare_rhf_virtual_pair_operators(
            fit, raw, execute, row_tile=2, max_bytes=1 << 20
        )


def test_stationary_epoch_replays_complete_physical_lambda_action() -> None:
    layout = UCCSDLayout(2, 2, 4, 4)
    fit, raw = _factor_fixture(5, 4)
    operators = prepare_rhf_virtual_pair_operators(
        fit, raw, execute, row_tile=3, max_bytes=1 << 20
    )
    dense = build_f12_response_programs(layout)
    rng = np.random.default_rng(758)
    feeds = {
        node.attrs["name"]: rng.normal(scale=0.2, size=node.spec.shape)
        for node in dense.primal.live_nodes
        if node.op == "input" and node.attrs["name"] not in BLOCK_NAMES
    }
    names = tuple(name for name in feeds if name.startswith("gvvvv_"))
    factors = {
        key + "_" + spin: value
        for spin in ("alpha", "beta")
        for key, value in (("fit_left", fit), ("raw_right", raw))
    }
    feeds.update(
        _dense_virtual_blocks(layout, _dense_virtual_integrals(4, 4, factors), names)
    )
    for value in feeds.values():
        value.setflags(write=False)
    packed = rng.normal(scale=0.1, size=layout.packed_size)
    amplitudes = layout.unpack(packed)
    with pytest.raises(MemoryError, match="frontier retained storage"):
        prepare_rhf_f12_lambda_action(
            layout, packed, feeds, operators, execute, max_retained_bytes=1
        )
    action = prepare_rhf_f12_lambda_action(
        layout, packed, feeds, operators, execute, max_retained_bytes=1 << 20
    )
    # The caller's amplitude work vector may be reused while this fixed epoch
    # owns its own snapshot.
    packed.fill(0.0)
    for _ in range(3):
        dual = rng.normal(size=layout.packed_size)
        seeds = {
            "bar_" + name.replace("t", "r", 1): value
            for name, value in layout.unpack_dual(dual).items()
        }
        bars = execute(
            dense.residual_vjp.program, {**feeds, **amplitudes, **seeds}
        ).outputs
        expected = layout.pack_dual({name: bars["bar_" + name] for name in BLOCK_NAMES})
        np.testing.assert_allclose(action(dual), expected, rtol=0, atol=2e-13)
