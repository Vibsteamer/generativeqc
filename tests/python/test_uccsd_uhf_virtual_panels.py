"""Check bounded unrestricted operators against a whole independent-spin DF graph."""

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc._uccsd_rhf_virtual_pairs import (
    prepare_spatial_virtual_pair_operators,
    spatial_virtual_pair_storage,
)
from generativeqc._uccsd_uhf_virtual_ladder import (
    UHFVirtualGraphResources,
    UHFVirtualLadderLimits,
    prepare_uhf_virtual_ladder,
)
from generativeqc._uccsd_uhf_virtual_panels import (
    prepare_uhf_mixed_virtual_panels,
    uhf_mixed_virtual_panel_storage,
)
from generativeqc_compiler.cc.uccsd_df_virtual_ladder import (
    build_uccsd_df_virtual_ladder_amplitude_programs,
)
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.cc.uccsd_uhf_virtual_panels import (
    MIXED_AMPLITUDES,
    build_uccsd_uhf_mixed_virtual_panel_programs,
    build_uccsd_uhf_same_spin_virtual_pair_programs,
)
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.cpu import emit_cpu

if TYPE_CHECKING:
    from generativeqc_compiler.tensor import Program


@pytest.mark.parametrize("width", [1, 2, 3])
def test_independent_spin_ladder_primal_tangent_and_transpose(width: int) -> None:
    layout = UCCSDLayout(2, 1, 3, 4)
    rng = np.random.default_rng(733021)
    auxiliary = 5
    factors = {
        f"{side}_{spin}": rng.normal(scale=0.3, size=(auxiliary, n, n))
        for side in ("fit_left", "raw_right")
        for spin, n in (("alpha", layout.va), ("beta", layout.vb))
    }
    bound, retained = uhf_mixed_virtual_panel_storage(
        layout.va, layout.vb, auxiliary, width
    )
    fit, raw = factors["fit_left_alpha"], factors["raw_right_beta"]
    panels = prepare_uhf_mixed_virtual_panels(
        fit, raw, execute, width=width, max_host_bytes=bound
    )
    assert sum(panel.operator.nbytes for panel in panels) == retained
    assert all(not panel.operator.flags.writeable for panel in panels)
    reference = build_uccsd_df_virtual_ladder_amplitude_programs(layout, auxiliary)
    amplitudes = layout.unpack(rng.normal(scale=0.5, size=layout.packed_size))
    directions = {
        "d_" + name: value
        for name, value in layout.unpack(rng.normal(size=layout.packed_size)).items()
    }
    mixed_seed = rng.normal(size=amplitudes["t2ab"].shape)
    seeds = {
        "bar_r2aa": np.zeros_like(amplitudes["t2aa"]),
        "bar_r2ab": mixed_seed,
        "bar_r2bb": np.zeros_like(amplitudes["t2bb"]),
    }
    bars = {"bar_" + name: np.zeros_like(amplitudes[name]) for name in MIXED_AMPLITUDES}
    primal, tangent = [], []
    for panel in panels:
        programs = build_uccsd_uhf_mixed_virtual_panel_programs(
            layout, panel.alpha_stop - panel.alpha_start
        )
        fields = {**amplitudes, "mixed_spin_coulomb_panel": panel.operator}
        primal.append(execute(programs.primal, fields).outputs["r2ab"])
        tangent.append(
            execute(programs.jvp.program, {**fields, **directions}).outputs["d_r2ab"]
        )
        active_seed = mixed_seed[:, :, panel.alpha_start : panel.alpha_stop, :]
        actual = execute(
            programs.vjp.program, {**fields, "bar_r2ab": active_seed}
        ).outputs
        for name, value in actual.items():
            bars[name] += value
    full = {**amplitudes, **factors}
    for name, actual, expected in (
        ("r2ab", primal, execute(reference.primal, full).outputs),
        (
            "d_r2ab",
            tangent,
            execute(reference.residual_jvp.program, {**full, **directions}).outputs,
        ),
    ):
        np.testing.assert_allclose(
            np.concatenate(actual, axis=2), expected[name], rtol=0, atol=4e-14
        )
    expected = execute(reference.residual_vjp.program, {**full, **seeds}).outputs
    for name, value in bars.items():
        np.testing.assert_allclose(value, expected[name], rtol=0, atol=4e-14)


def test_uhf_mixed_panel_preflights_host_before_native_call() -> None:
    def never_run(*values: object) -> None:
        raise AssertionError("Native fitting started before joint host admission")

    factors = np.zeros((2, 3, 3)), np.zeros((2, 4, 4))
    bounds, _ = uhf_mixed_virtual_panel_storage(3, 4, 2, 2)
    with pytest.raises(MemoryError, match="admitted host numeric budget"):
        prepare_uhf_mixed_virtual_panels(
            *factors, never_run, width=2, max_host_bytes=bounds - 1
        )


@pytest.mark.parametrize("spin,channel", [("alpha", "aa"), ("beta", "bb")])
def test_independent_packed_same_spin_virtual_pair_actions(
    spin: str, channel: str
) -> None:
    layout = UCCSDLayout(3, 2, 3, 4)
    rng = np.random.default_rng(737222)
    auxiliary = 5
    factors = {}
    seed = rng.normal(scale=0.3, size=(auxiliary, auxiliary))
    metric = seed @ seed.T + np.eye(auxiliary)
    for owner, extent in (("alpha", layout.va), ("beta", layout.vb)):
        raw = rng.normal(scale=0.1, size=(auxiliary, extent, extent))
        raw += raw.swapaxes(1, 2).copy()
        fit = np.linalg.solve(metric, raw.reshape(auxiliary, -1)).reshape(raw.shape)
        factors["fit_left_" + owner] = fit
        factors["raw_right_" + owner] = raw
    fit, raw = (factors[name + "_" + spin] for name in ("fit_left", "raw_right"))
    bound, retained = spatial_virtual_pair_storage(
        fit.shape[1], auxiliary, 3, include_symmetric=False
    )
    operator = prepare_spatial_virtual_pair_operators(
        fit, raw, execute, row_tile=3, max_bytes=bound, include_symmetric=False
    )
    assert set(operator) == {"spatial_coulomb_antisymmetric"}
    assert operator["spatial_coulomb_antisymmetric"].nbytes == retained
    packed = build_uccsd_uhf_same_spin_virtual_pair_programs(layout, spin)
    reference = build_uccsd_df_virtual_ladder_amplitude_programs(layout, auxiliary)
    amplitudes = layout.unpack(rng.normal(scale=0.1, size=layout.packed_size))
    tangents = {
        "d_" + name: value
        for name, value in layout.unpack(rng.normal(size=layout.packed_size)).items()
    }
    dense_fields, pair_fields = {**amplitudes, **factors}, {**amplitudes, **operator}
    residual = "r2" + channel
    for graph, control, extra in (
        (packed.primal, reference.primal, {}),
        (packed.jvp.program, reference.residual_jvp.program, tangents),
    ):
        actual = execute(graph, {**pair_fields, **extra}).outputs
        expected = execute(control, {**dense_fields, **extra}).outputs
        for name, value in actual.items():
            np.testing.assert_allclose(value, expected[name], rtol=0, atol=5e-15)
    packed_seed = layout.pack_dual(layout.unpack(rng.normal(size=layout.packed_size)))
    seeds = {
        "bar_" + name.replace("t", "r", 1): value
        for name, value in layout.unpack_dual(packed_seed).items()
    }
    for name, value in seeds.items():
        if name != "bar_" + residual:
            value.fill(0)
    actual = execute(packed.vjp.program, {**pair_fields, **seeds}).outputs
    expected = execute(
        reference.residual_vjp.program, {**dense_fields, **seeds}
    ).outputs
    actual_dual, expected_dual = (
        layout.pack_dual(
            {
                name: bars.get("bar_" + name, np.zeros_like(value))
                for name, value in amplitudes.items()
            }
        )
        for bars in (actual, expected)
    )
    np.testing.assert_allclose(actual_dual, expected_dual, rtol=0, atol=5e-14)


@pytest.mark.parametrize("beta_occupied", [0, 2])
def test_all_uhf_operators_admit_before_native_setup_and_share_physical_vjp(
    beta_occupied: int,
) -> None:
    layout = UCCSDLayout(3, beta_occupied, 3, 4)
    reference = build_uccsd_df_virtual_ladder_amplitude_programs(layout, 3)
    rng = np.random.default_rng(103497)
    factors = {}
    for spin, extent in (("alpha", layout.va), ("beta", layout.vb)):
        raw = rng.normal(scale=0.1, size=(3, extent, extent))
        raw += raw.swapaxes(1, 2).copy()
        factors["fit_left_" + spin] = raw.copy()
        factors["raw_right_" + spin] = raw

    def graph_resource(program: "Program") -> UHFVirtualGraphResources:
        _, plan = emit_cpu(program, max_bytes=1 << 24, max_work=1 << 30)
        return UHFVirtualGraphResources(plan["required_bytes"], 0, plan["scalar_work"])

    def never_run(*values: object) -> None:
        raise AssertionError("Native fitting started before joint work admission")

    limits = UHFVirtualLadderLimits(
        max_host_bytes=1 << 24,
        max_device_bytes=1 << 24,
        max_setup_work=1 << 20,
        max_action_work=1 << 20,
        pair_rows=2,
        mixed_alpha_width=2,
    )
    with pytest.raises(MemoryError, match="setup or action work"):
        prepare_uhf_virtual_ladder(
            layout,
            factors,
            never_run,
            graph_resource,
            replace(limits, max_setup_work=1),
        )
    with pytest.raises(MemoryError, match="host or device allowance"):
        prepare_uhf_virtual_ladder(
            layout,
            factors,
            never_run,
            graph_resource,
            replace(limits, max_host_bytes=1),
        )
    operator = prepare_uhf_virtual_ladder(
        layout, factors, execute, graph_resource, limits
    )
    assert (
        0
        < operator.retained_host_bytes
        <= operator.planned_host_bytes
        <= limits.max_host_bytes
    )
    amplitudes = layout.unpack(rng.normal(size=layout.packed_size))
    seeds = {
        "bar_" + name.replace("t", "r", 1): value
        for name, value in layout.unpack_dual(
            rng.normal(size=layout.packed_size)
        ).items()
    }
    expected_primal = execute(reference.primal, {**amplitudes, **factors}).outputs
    for name, value in operator.residual(amplitudes).items():
        np.testing.assert_allclose(value, expected_primal[name], rtol=0, atol=5e-14)
    expected_bars = execute(
        reference.residual_vjp.program, {**amplitudes, **factors, **seeds}
    ).outputs
    duals = (
        layout.pack_dual({name: bars["bar_" + name] for name in amplitudes})
        for bars in (operator.amplitude_bars(amplitudes, seeds), expected_bars)
    )
    np.testing.assert_allclose(*duals, rtol=0, atol=5e-14)
