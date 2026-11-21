"""Check bounded unlike-spin panels against a whole independent-spin DF graph."""

import numpy as np
import pytest
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
)
from generativeqc_compiler.tensor import execute


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
