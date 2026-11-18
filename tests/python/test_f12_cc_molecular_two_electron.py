"""Differentiate the same fitted ordinary Coulomb tensor the bounded CC driver consumes."""

import numpy as np
import pytest
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.ad_program import transpose_program

from benchmarks.f12_h2_cc_molecular_correlation import (
    _small_df_coulomb,
    h2_cc_correlation,
)


def test_bounded_cc_coulomb_adjoint_includes_both_electron_lines() -> None:
    rng = np.random.default_rng(5711)
    fit, raw, dfit, draw = rng.normal(size=(4, 5, 3, 3))
    bar = rng.normal(size=(3, 3, 3, 3))
    graph = _small_df_coulomb(3, 5)
    output = execute(graph, {"fit": fit, "raw": raw}).outputs["eri"]
    expected = np.einsum("pij,pkl->ijkl", fit, raw)
    np.testing.assert_allclose(output, expected, atol=2e-14, rtol=0)
    g = execute(
        transpose_program(graph, ("eri",)).program,
        {"fit": fit, "raw": raw, "bar_eri": bar},
    ).outputs
    actual = np.vdot(dfit, g["bar_fit"]) + np.vdot(draw, g["bar_raw"])
    ref = np.einsum("ijkl,pij,pkl", bar, dfit, raw) + np.einsum(
        "ijkl,pij,pkl", bar, fit, draw
    )
    np.testing.assert_allclose(actual, ref, atol=2e-13, rtol=0)


@pytest.mark.parametrize("spaces", ((2, 3, 6), (1, 25, 30)))
def test_cc_two_electron_driver_enforces_occupied_and_diagnostic_limits(
    spaces: tuple[int, int, int],
) -> None:
    with pytest.raises(ValueError, match="one occupied pair"):
        h2_cc_correlation({"space": spaces}, {}, np.eye(3), execute, pullback=True)
