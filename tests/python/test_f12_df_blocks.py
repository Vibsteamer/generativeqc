"""Regression checks for Coulomb-metric robust geminal fitting and its response."""

import pytest
from generativeqc_compiler.mp2.f12_density_fitting import fitted_integral_block
from generativeqc_compiler.tensor import execute

from benchmarks.f12_df_component import molecular_case, response_case


@pytest.mark.parametrize("case", ["h2", "water"])
def test_robust_occupied_pair_projection(case: str) -> None:
    molecular_case(case, execute)


def test_rectangular_block_and_adjoint() -> None:
    response_case(execute)


@pytest.mark.parametrize("dimensions", [(0, 1, 2), (1, -1, 2), (5, 2, True)])
def test_positive_block_sizes(dimensions: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="positive dimensions"):
        fitted_integral_block(*dimensions)
