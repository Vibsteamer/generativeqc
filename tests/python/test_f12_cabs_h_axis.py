"""Shared Gaussian moments for high-angular CABS functions in orbital slots."""

import numpy as np
import pytest
from generativeqc_compiler.integral.df_derivatives import axis_polynomial
from generativeqc_compiler.integral.df_values import build_df_axis_moment


@pytest.mark.parametrize(
    "powers, maximum",
    [
        ((5, 0, 0), 5),
        ((5, 3, 4), 5),
        ((5, 5, 4), 5),
        ((6, 5, 4), 5),
        ((5, 6, 4), 5),
        ((5, 5, 5), 5),
        ((6, 6, 6), 6),
        ((7, 6, 6), 6),
        ((6, 7, 6), 6),
    ],
)
def test_high_cabs_angular_polynomial_against_gauss_hermite(
    powers: tuple[int, ...], maximum: int
) -> None:
    # Independent tensor-product Gaussian quadrature integrates the highest
    # polynomial degree 19 exactly, including an internally raised derivative.
    pa, pb, dx, sx, sy, ip, iq = (0.31, -0.24, 0.77, 0.4, 0.6, 0.45, 0.675)
    variables = {"pa": pa, "pb": pb, "dx": dx, "sx": sx, "sy": sy, "ip": ip, "iq": iq}
    graph, roots = axis_polynomial(
        *powers,
        auxiliary_h_derivative=maximum == 5,
        auxiliary_i_derivative=maximum == 6,
        maximum_orbital=maximum,
    )
    polynomial = np.array([graph.evaluate(root, variables) for root in roots])
    nodes, weights = np.polynomial.hermite.hermgauss(10)
    u1, u2 = np.meshgrid(np.sqrt(2) * nodes, np.sqrt(2) * nodes, indexing="ij")
    for t2 in (0.0, 0.25, 0.84):
        var_x, cov, var_y = ip * (1 - sx * t2), ip * sy * t2, iq * (1 - sy * t2)
        z = np.sqrt(var_x) * u1
        y = cov / np.sqrt(var_x) * u1 + np.sqrt(var_y - cov**2 / var_x) * u2
        integral = (
            (pa - dx * sx * t2 + z) ** powers[0]
            * (pb - dx * sx * t2 + z) ** powers[1]
            * (dx * sy * t2 + y) ** powers[2]
        )
        expected = weights @ integral @ weights / np.pi
        assert np.polynomial.polynomial.polyval(t2, polynomial) == pytest.approx(
            expected, rel=2e-12, abs=2e-12
        )


@pytest.mark.parametrize(
    "powers, options",
    [
        ((5, 5, 4), {"auxiliary_h_derivative": True, "maximum_orbital": 4}),
        ((5, 5, 4), {"maximum_orbital": 5}),
        ((6, 5, 4), {"auxiliary_h": True, "maximum_orbital": 5}),
        ((6, 6, 4), {"auxiliary_h_derivative": True, "maximum_orbital": 5}),
        ((5, 5, 6), {"auxiliary_h_derivative": True, "maximum_orbital": 5}),
        ((6, 6, 6), {"maximum_orbital": 6}),
        ((7, 7, 6), {"auxiliary_i_derivative": True, "maximum_orbital": 6}),
        ((6, 6, 7), {"auxiliary_i_derivative": True, "maximum_orbital": 6}),
    ],
)
def test_cabs_h_angular_scope(powers: tuple[int, ...], options: dict) -> None:
    with pytest.raises(ValueError):
        build_df_axis_moment(*powers, **options)


@pytest.mark.parametrize(
    "options",
    [
        {"maximum_orbital": 5, "auxiliary_h": True},
        {"cabs_h": True, "auxiliary_h": True, "maximum_orbital": 4},
        {"cabs_h": True, "maximum_orbital": 5},
        {
            "cabs_h": True,
            "maximum_orbital": 5,
            "auxiliary_h": True,
            "scalar_type": "float",
        },
    ],
)
def test_molecular_h_source_requires_the_explicit_bounded_variant(
    options: dict,
) -> None:
    from generativeqc_compiler.integral.df_derivatives_cuda import (
        emit_df_derivatives_cuda,
    )

    with pytest.raises(ValueError):
        emit_df_derivatives_cuda(**options)
