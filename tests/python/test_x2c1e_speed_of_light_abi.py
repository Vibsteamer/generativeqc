"""X2C-1e speed of light: 0 selects CODATA; negative or non-finite values are rejected."""

import hashlib
import math

import numpy as np
import pytest
from generativeqc import (
    Atom,
    BasisProvenance,
    BasisSet,
    BasisShell,
    ElementBasis,
    _x2c1e,
)

REJECTED = (-1.0, -_x2c1e.SPEED_OF_LIGHT_AU, -5e-324, math.nan, math.inf, -math.inf)


def shell(angular: int, exponent: str) -> BasisShell:
    return BasisShell(angular, (exponent,), (("1.0",),))


ATOMS = (Atom(10, (0.0, 0.0, 0.0)), Atom(1, (0.0, 0.3, 1.9)))
BASIS = BasisSet(
    "X2C speed-of-light probe",
    (
        ElementBasis(
            10, (shell(0, "4000"), shell(0, "60"), shell(1, "25"), shell(1, "0.8"))
        ),
        ElementBasis(1, (shell(0, "1.2"),)),
    ),
    BasisProvenance("synthetic", "1", "test-only", hashlib.sha256(b"neh+").hexdigest()),
    representation="spherical",
)
KW = {"charge": 1, "multiplicity": 1}


@pytest.mark.parametrize("delta_only", [False, True])
def test_zero_sentinel_is_codata_and_custom_values_are_used(delta_only: bool) -> None:
    hcore = lambda c: _x2c1e.x2c1e_hcore(
        ATOMS, BASIS, speed_of_light=c, delta_only=delta_only, **KW
    )
    codata = hcore(_x2c1e.SPEED_OF_LIGHT_AU)
    assert np.array_equal(hcore(0.0), codata)
    assert np.array_equal(hcore(-0.0), codata)
    assert not np.allclose(hcore(100.0), codata, rtol=0.0, atol=1e-10)
    _, configured, _ = _x2c1e.one_electron_integrals_cpu(
        ATOMS, BASIS, x2c1e_mode="dlu1e", speed_of_light=0.0, **KW
    )
    _, explicit, _ = _x2c1e.one_electron_integrals_cpu(
        ATOMS, BASIS, x2c1e_mode="dlu1e", speed_of_light=_x2c1e.SPEED_OF_LIGHT_AU, **KW
    )
    assert np.array_equal(configured, explicit)


@pytest.mark.parametrize("speed_of_light", REJECTED, ids=repr)
def test_negative_or_non_finite_speed_of_light_is_rejected(
    speed_of_light: float,
) -> None:
    weights = np.eye(_x2c1e._ao_count(ATOMS, BASIS))
    calls = (
        lambda: _x2c1e.x2c1e_hcore(ATOMS, BASIS, speed_of_light=speed_of_light, **KW),
        lambda: _x2c1e.x2c1e_hcore_derivatives(
            ATOMS, BASIS, speed_of_light=speed_of_light, **KW
        ),
        lambda: _x2c1e.x2c1e_weighted_gradient(
            ATOMS, BASIS, weights, speed_of_light=speed_of_light, **KW
        ),
        lambda: _x2c1e.one_electron_integrals_cpu(
            ATOMS, BASIS, x2c1e_mode="dlu1e", speed_of_light=speed_of_light, **KW
        ),
    )
    for call in calls:
        with pytest.raises(
            RuntimeError, match="GENERATIVEQC error 1: .*finite and positive"
        ):
            call()
