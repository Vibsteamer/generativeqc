"""Independently check the Eq. (12) occupied scalar and all three adjoints."""

from decimal import Decimal, localcontext

import pytest
from generativeqc_compiler.cc.tplus import build_tplus_program, evaluate_tplus


def test_joint_response_at_two_steps_matches_decimal_difference() -> None:
    # Give each orbital a different MP2 and F12 derivative and independently
    # move the triples as well. A constant-q approximation fails this test.
    fields = (
        ("-0.251", "-0.102", "-0.179"),
        ("-0.032", "-0.024", "-0.002"),
        ("-0.013", "-0.0021", "0.00032"),
    )
    directions = (
        ("0.011", "-0.020", "0.008"),
        ("0.007", "0.010", "-0.006"),
        ("0.009", "-0.003", "0.001"),
    )
    response = evaluate_tplus(*(tuple(map(float, field)) for field in fields))
    cotangents = (
        response.mp2_cotangent,
        response.f12_cotangent,
        response.triples_cotangent,
    )
    with localcontext() as context:
        context.prec = 48

        def energy(h: Decimal) -> Decimal:
            moved = [
                tuple(
                    Decimal(value) + h * Decimal(direction)
                    for value, direction in zip(field, tangent, strict=True)
                )
                for field, tangent in zip(fields, directions, strict=True)
            ]
            return sum((m + f) * t / m for m, f, t in zip(*moved, strict=True))

        expected = sum(
            value * float(direction)
            for gradient, tangent in zip(cotangents, directions, strict=True)
            for value, direction in zip(gradient, tangent, strict=True)
        )
        assert response.energy == pytest.approx(float(energy(Decimal(0))), abs=2e-17)
        for step in (Decimal("0.0004"), Decimal("0.0002")):
            central = float((energy(step) - energy(-step)) / (2 * step))
            assert central == pytest.approx(expected, abs=5e-11)


def test_disconnected_heterofragments_keep_their_own_factors() -> None:
    left = ((-0.20, -0.14), (-0.012, -0.009), (-0.027, -0.003))
    right = ((-0.24,), (-0.064,), (-0.014,))
    a, b = evaluate_tplus(*left), evaluate_tplus(*right)
    joined = tuple(x + y for x, y in zip(left, right, strict=True))
    pair = evaluate_tplus(*joined)
    assert pair.energy == pytest.approx(a.energy + b.energy, abs=1e-17)
    assert pair.mp2_cotangent == a.mp2_cotangent + b.mp2_cotangent
    assert pair.f12_cotangent == a.f12_cotangent + b.f12_cotangent
    assert pair.multipliers == a.multipliers + b.multipliers
    mp2, f12, triples = joined
    global_scale = (sum(mp2) + sum(f12)) / sum(mp2)
    assert abs(pair.energy - global_scale * sum(triples)) > 1e-4


def test_empty_spin_and_structural_zero_have_zero_response() -> None:
    empty = evaluate_tplus([], [], [])
    assert empty.energy == 0.0
    assert empty.multipliers == empty.mp2_cotangent == empty.f12_cotangent == ()
    sparse = evaluate_tplus(
        [-0.3, 0.0], [-0.03, -0.0], [-0.01, 0.0], inactive=[False, True]
    )
    active = evaluate_tplus([-0.3], [-0.03], [-0.01])
    assert sparse.energy == active.energy
    assert sparse.multipliers == (*active.multipliers, 0.0)
    assert sparse.mp2_cotangent == (*active.mp2_cotangent, 0.0)
    assert sparse.f12_cotangent == (*active.f12_cotangent, 0.0)


@pytest.mark.parametrize(
    "f12,triples", [(0.05, 0.02), (0.0, 0.02), (0.05, 0.0), (0.0, 0.0)]
)
def test_active_zero_denominator_is_singular(f12: float, triples: float) -> None:
    with pytest.raises(ValueError, match="orbital 0 has a singular MP2 denominator"):
        evaluate_tplus([0.0], [f12], [triples])


def test_small_regular_direction_has_finite_limit_without_a_floor() -> None:
    for scale in (1.0, 1e-100, 1e-250, 1e-308):
        response = evaluate_tplus([-scale], [-0.2 * scale], [-0.3 * scale])
        assert response.energy / scale == pytest.approx(-0.36)
        assert response.multipliers == pytest.approx((1.2,))
        assert response.mp2_cotangent == pytest.approx((-0.06,))
        assert response.f12_cotangent == pytest.approx((0.3,))
    # A complete-basis-limit energy reduces to conventional (T), with
    # a finite F12 cotangent remaining for departures from that limit.
    cbs = evaluate_tplus([-0.21], [0.0], [-0.004])
    assert cbs.energy == -0.004
    assert cbs.multipliers == (1.0,)
    assert cbs.f12_cotangent == pytest.approx((0.004 / 0.21,))


def test_inactive_flags_and_input_dimensions_are_enforced() -> None:
    with pytest.raises(ValueError, match="inactive orbital 0 has nonzero data"):
        evaluate_tplus([0.0], [0.01], [0.0], inactive=[True])
    with pytest.raises(ValueError, match="one Boolean inactivity flag"):
        evaluate_tplus([0.0], [0.0], [0.0], inactive=[1])
    with pytest.raises(ValueError, match="identical occupied dimensions"):
        evaluate_tplus([-0.21], [0.0], [])


def test_packed_program_lowers_all_responses_for_sm89() -> None:
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.tensor.cuda_emit import emit_cuda
    from generativeqc_compiler.tensor.cuda_plan import plan_cuda

    program = build_tplus_program(4)
    plan = plan_cuda(program, cuda_target_info("sm_89"))
    assert set(plan.program.outputs) == {
        "energy",
        "mp2_cotangent",
        "f12_cotangent",
        "triples_cotangent",
    }
    assert emit_cuda(plan)
