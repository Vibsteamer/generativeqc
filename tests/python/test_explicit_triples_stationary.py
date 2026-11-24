"""Check stationary explicit-triples actions against independent dense solves."""

import numpy as np
import pytest
from generativeqc_compiler.cc.explicit_triples_stationary import (
    build_explicit_triples_stationary_programs,
)
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    einsum,
    execute,
    input_tensor,
    multiply,
)


def field(name: str, *indices: Index) -> Node:
    return input_tensor(name, TensorSpec(indices, role="input", differentiable=True))


def test_projected_scalar_and_coupled_triples_stationarity() -> None:
    space = IndexSpace("physical_triples", "batch", 3)
    i, j = Index("i", space), Index("j", space)
    t3 = field("t3", i)
    source = field("source", i)
    action = field("action", i, j)
    gap = field("gap", i)
    linear = field("linear", i)
    quadratic = field("quadratic", i, j)
    residual = add(source, einsum("ij,j->i", action, t3), multiply(gap, t3))
    energy = add(
        einsum("i,i->", linear, t3),
        einsum("i,ij,j->", t3, quadratic, t3),
        coefficients=(1, "1/2"),
    )
    stationary = build_explicit_triples_stationary_programs(
        Program({"triples_energy": energy, "r3": residual}), {"t3": "r3"}
    )
    feeds = {
        "source": np.array([0.2, -0.3, 0.43]),
        "gap": np.array([1.45, 1.8, 2.13]),
        "action": np.array([[0.0, -0.24, 0.09], [0.42, 0.0, 0.1], [0.2, -0.16, 0.0]]),
        "linear": np.array([0.12, -0.03, -0.24]),
        "quadratic": np.array(
            [[0.01, -0.21, -0.03], [0.03, 0.41, 0.2], [0.04, 0.02, 0.16]]
        ),
    }

    def independently_solved_energy(fields: dict) -> float:
        matrix = fields["action"] + np.diag(fields["gap"])
        t = np.linalg.solve(matrix, -fields["source"])
        return float(fields["linear"] @ t + 0.5 * t @ fields["quadratic"] @ t)

    matrix = feeds["action"] + np.diag(feeds["gap"])
    t = np.linalg.solve(matrix, -feeds["source"])
    values = {**feeds, "t3": t}
    primal = execute(stationary.primal, values).outputs
    np.testing.assert_allclose(primal["r3"], 0.0, atol=1e-15)
    assert primal["triples_energy"] == pytest.approx(independently_solved_energy(feeds))

    direction, residual_seed = np.array([0.13, 0.5, -0.2]), np.array([0.35, -0.7, 0.42])
    tangent = execute(stationary.residual_jvp.program, {**values, "d_t3": direction})
    transpose = execute(
        stationary.residual_vjp.program, {**values, "bar_r3": residual_seed}
    )
    np.testing.assert_allclose(tangent.outputs["d_r3"], matrix @ direction, atol=1e-15)
    np.testing.assert_allclose(
        transpose.outputs["bar_t3"], matrix.T @ residual_seed, atol=1e-15
    )
    assert residual_seed @ tangent.outputs["d_r3"] == pytest.approx(
        direction @ transpose.outputs["bar_t3"], abs=1e-15
    )

    rhs = execute(
        stationary.lambda_rhs.program, {**values, "bar_triples_energy": np.array(1.0)}
    )
    independent_rhs = (
        -feeds["linear"] - 0.5 * (feeds["quadratic"] + feeds["quadratic"].T) @ t
    )
    np.testing.assert_allclose(rhs.outputs["bar_t3"], independent_rhs, atol=1e-15)
    multiplier = np.linalg.solve(matrix.T, rhs.outputs["bar_t3"])
    response = stationary.parameter_vjp(feeds)
    cotangents = execute(
        response.program,
        {**values, "bar_triples_energy": np.array(1.0), "bar_r3": multiplier},
    ).outputs
    rng = np.random.default_rng(27)
    for name, expected in {
        "source": multiplier,
        "action": np.outer(multiplier, t),
        "gap": multiplier * t,
        "linear": t,
        "quadratic": 0.5 * np.outer(t, t),
    }.items():
        np.testing.assert_allclose(cotangents["bar_" + name], expected, atol=1e-15)
        delta = rng.normal(scale=0.1, size=expected.shape)
        for h in (0.0004, 0.0002):
            plus, minus = (
                independently_solved_energy(
                    {**feeds, name: feeds[name] + sign * h * delta}
                )
                for sign in (1, -1)
            )
            assert (plus - minus) / (2 * h) == pytest.approx(
                np.vdot(expected, delta), abs=5e-11
            )


def test_cross_block_transpose_and_zero_scalar_rhs() -> None:
    i = Index("i", IndexSpace("block_one", "batch", 2))
    a = Index("a", IndexSpace("block_two", "batch", 2))
    first, second = field("first", i), field("second", a)
    mixing = field("mixing", a, i)
    select = field("select", i)
    energy = einsum("i,i->", first, select)
    coupled = add(second, einsum("ai,i->a", mixing, first))
    primal = Program({"triples_energy": energy, "r_first": first, "r_second": coupled})
    stationary = build_explicit_triples_stationary_programs(
        primal, {"second": "r_second", "first": "r_first"}
    )
    feeds = {
        "first": np.array([0.3, -0.2]),
        "second": np.array([-0.5, 0.1]),
        "mixing": np.array([[0.3, -0.7], [0.1, 0.4]]),
        "select": np.array([0.2, 0.9]),
    }
    rhs = execute(
        stationary.lambda_rhs.program, {**feeds, "bar_triples_energy": np.array(1.0)}
    )
    np.testing.assert_array_equal(rhs.outputs["bar_first"], -feeds["select"])
    np.testing.assert_array_equal(rhs.outputs["bar_second"], np.zeros(2))
    seed = np.array([0.4, 0.2])
    vjp = execute(
        stationary.residual_vjp.program,
        {**feeds, "bar_r_first": np.zeros(2), "bar_r_second": seed},
    )
    np.testing.assert_allclose(vjp.outputs["bar_first"], feeds["mixing"].T @ seed)
    np.testing.assert_array_equal(vjp.outputs["bar_second"], seed)
    with pytest.raises(ValueError, match="incompatible triples coordinate pair"):
        build_explicit_triples_stationary_programs(primal, {"first": "r_second"})
    with pytest.raises(ValueError, match="independent source parameters"):
        stationary.parameter_vjp(("first",))
