"""Exercise the opt-in fixed-source UCCSD-F12b solve and adjoint owners."""

from dataclasses import replace

import numpy as np
import pytest
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.cc.uccsd_equations import build_program
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.tensor import execute
from test_uccsd_equations import _feeds, _random_hamiltonian
from test_uccsd_f12 import (
    _dense_correction,
    _pack_residual,
    _pair_sources,
    _PairSources,
    _source_feeds,
)

from tools.generativeqc_cc.native_tensor_cpu import NativeCCTensorExecutor
from tools.generativeqc_cc.solver import SolverOptions
from tools.generativeqc_cc.uccsd_fixed_f12 import (
    _OrbitalGapGMRES,
    _PreparedFixedSourceUCCSD,
    _PreparedFixedSourceUCCSDLambda,
)
from tools.generativeqc_cc.uccsd_lambda import UCCSDLambdaOptions
from tools.generativeqc_response.implicit import ImplicitSolveError

_LAYOUT = UCCSDLayout(2, 2, 2, 2)
_ENERGY_ALPHA = np.repeat((-0.8, 0.7), 2)
_ENERGY_BETA = np.repeat((-0.45, 0.9), 2)
_FOCK = np.diag(
    np.concatenate(
        (_ENERGY_ALPHA[:2], _ENERGY_BETA[:2], _ENERGY_ALPHA[2:], _ENERGY_BETA[2:])
    )
)
_REFERENCE_ENERGY = -2.3
_OPTIONS = SolverOptions(energy_tolerance=1e-13, residual_tolerance=2e-11)


@pytest.fixture(scope="module")
def executor(tmp_path_factory: pytest.TempPathFactory) -> NativeCCTensorExecutor:
    return NativeCCTensorExecutor(
        max_bytes=_OPTIONS.max_bytes,
        cache=tmp_path_factory.mktemp("uccsd-fixed-f12-native"),
    )


def _prepare(
    executor: NativeCCTensorExecutor,
    integrals: np.ndarray,
    sources: dict[str, np.ndarray],
    options: SolverOptions = _OPTIONS,
) -> _PreparedFixedSourceUCCSD:
    amplitudes = _LAYOUT.unpack(np.zeros(_LAYOUT.packed_size))
    ordinary = _feeds(_LAYOUT, _FOCK, integrals, amplitudes)
    return _PreparedFixedSourceUCCSD(
        _LAYOUT,
        {name: value for name, value in ordinary.items() if name not in amplitudes},
        sources,
        orbital_energies_alpha=_ENERGY_ALPHA,
        orbital_energies_beta=_ENERGY_BETA,
        reference_id="toy-uhf",
        hamiltonian_id="toy-uhf-hamiltonian",
        reference_energy=_REFERENCE_ENERGY,
        options=options,
        executor=executor,
    )


def test_zero_sources_recover_ordinary_energy_and_residuals(
    executor: NativeCCTensorExecutor,
) -> None:
    layout = _LAYOUT
    _, integrals = _random_hamiltonian(layout)
    amplitudes = np.random.default_rng(3215).normal(scale=0.03, size=layout.packed_size)
    ordinary = execute(
        build_program(layout.oa, layout.ob, layout.va, layout.vb),
        _feeds(layout, _FOCK, integrals, layout.unpack(amplitudes)),
    ).outputs
    sources = {
        name: np.zeros_like(value)
        for name, value in _source_feeds(layout, _pair_sources(layout, 53)).items()
    }
    with _prepare(executor, integrals, sources) as owner:
        actual = owner.evaluate(amplitudes)
    assert actual.keys() == ordinary.keys()
    for name in ordinary:
        np.testing.assert_allclose(actual[name], ordinary[name], rtol=0, atol=4e-14)


@pytest.mark.parametrize("preconditioned", [False, True])
def test_fixed_source_stationarity_and_cotangent_against_independent_root(
    executor: NativeCCTensorExecutor,
    preconditioned: bool,
) -> None:
    layout = _LAYOUT
    source = _pair_sources(layout, 2237)
    integrals = np.zeros((_FOCK.shape[0],) * 4)
    gap_a = _ENERGY_ALPHA[-1] - _ENERGY_ALPHA[0]
    gap_b = _ENERGY_BETA[-1] - _ENERGY_BETA[0]
    gaps = np.repeat(
        (gap_a, gap_b, 2 * gap_a, gap_a + gap_b, 2 * gap_b),
        [block.packed_layout.storage_elements for block in layout.blocks],
    )
    n_singles = layout.oa * layout.va + layout.ob * layout.vb

    def reference(
        amplitudes: np.ndarray, sources: _PairSources = source
    ) -> tuple[float, np.ndarray]:
        values = _dense_correction(layout, layout.unpack(amplitudes), sources)
        residual = gaps * amplitudes + _pack_residual(layout, values)
        return float(values["correlation_energy"]), residual

    def root(sources: _PairSources = source) -> np.ndarray:
        amplitudes = np.zeros(layout.packed_size)
        # Diagonal ordinary Fock makes each single equation independent.
        # The fixed-source doubles depend on singles, then their own gap.
        for domain in (slice(None, n_singles), slice(n_singles, None)):
            residual = reference(amplitudes, sources)[1]
            amplitudes[domain] = -residual[domain] / gaps[domain]
        return amplitudes

    expected = root()
    expected_energy, residual = reference(expected)
    np.testing.assert_allclose(residual, 0, rtol=0, atol=3e-16)
    assert np.linalg.norm(expected) > 0.01

    # A tiny explicit finite-difference oracle supplies the packed adjoint.
    eye = np.eye(layout.packed_size)
    step = 2e-5
    plus = [reference(expected + step * direction) for direction in eye]
    minus = [reference(expected - step * direction) for direction in eye]
    energy_gradient = np.array([p[0] - m[0] for p, m in zip(plus, minus, strict=True)])
    jacobian = np.column_stack([p[1] - m[1] for p, m in zip(plus, minus, strict=True)])
    expected_lambda = np.linalg.solve(jacobian.T, -energy_gradient)
    assert np.linalg.norm(expected_lambda) > 0.01

    # Perturb a complete antisymmetrized alpha-alpha block. This directional
    # derivative probes the nontrivial residual contribution to Lambda.
    direction = np.zeros_like(source.c)
    for i, j, ij in ((0, 1, 1), (1, 0, -1)):
        for a, b, ab in ((0, 1, 1), (1, 0, -1)):
            direction[i, j, a, b] = 0.27 * ij * ab
    shifted = [
        replace(source, c=source.c + sign * step * direction) for sign in (1, -1)
    ]
    energies = [reference(root(sources), sources)[0] for sources in shifted]
    expected_derivative = (energies[0] - energies[1]) / (2 * step)
    assert abs(expected_derivative) > 1e-6
    with _prepare(executor, integrals, _source_feeds(layout, source)) as owner:
        state = owner.solve()
        assert state.converged, state.reason
        assert state.correlation_energy == pytest.approx(
            expected_energy, rel=0, abs=2e-12
        )
        assert state.total_energy == pytest.approx(
            _REFERENCE_ENERGY + expected_energy, rel=0, abs=2e-12
        )
        np.testing.assert_allclose(
            state.packed_amplitudes, expected, rtol=0, atol=2e-11
        )
        archived_t = state.packed_amplitudes.copy()
        replayed_t = owner._replay_fixed_source_amplitudes(
            archived_t, state.correlation_energy
        )
        assert replayed_t.converged
        assert replayed_t.correlation_energy == state.correlation_energy
        archived_t[0] += 0.01
        np.testing.assert_array_equal(
            replayed_t.packed_amplitudes, state.packed_amplitudes
        )
        with pytest.raises(ImplicitSolveError, match="fresh physical replay"):
            owner._replay_fixed_source_amplitudes(archived_t, state.correlation_energy)
        with pytest.raises(ImplicitSolveError, match="fresh physical replay"):
            owner._replay_fixed_source_amplitudes(
                state.packed_amplitudes, state.correlation_energy + 1e-5
            )
        with _PreparedFixedSourceUCCSDLambda(
            owner, replayed_t, orbital_gap_preconditioner=preconditioned
        ) as response:
            result = response.solve()
            assert result.converged
            assert result.stationarity_max <= response.options.lambda_tolerance
            expected_backend = (
                "host-orbital-gap-response-gmres"
                if preconditioned
                else "python-response-gmres"
            )
            assert result.provenance["solver_backend"] == expected_backend
            np.testing.assert_allclose(
                result.packed_multipliers, expected_lambda, rtol=0, atol=3e-11
            )
            # Perturb all physical sources together and solve both displaced
            # dense spin-orbital problems, including their amplitude response.
            all_direction = _pair_sources(layout, 54722)
            changes = _source_feeds(layout, all_direction)
            derivative = response.fixed_source_directional_derivative(result, changes)
            for h in (2e-4, 1e-4):
                sources = [
                    _PairSources(
                        source.w + sign * h * all_direction.w,
                        source.c + sign * h * all_direction.c,
                        source.scalar + sign * h * all_direction.scalar,
                    )
                    for sign in (1, -1)
                ]
                ep, em = (reference(root(s), s)[0] for s in sources)
                assert derivative == pytest.approx((ep - em) / (2 * h), rel=0, abs=3e-10)
            with pytest.raises(ValueError, match="fixed-source direction"):
                response.fixed_source_directional_derivative(result, {})
            alpha_direction = direction[:2, :2, :2, :2].copy()
            cotangent = response.parameter_cotangent(result, "c_f12_aaaa")
            actual_derivative = cotangent.contract(alpha_direction)
            assert actual_derivative == pytest.approx(
                expected_derivative, rel=0, abs=2e-10
            )
            archived = result.packed_multipliers.copy()
            replay = response._replay_fixed_source_parameter_cotangent(
                archived, "c_f12_aaaa"
            )
            np.testing.assert_array_equal(replay.values, cotangent.values)
            assert replay.lambda_identity == result.lambda_identity
            assert (
                replay.provenance["primal_residual_max"]
                <= response.options.cc_tolerance
            )
            assert (
                replay.provenance["stationarity_max"]
                <= response.options.lambda_tolerance
            )
            archived[0] += 0.01
            with pytest.raises(
                ImplicitSolveError, match="fresh Lagrangian stationarity"
            ):
                response._replay_fixed_source_parameter_cotangent(
                    archived, "c_f12_aaaa"
                )
            assert replay.contract(alpha_direction) == actual_derivative
            for wrong in (archived[:-1], archived.astype(np.float32)):
                with pytest.raises(ValueError, match="native FP64"):
                    response._replay_fixed_source_parameter_cotangent(
                        wrong, "c_f12_aaaa"
                    )
            archived[0] = np.nan
            with pytest.raises(ValueError, match="finite"):
                response._replay_fixed_source_parameter_cotangent(
                    archived, "c_f12_aaaa"
                )


def test_fixed_source_binding_and_numeric_admission(
    executor: NativeCCTensorExecutor,
) -> None:
    source = _pair_sources(_LAYOUT, 228)
    integrals = np.zeros((_FOCK.shape[0],) * 4)
    feeds = _source_feeds(_LAYOUT, source)
    with pytest.raises(MemoryError):
        _prepare(executor, integrals, feeds, replace(_OPTIONS, max_bytes=1))
    bad_scalar = {**feeds, "fixed_geminal_energy": np.array([source.scalar])}
    with pytest.raises(ValueError, match="fixed_geminal_energy"):
        _prepare(executor, integrals, bad_scalar)

    changed = _source_feeds(_LAYOUT, replace(source, scalar=source.scalar + 0.1))
    with (
        _prepare(executor, integrals, feeds) as first,
        _prepare(executor, integrals, changed) as second,
    ):
        state = first.solve()
        assert state.converged, state.reason
        with pytest.raises(ValueError, match="identit"):
            _PreparedFixedSourceUCCSDLambda(second, state)
        restricted = UCCSDLambdaOptions(gmres=GMRESOptions(max_workspace_bytes=1))
        with pytest.raises(MemoryError, match="orbital-gap workspace"):
            _PreparedFixedSourceUCCSDLambda(
                first, state, options=restricted, orbital_gap_preconditioner=True
            )


@pytest.mark.parametrize(
    "gaps", [np.zeros(2), np.array([1.0, np.nan]), np.ones(3), np.array([1.0, 1e-18])]
)
def test_orbital_gap_solver_rejects_invalid_diagonal(gaps: np.ndarray) -> None:
    with pytest.raises(ValueError, match="orbital gaps|near-zero entry"):
        _OrbitalGapGMRES(dimension=2, options=GMRESOptions(), gaps=gaps)
