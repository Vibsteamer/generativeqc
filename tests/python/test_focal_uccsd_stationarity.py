import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc._f12_focal_cc_fixed import fixed_ucc_source_forces
from generativeqc._f12_ucc_solver import solve_focal_uccsd_f12b_with_cotangents
from generativeqc.experimental.f12_focal import MolecularFocalProvider
from test_focal_molecular_provider import XYZ, request

from tools.generativeqc_cc.solver import SolverOptions
from tools.generativeqc_cc.uccsd_focal_state import solve_focal_uccsd_f12b


def test_h3_cold_focal_ucc_stationarity() -> None:
    library = os.environ.get("GENERATIVEQC_F12_HOST_SINK")
    if not library:
        pytest.skip("set GENERATIVEQC_F12_HOST_SINK for the molecular shell library")
    provider = MolecularFocalProvider(shell_library=Path(library))
    molecule = request(XYZ, (0, 1, 2))
    sources = provider.ucc_equation_sources(molecule)
    state, dual, cotangents = solve_focal_uccsd_f12b_with_cotangents(
        sources,
        options=SolverOptions(residual_tolerance=3e-12, energy_tolerance=5e-13),
    )
    assert state.converged and dual.converged
    assert cotangents.ordinary_weights.keys() == sources.ordinary.keys()
    assert sources.fixed.feeds.keys() - cotangents.fixed_source_weights.keys() == {
        "w_vo_baab"
    }
    assert cotangents.fixed_source_weights["fixed_geminal_energy"] == 1.0
    # Five-point differences of each independently regenerated canonical input.
    finite_difference = [
        [-0.001841326826487281, 0.00038214690550296876, 0.0005076633792222099],
        [0.0006539511085712953, 0.002158614768749586, 5.824458154205884e-05],
        [0.0011873757019796594, -0.0025407616670361046, -0.0005659079618224502],
    ]
    np.testing.assert_allclose(
        fixed_ucc_source_forces(
            provider, molecule, sources, cotangents.fixed_source_weights
        ),
        finite_difference,
        atol=5e-10,
        rtol=0,
    )
    assert state.packed_amplitudes.shape == dual.packed_multipliers.shape == (59,)
    assert dual.primal_residual_max < 4e-12 and dual.stationarity_max < 1e-9
    assert len(state.history) > 1
    # Independently optimized by a dense SciPy solve of the NumPy residuals.
    assert state.correlation_energy == pytest.approx(
        -0.06269423582838254, abs=2e-11, rel=0
    )
    np.testing.assert_allclose(
        state.history[0].correlation_energy,
        sources.fixed.feeds["fixed_geminal_energy"],
        rtol=0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        state.total_energy,
        sources.fixed.reference.energy + state.correlation_energy,
        rtol=0,
        atol=1e-13,
    )
    cuda_ref = replace(sources.fixed.reference, executed_backend="cuda")
    with pytest.raises(ValueError, match="share a backend"):
        solve_focal_uccsd_f12b(
            replace(sources, fixed=replace(sources.fixed, reference=cuda_ref))
        )
