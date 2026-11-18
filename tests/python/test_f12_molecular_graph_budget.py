"""Offline admission for the forward and reverse graphs in the DZ-F12 water test."""

from argparse import ArgumentParser

import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.mp2.f12_density_fitting import molecular_orbital_sources
from generativeqc_compiler.tensor.ad_program import transpose_program
from generativeqc_compiler.tensor.cuda_plan import plan_cuda

from benchmarks._f12_cuda import add_molecular_graph_budget


def test_water_source_and_pullback_fit_the_default_budget() -> None:
    parser = ArgumentParser()
    add_molecular_graph_budget(parser)
    budget = parser.parse_args([]).budget_mib << 20
    # Contracted cc-pVDZ-F12/OptRI H2O: 158 extended AOs and MOs, 5 occupied,
    # 198 auxiliary functions. The reverse graph retains additional AO weights.
    forward = molecular_orbital_sources(158, 158, 5, 198)
    reverse = transpose_program(forward, tuple(forward.outputs)).program
    for graph in (forward, reverse):
        plan = plan_cuda(graph, cuda_target_info("sm_89"), max_bytes=budget)
        assert 0 < plan.device_bytes + plan.host_bytes <= budget


def test_budget_override_and_invalid_mib() -> None:
    parser = ArgumentParser()
    add_molecular_graph_budget(parser)
    assert parser.parse_args(["--budget-mib", "768"]).budget_mib == 768
    with pytest.raises(SystemExit):
        parser.parse_args(["--budget-mib", "0"])
