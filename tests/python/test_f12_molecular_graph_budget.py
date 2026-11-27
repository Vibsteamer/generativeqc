"""Offline admission for the forward and reverse graphs in the DZ-F12 water test."""

from argparse import ArgumentParser

import pytest

from benchmarks._f12_cuda import add_molecular_graph_budget




def test_budget_override_and_invalid_mib() -> None:
    parser = ArgumentParser()
    add_molecular_graph_budget(parser)
    assert parser.parse_args(["--budget-mib", "768"]).budget_mib == 768
    with pytest.raises(SystemExit):
        parser.parse_args(["--budget-mib", "0"])
