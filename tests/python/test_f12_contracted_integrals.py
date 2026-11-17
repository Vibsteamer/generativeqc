"""Actual radial contractions and pure-function projections against Libint2."""

import copy
import json
import shutil
from pathlib import Path

import pytest
from generativeqc_compiler.integral.f12_contracted import emit_f12_df_contracted

from benchmarks.f12_contracted_integrals import check

ROOT = Path(__file__).resolve().parents[2]


def test_contracted_values_and_nuclear_response(tmp_path: Path) -> None:
    if not shutil.which("c++") or not (
        shutil.which("sccache") or shutil.which("ccache")
    ):
        pytest.skip("host compiler and compiler cache required")
    data = json.loads(
        (ROOT / "tests/reference_data/f12/libint_contracted.json").read_text()
    )
    # The allocated L20 qualification additionally includes the g metric and
    # the longer oxygen core contraction. CI retains both Cartesian and pure d.
    selected = (0, 1)
    fixture = copy.deepcopy(data)
    fixture["groups"] = [data["groups"][i] for i in selected]
    fixture["rows"] = [row for row in fixture["rows"] if row["group"] in selected]
    for row in fixture["rows"]:
        row["group"] = selected.index(row["group"])
    report = check(fixture, tmp_path, None)
    assert report["max_value_and_gradient_scaled_error"] < 3e-10
    assert report["invalid_inputs_preserve_output"]


@pytest.mark.parametrize(
    ("components", "name"),
    [
        ((), "project"),
        ((("",),), "project"),
        ((("", ""), ("", "")), "project"),
        ((("x", ""), ("", "x")), "project"),
        ((("a", ""),), "project"),
        ((("", ""),), "bad name"),
    ],
)
def test_invalid_signature(components: tuple[tuple[str, ...], ...], name: str) -> None:
    with pytest.raises(ValueError):
        emit_f12_df_contracted(components, name)
