"""Compiled geminal DF centers against independent Libint2 primitives."""

import json
import shutil
from pathlib import Path

import pytest
from generativeqc_compiler.integral.f12_df import emit_f12_df_component

from benchmarks.f12_df_integrals import check

ROOT = Path(__file__).resolve().parents[2]


def test_compiled_primitive_values_and_centers(tmp_path: Path) -> None:
    if not shutil.which("c++") or not (
        shutil.which("sccache") or shutil.which("ccache")
    ):
        pytest.skip("host compiler and compiler cache required")
    data = json.loads(
        (ROOT / "tests/reference_data/f12/libint_df_primitives.json").read_text()
    )
    # Keep the mandatory test small. The real-device qualification evaluates the
    # complete archived panel through orbital g and auxiliary h, including ggh.
    selected = (0, 1, 4, 5)
    data["components"] = [data["components"][i] for i in selected]
    data["rows"] = [row for row in data["rows"] if row["input"][0] in selected]
    for row in data["rows"]:
        row["input"][0] = selected.index(row["input"][0])
    report = check(data, tmp_path, None)
    assert report["backend"] == "host-fp64"
    assert report["max_value_and_derivative_scaled_error"] < 3e-10


@pytest.mark.parametrize(
    ("components", "name"),
    [
        (("",), "test"),
        (("aaaa", ""), "test"),
        (("xxxxxx", ""), "test"),
        (("", ""), "invalid-name"),
    ],
)
def test_reject_unsupported_components(components: tuple[str, ...], name: str) -> None:
    with pytest.raises(ValueError):
        emit_f12_df_component(components, name)
