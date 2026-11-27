"""The batch and dispersion interfaces import on Python 3.10."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("extensions", ["default", "blocked"])
def test_batch_and_dispersion_import_on_python310(extensions: str) -> None:
    executable = os.environ.get("GENERATIVEQC_TEST_PYTHON310")
    if not executable:
        if sys.version_info[:2] != (3, 10):
            pytest.skip("Run with Python 3.10 or set GENERATIVEQC_TEST_PYTHON310")
        executable = sys.executable

    source = Path(__file__).resolve().parents[2] / "python"
    script = """
import sys
from typing import get_type_hints

assert sys.version_info[:2] == (3, 10), sys.version
sys.path.insert(0, sys.argv[1])
if sys.argv[2] == "blocked":
    sys.modules["typing_extensions"] = None
from generativeqc import (
    D3CorrectionBatch, D4CorrectionBatch, PreparedBatch, R2SCAN3CCorrectionBatch,
)
from generativeqc._precision_batch import PreparedPrecisionBatch

for batch_type in (
    PreparedBatch, PreparedPrecisionBatch, D3CorrectionBatch,
    D4CorrectionBatch, R2SCAN3CCorrectionBatch,
):
    assert "return" in get_type_hints(batch_type.__enter__)
"""
    result = subprocess.run(
        [executable, "-I", "-B", "-c", script, str(source), extensions],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
