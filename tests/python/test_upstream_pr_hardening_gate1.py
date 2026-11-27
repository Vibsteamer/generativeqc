"""Gate 1 unit tests for Upstream-PR Hardening Track B (B1, H1, H5, M2, M7, M8)."""

from __future__ import annotations
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# B1: Optional basis_set_exchange guard & packaging declaration
# ============================================================================


def test_b1_pyproject_declares_basis_set_exchange_optional_extra() -> None:
    """pyproject.toml declares basis-set-exchange in optional dependencies."""
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    opt = data["project"]["optional-dependencies"]
    assert any(dep.startswith("basis-set-exchange") for dep in opt["reference-test"])
    assert any(dep.startswith("basis-set-exchange") for dep in opt["bse"])
