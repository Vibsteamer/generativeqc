"""Gate 1 regression & counterexample tests for Audit Fix-4 (Python endpoints, budget, AOT, Delta-CV, hygiene)."""

from __future__ import annotations
from pathlib import Path
import pytest


def test_d4_host_libraries_reject_invalid_explicit_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An invalid explicit test library must fail before any alternative is selected."""
    from conftest import f12_host_libraries

    missing = tmp_path / "does_not_exist.so"
    monkeypatch.setenv("GENERATIVEQC_F12_HOST_SINK", str(missing))
    with pytest.raises(FileNotFoundError, match=missing.name):
        f12_host_libraries()

    shell = tmp_path / "shell.so"
    shell.touch()
    monkeypatch.setenv("GENERATIVEQC_F12_HOST_SINK", str(shell))
    monkeypatch.setenv("GENERATIVEQC_F12_ONE_ECP", str(missing))
    with pytest.raises(FileNotFoundError, match=missing.name):
        f12_host_libraries()
    monkeypatch.delenv("GENERATIVEQC_F12_ONE_ECP")
    assert f12_host_libraries() == (shell, shell)
