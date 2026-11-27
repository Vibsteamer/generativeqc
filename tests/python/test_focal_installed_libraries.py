"""Select the installed F12 shell for the requested backend."""

from pathlib import Path

import pytest
from generativeqc import _native


def test_installed_f12_shell_matches_requested_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "site-packages" / "generativeqc"
    libdir = package / "lib"
    libdir.mkdir(parents=True)
    host = libdir / "libfocal_shell_i_host.so"
    host.touch()
    monkeypatch.setattr(_native, "PACKAGE_DIR", package)

    assert _native.installed_f12_shell_library(host_reference=True) == host
    assert _native.installed_f12_shell_library(host_reference=False) is None
