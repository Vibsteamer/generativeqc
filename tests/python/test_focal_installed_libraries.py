"""Installed F12 library selection through the public molecular provider."""

import ctypes
from pathlib import Path
from unittest.mock import Mock

import pytest
from generativeqc import _native
from generativeqc.experimental.f12_focal import MolecularFocalProvider


def test_molecular_provider_selects_only_its_installed_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "site-packages" / "generativeqc"
    libdir = package / "lib"
    libdir.mkdir(parents=True)
    host = libdir / "libfocal_shell_i_host.so"
    host.touch()
    monkeypatch.setattr(_native, "PACKAGE_DIR", package)

    assert MolecularFocalProvider().shell_library == host
    assert _native.installed_f12_shell_library(host_reference=True) == host
    assert _native.installed_f12_shell_library(host_reference=False) is None

    kwargs = {"architecture": "sm_89", "cache_directory": tmp_path / "cache"}
    with pytest.raises(RuntimeError, match="focal_shell_i'"):
        MolecularFocalProvider.cuda(**kwargs)
    cuda = libdir / "libfocal_shell_i.so"
    cuda.touch()
    with pytest.raises(RuntimeError, match="focal_hf_cabs_response"):
        MolecularFocalProvider.cuda(**kwargs)

    response = libdir / "libfocal_hf_cabs_response.so"
    response.touch()
    monkeypatch.setattr(_native, "load_library", lambda **kw: Mock())
    opened = Mock()
    monkeypatch.setattr(ctypes, "CDLL", opened)
    monkeypatch.setenv("PATH", "")

    provider = MolecularFocalProvider.cuda(**kwargs)
    assert provider.shell_library == cuda
    assert provider.metric_library == response
    opened.assert_called_once_with(str(response))
    assert provider.response_library is opened.return_value
    assert provider.run.profile_snapshot()["jit_compile_count"] == 0
    with pytest.raises(RuntimeError, match="requires nvcc"):
        provider.run(Mock(logical_hash="needs-tensorir"), {})

    explicit = tmp_path / "qualified-response.so"
    provider = MolecularFocalProvider.cuda(response_library=explicit, **kwargs)
    assert provider.shell_library == cuda
    assert provider.metric_library == explicit
    opened.assert_called_with(str(explicit))

    host.unlink()
    with pytest.raises(RuntimeError, match="focal_shell_i_host"):
        MolecularFocalProvider()
