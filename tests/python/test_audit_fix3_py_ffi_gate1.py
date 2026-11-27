"""Executable Gate 1 counterexample tests for Audit Fix-3 (Python FFI, cache, ownership, binding, projector)."""

from __future__ import annotations

import ctypes as ct
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from generativeqc._f12_focal_integral_kernel import (
    LEGACY_SLATER_FIT_NORMALIZATION,
    LEGACY_SLATER_NORM_CODE,
    contract_geometry,
    slater_normalization_code,
)
from generativeqc._f12_focal_integral_sink import _bind_fallback_weighted


def test_m9_legacy_integral_bind_and_execute_consistent_norm_code(
    tmp_path: Path,
) -> None:
    """Audit m9: legacy fallback bind and contract_geometry agree identically on norm_code."""
    fake_lib = tmp_path / "lib_legacy_only.so"
    fake_lib.write_bytes(b"")

    def fake_legacy_kernel(
        ncenter, op, beta, nshell, nprim, nblocks, nweights,
        angular, prim_offsets, exponents, radial, centers,
        blocks, offsets, weights, output, budget, size_ptr,
    ):
        output[:] = 0.0
        output[:, 0] = 1.25
        ct.cast(size_ptr, ct.POINTER(ct.c_uint64))[0] = 64
        return 0

    def fake_bind(library, kernel, host_reference, *, extended=False):
        if extended:
            raise AttributeError("legacy library has no *_ex symbol")
        return fake_legacy_kernel

    norm_default = slater_normalization_code(LEGACY_SLATER_FIT_NORMALIZATION)
    norm_unscaled = slater_normalization_code("negative_exponential")
    assert norm_default == LEGACY_SLATER_NORM_CODE == 1
    assert norm_unscaled == 0

    # 1. Non-_ex bind: accepts compiled default norm_code=1 (1/beta), rejects norm_code=0.
    with patch("generativeqc._f12_focal_integral_sink.bind", side_effect=fake_bind):
        fn, is_ext = _bind_fallback_weighted(fake_lib, "geminal", True, norm_default)
        assert fn is fake_legacy_kernel and is_ext is False
        with pytest.raises(RuntimeError, match="SlaterFitNormalization"):
            _bind_fallback_weighted(fake_lib, "geminal", True, norm_unscaled)

    # 2. Non-_ex execute (contract_geometry with extended_abi=False):
    #    MUST agree identically — accepts norm_code=1 (1/beta), rejects norm_code=0!
    fixture = {
        "metric_blocks": np.array([[0, 0]], dtype=np.int32),
        "metric_offsets": np.array([0, 1], dtype=np.int32),
        "weights_metric_f12": np.array([1.0], dtype=np.float64),
        "angular": np.array([0], dtype=np.int32),
        "atom": np.array([0], dtype=np.int32),
        "prim_offsets": np.array([0, 1], dtype=np.int32),
        "exponents": np.array([1.0], dtype=np.float64),
        "radial": np.array([1.0], dtype=np.float64),
        "centers": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    }
    manifest = {
        "elements": ["H"],
        "geometry_bohr": [[0.0, 0.0, 0.0]],
        "options": {"f12_beta": 0.9},
    }

    label, row = contract_geometry(
        fixture, manifest, {"geminal": fake_legacy_kernel},
        "metric", "f12", 1 << 20, True,
        slater_normalization="negative_exponential_over_beta",
        extended_abi=False,
    )
    assert label == "metric_f12" and row["weighted_value_Eh"] == pytest.approx(1.25)

    with pytest.raises(RuntimeError, match="SlaterFitNormalization"):
        contract_geometry(
            fixture, manifest, {"geminal": fake_legacy_kernel},
            "metric", "f12", 1 << 20, True,
            slater_normalization="negative_exponential",
            extended_abi=False,
        )
