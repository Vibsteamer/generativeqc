"""Work-bounded exact fallback of the compensated CABS products."""

from __future__ import annotations

import fractions
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc._f12_focal_projector import (
    EXACT_PRODUCT_MULTIPLY_ADD_LIMIT,
    CompensatedProductWorkLimit,
    _ozaki_mm,
    _ozaki_mm_left_dd,
    compensated_cabs_orthonormality,
    full_rank_cabs_overlap_bar,
)
from generativeqc.experimental.f12_focal import (
    CompensatedProductWorkLimit as PublicCompensatedProductWorkLimit,
)
from generativeqc.experimental.f12_focal import (
    FocalOneElectron,
    focal_cabs_projector,
)
from scipy.linalg import cho_factor, cho_solve

_WATER = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"


def _deep_scale(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows spanning 2^-475 that only exact accumulation resolves; A @ B is ones."""
    layers = 20
    a = np.tile(2.0 ** (-25.0 * np.arange(layers)), (rows, 1))
    b = np.zeros((layers, cols))
    b[-1] = 2.0 ** (25 * (layers - 1))
    return a, b


@pytest.fixture
def no_rational_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object) -> None:
        raise AssertionError("exact rational accumulation started")

    monkeypatch.setattr(fractions, "Fraction", refuse)


def test_slice_limit_product_within_the_bound_is_exact() -> None:
    a, b = _deep_scale(64, 64)
    assert a.shape[0] * a.shape[1] * b.shape[1] <= EXACT_PRODUCT_MULTIPLY_ADD_LIMIT
    hi, lo = _ozaki_mm(a, b, slices=4)
    np.testing.assert_array_equal(hi, np.ones((64, 64)))
    np.testing.assert_array_equal(lo, np.zeros((64, 64)))
    hi, lo = _ozaki_mm_left_dd(a, b, np.zeros_like(b), slices=4)
    np.testing.assert_array_equal(hi + lo, np.ones((64, 64)))


def test_scaling_overflow_product_within_the_bound_is_exact() -> None:
    hi, lo = _ozaki_mm(np.full((8, 8), 2.0**1000), np.full((8, 8), 2.0**-1000))
    np.testing.assert_array_equal(hi, np.full((8, 8), 8.0))
    np.testing.assert_array_equal(lo, np.zeros((8, 8)))


@pytest.mark.usefixtures("no_rational_work")
def test_slice_limit_product_above_the_bound_is_refused() -> None:
    a, b = _deep_scale(103, 128)
    with pytest.raises(CompensatedProductWorkLimit) as refused:
        _ozaki_mm(a, b, slices=4)
    error = refused.value
    assert isinstance(error, RuntimeError)
    assert error.shape == (103, 20, 128)
    assert error.trigger == "slice_limit"
    assert error.multiply_adds == 103 * 20 * 128 > EXACT_PRODUCT_MULTIPLY_ADD_LIMIT
    assert error.limit == EXACT_PRODUCT_MULTIPLY_ADD_LIMIT
    assert "103x20 @ 20x128" in str(error)


@pytest.mark.usefixtures("no_rational_work")
def test_production_shaped_overflow_product_is_refused_before_rational_work() -> None:
    # A water cc-pVQZ-F12/OptRI overlap times its CABS frame is 341x341 @ 341x186.
    a = np.full((341, 341), 2.0**1000)
    b = np.full((341, 186), 2.0**-1010)
    with pytest.raises(PublicCompensatedProductWorkLimit) as refused:
        _ozaki_mm(a, b, slices=4)
    assert refused.value.shape == (341, 341, 186)
    assert refused.value.trigger == "scaling_overflow"
    assert refused.value.multiply_adds == 341 * 341 * 186


def _qz_cabs(obs_basis: str, cabs_basis: str) -> tuple[np.ndarray, int, np.ndarray]:
    gto = pytest.importorskip("pyscf.gto")
    orbital = gto.M(atom=_WATER, basis=obs_basis, verbose=0)
    cabs = gto.M(atom=_WATER, basis=cabs_basis, verbose=0)
    cross = gto.intor_cross("int1e_ovlp", orbital, cabs)
    s = np.block(
        [[orbital.intor("int1e_ovlp"), cross], [cross.T, cabs.intor("int1e_ovlp")]]
    )
    s = 0.5 * (s + s.T)
    obs = orbital.nao
    one = FocalOneElectron(
        s, s, None, None, obs, "geometry", "basis", "cpu_reference", 0
    )
    request = SimpleNamespace(
        role="hf_cabs",
        backend="cpu_reference",
        surface_id="geometry",
        basis=SimpleNamespace(basis_id="basis", cabs=object()),
    )
    projector = focal_cabs_projector(request, one)
    assert projector.rank == len(s) - obs
    return s, obs, projector.frame


@pytest.mark.usefixtures("no_rational_work")
@pytest.mark.parametrize(
    "obs_basis, cabs_basis",
    [("cc-pvqz-f12", "cc-pvqz-f12-optri"), ("aug-cc-pvqz", "aug-cc-pvqz-optri")],
)
def test_representative_qz_cabs_products_stay_in_the_ozaki_range(
    obs_basis: str, cabs_basis: str
) -> None:
    s, obs, frame = _qz_cabs(obs_basis, cabs_basis)
    n, rank = frame.shape
    gram, obs_dual = compensated_cabs_orthonormality(s, obs, frame)
    assert np.max(np.abs(gram - np.eye(rank))) < 1e-10
    assert np.max(np.abs(obs_dual)) < 1e-10

    # f(C) = <u u^T, C C^T> has bar_C = 2 u u^T C. With the full-rank kernel
    # C C^T = S^-1 - embed(S_oo^-1), its overlap slope has a closed form.
    rng = np.random.default_rng(20261126)
    u = s @ rng.normal(scale=0.1, size=(n, 3))
    bar_s = full_rank_cabs_overlap_bar(s, obs, frame, 2.0 * u @ (u.T @ frame))
    np.testing.assert_array_equal(bar_s, bar_s.T)
    solved = np.linalg.solve(s, u)
    solved_o = cho_solve(cho_factor(s[:obs, :obs], lower=True), u[:obs])
    l_s = np.linalg.cholesky(s)
    for _ in range(2):
        raw = rng.normal(size=(n, n))
        d_s = l_s @ (0.5 * (raw + raw.T)) @ l_s.T
        expected = -np.vdot(solved, d_s @ solved) + np.vdot(
            solved_o, d_s[:obs, :obs] @ solved_o
        )
        assert np.vdot(bar_s, d_s) == pytest.approx(expected, rel=1e-8)


@pytest.mark.usefixtures("no_rational_work")
def test_full_rank_overlap_bar_refuses_an_out_of_range_cotangent() -> None:
    s, obs, frame = _qz_cabs("cc-pvqz-f12", "cc-pvqz-f12-optri")
    n, rank = frame.shape
    with pytest.raises(CompensatedProductWorkLimit) as refused:
        full_rank_cabs_overlap_bar(s, obs, frame, np.full((n, rank), 2.0**1000))
    assert refused.value.shape == (rank, n, rank)
    assert refused.value.trigger == "scaling_overflow"
