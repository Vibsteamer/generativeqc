"""Private combined AO one-electron provider for a supplied scalar ECP residual."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
from generativeqc._one_electron_gradient import IntegralGradient

from benchmarks.f12_cabs_h_batch import CabsHShells


def _owned_fp64(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite FP64 with shape {shape}")
    return np.frombuffer(value.tobytes(order="C"), dtype=np.float64).reshape(shape)


@dataclass(frozen=True)
class ResidualEcpPanel:
    """Owned residual V and positive nuclear dV in the evaluator's AO order.

    AO row i maps to ao_to_source[i], with the same real-spherical phase and
    normalization. Nuclear rows have the one-electron provider's atom order.
    The panel excludes the effective-charge point attraction already in Hcore.
    """

    matrix: np.ndarray
    derivative: np.ndarray | None
    ao_to_source: np.ndarray = field(kw_only=True)
    coordinates_bohr: np.ndarray = field(kw_only=True)
    effective_charges: np.ndarray = field(kw_only=True)
    resources: Mapping[str, object] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        ao = np.asarray(self.ao_to_source)
        charges = np.asarray(self.effective_charges)
        if (
            ao.ndim != 1
            or ao.size == 0
            or ao.dtype.kind not in "iu"
            or not np.array_equal(np.sort(ao), np.arange(ao.size))
        ):
            raise ValueError("residual AO map must be a nonempty integer bijection")
        if (
            charges.ndim != 1
            or not charges.size
            or charges.dtype.kind not in "iu"
            or np.any(charges <= 0)
        ):
            raise ValueError("residual effective charges must be positive integers")
        matrix = _owned_fp64(self.matrix, (ao.size, ao.size), "residual ECP matrix")
        geometry = _owned_fp64(self.coordinates_bohr, (charges.size, 3), "residual ECP nuclei")
        if not np.allclose(matrix, matrix.T, rtol=0, atol=2e-9):
            raise ValueError("scalar residual ECP matrix must be symmetric")
        if self.derivative is not None:
            gradient = _owned_fp64(
                self.derivative, (charges.size, 3, *matrix.shape), "residual ECP nuclear derivative"
            )
            if not np.allclose(gradient, gradient.swapaxes(-1, -2), rtol=0, atol=2e-9):
                raise ValueError("scalar residual ECP nuclear derivative must be symmetric")
            object.__setattr__(self, "derivative", gradient)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "coordinates_bohr", geometry)
        object.__setattr__(self, "ao_to_source", np.frombuffer(ao.tobytes(), dtype=ao.dtype))
        object.__setattr__(
            self, "effective_charges", np.frombuffer(charges.tobytes(), dtype=charges.dtype)
        )
        object.__setattr__(self, "resources", MappingProxyType(dict(self.resources or {})))


class CabsEcpOneElectron:
    """Compose a native full-basis bare core and a matched residual ECP panel."""

    def __init__(self, bare: CabsHShells, residual: ResidualEcpPanel) -> None:
        if not isinstance(bare, CabsHShells) or not bare.full_basis:
            raise ValueError("the scalar ECP bridge requires the full-basis one-electron batch")
        self._bare = bare
        self._residual = residual
        self._batch = None
        self._check_basis()

    def _check_basis(self) -> None:
        bare, residual = self._bare, self._residual
        if (
            residual.ao_to_source.size != bare.n
            or not np.array_equal(residual.coordinates_bohr, bare.xyz)
            or not np.array_equal(residual.effective_charges, bare.charges)
        ):
            raise ValueError(
                "residual ECP panel does not match the one-electron AO count, geometry or charge"
            )

    def matrices(
        self, session_library: Path | None = None, *, derivatives: bool, max_bytes: int
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Evaluate S and Hcore in source AO order and retain the bare Jacobian."""
        self._batch = None
        self._check_basis()
        bare, residual = self._bare, self._residual
        if derivatives and residual.derivative is None:
            raise ValueError("residual ECP nuclear derivatives are required")
        native = bare.evaluate(derivatives=derivatives, max_bytes=max_bytes)
        overlap, hcore = bare.assemble()
        ix = np.ix_(residual.ao_to_source, residual.ao_to_source)
        hcore[ix] += residual.matrix
        if not np.isfinite(hcore).all():
            raise FloatingPointError("nonfinite complete ECP core Hamiltonian")
        if derivatives:
            self._batch = bare.data
        return overlap, hcore, {
            "backend": bare.backend,
            "device_bytes": max(native["device_bytes"], residual.resources.get("device_bytes", 0)),
            "bare": native,
            "residual": dict(residual.resources),
        }

    def gradient(
        self, ws: np.ndarray, wh: np.ndarray, max_bytes: int
    ) -> IntegralGradient:
        """Contract the retained (S, Hcore) derivatives in the source AO order."""
        self._check_basis()
        bare, residual = self._bare, self._residual
        if self._batch is None or self._batch is not bare.data:
            raise ValueError("evaluate the current complete AO nuclear derivatives first")
        if residual.derivative is None:
            raise ValueError("residual ECP nuclear derivatives are required")
        bare_gradient, bare_cost = bare.high_gradient(ws, wh, max_bytes)
        ix = np.ix_(residual.ao_to_source, residual.ao_to_source)
        residual_weights = np.ascontiguousarray(wh[ix])
        residual_gradient, residual_cost = bare._contract_nuclear_panel(
            residual.derivative, residual_weights, max_bytes
        )
        gradient = bare_gradient + residual_gradient
        if not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite complete ECP nuclear derivative")
        return IntegralGradient(
            gradient,
            {
                "device_bytes": max(bare_cost["device_bytes"], residual_cost["device_bytes"]),
                "bare_contraction_device_bytes": bare_cost["device_bytes"],
                "residual_contraction_device_bytes": residual_cost["device_bytes"],
                "native_contraction_calls": bare_cost["batch_calls"] + residual_cost["batch_calls"],
            },
        )
