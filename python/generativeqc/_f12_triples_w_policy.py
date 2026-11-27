"""Device admission and provenance of single-pass FP32 triples W analytic forces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._cuda_device import CudaDeviceIdentity, device_identity

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping

FP32_SINGLE_W_FORCES_DIAGNOSTIC = "fp32_single_w_forces"
# Devices whose single-pass FP32 W forces met the FP64-reference gates
# (1e-8 Eh/bohr for forces, 1e-9 Eh for energies).
SINGLE_PASS_W_FORCE_DEVICES = frozenset({CudaDeviceIdentity("NVIDIA L20", "sm_89")})
HOST_DEVICE_NAME = "cpu"


class TriplesWPrecisionError(ValueError):
    """The requested triples W precision is unavailable for analytic forces here."""


def mark_diagnostic_result(telemetry: MutableMapping[str, object], mode: str) -> None:
    """Tag a result whose numerics are outside the production acceptance gates."""
    telemetry["production_eligible"] = False
    telemetry["diagnostic_mode"] = mode


def _describe(device: CudaDeviceIdentity | None) -> str:
    if device is None:
        return "the host CPU backend"
    return f"{device.name!r} ({device.architecture})"


@dataclass(frozen=True)
class TriplesWForceAdmission:
    """The W schedule admitted for one analytic triples force evaluation.

    device is the CUDA device of the native work, or None on the host backend.
    single_pass marks an explicit single-pass FP32 W request, and diagnostic
    the non-production comparison of that schedule.
    """

    device: CudaDeviceIdentity | None
    preconvergence: str
    single_pass: bool
    diagnostic: bool

    def record(self, telemetry: MutableMapping[str, object]) -> None:
        """Add the device and preconvergence to the executed W telemetry.

        Refuse a different nonempty schedule or a resource fallback in
        production single-pass forces.
        """
        executed = telemetry.get("w_precision_schedule")
        if self.single_pass and executed not in ("fp32_single", "n/a"):
            raise TriplesWPrecisionError(
                "Single-pass FP32 triples W was requested for analytic forces, but the "
                f"native triples executed w_precision_schedule={executed!r}"
            )
        if (
            self.single_pass
            and not self.diagnostic
            and telemetry.get("triples_fp32_w_resource_fallback") is True
        ):
            raise TriplesWPrecisionError(
                "Single-pass FP32 triples W was requested for analytic forces, but "
                "the native triples used a resource fallback "
                f"(forward_w_precision={telemetry.get('forward_w_precision')!r}, "
                f"reverse_w_precision={telemetry.get('reverse_w_precision')!r})"
            )
        telemetry.setdefault("preconvergence", self.preconvergence)
        telemetry["device_name"] = (
            HOST_DEVICE_NAME if self.device is None else self.device.name
        )
        telemetry["device_architecture"] = (
            "n/a" if self.device is None else self.device.architecture
        )
        if self.diagnostic:
            mark_diagnostic_result(telemetry, FP32_SINGLE_W_FORCES_DIAGNOSTIC)


def admit_triples_w_forces(
    backend: str,
    *,
    triples_fp32_w: bool,
    triples_w_schedule: str | None,
    preconvergence: str,
    cuda_device: Callable[[], CudaDeviceIdentity] | None,
    diagnostic_fp32_single_w_forces: bool,
) -> TriplesWForceAdmission:
    """Admit the requested W schedule for analytic forces before any native work.

    Single-pass FP32 W runs for production only on a device in
    SINGLE_PASS_W_FORCE_DEVICES. diagnostic_fp32_single_w_forces=True runs it on
    any device and tags the result. cuda_device identifies the CUDA device;
    None queries ordinal 0 from the driver.
    """
    if not isinstance(diagnostic_fp32_single_w_forces, bool):
        raise TypeError("diagnostic_fp32_single_w_forces must be a boolean")
    single_pass = bool(triples_fp32_w) or triples_w_schedule == "fp32_single"
    if diagnostic_fp32_single_w_forces and not single_pass:
        raise ValueError(
            "diagnostic_fp32_single_w_forces=True requires the single-pass FP32 W schedule "
            "(triples_fp32_w=True)"
        )
    device = None
    if backend == "cuda":
        device = (cuda_device or device_identity)()
        if not isinstance(device, CudaDeviceIdentity):
            raise TypeError("the CUDA device query must return a CudaDeviceIdentity")
    if (
        single_pass
        and not diagnostic_fp32_single_w_forces
        and device not in SINGLE_PASS_W_FORCE_DEVICES
    ):
        raise TriplesWPrecisionError(
            "Single-pass FP32 triples W (triples_fp32_w=True) is admitted for analytic "
            f"forces only on an NVIDIA L20 (sm_89); this evaluation runs on "
            f"{_describe(device)}. Request the default FP64 W on this device, or pass "
            "diagnostic_fp32_single_w_forces=True for a non-production diagnostic."
        )
    return TriplesWForceAdmission(
        device, preconvergence, single_pass, diagnostic_fp32_single_w_forces
    )


def reject_unqualified_single_pass_w_forces(
    telemetry: Mapping[str, object], *, executed_backend: str | None
) -> None:
    """Validate the schedule and device recorded for a production force result."""
    executed = telemetry.get("w_precision_schedule")
    if (
        executed != "fp32_single"
        and telemetry.get("requested_w_precision_schedule") != "fp32_single"
        and telemetry.get("triples_fp32_w") is not True
    ):
        return
    if executed not in ("fp32_single", "n/a"):
        raise TriplesWPrecisionError(
            "single-pass FP32 triples W forces enter production focal validation or "
            f"assembly only with matching W telemetry, got w_precision_schedule={executed!r}"
        )
    if telemetry.get("triples_fp32_w_resource_fallback") is True:
        raise TriplesWPrecisionError(
            "single-pass FP32 triples W forces that used a resource fallback "
            "cannot enter production focal validation or assembly"
        )
    name, architecture = (
        telemetry.get("device_name"),
        telemetry.get("device_architecture"),
    )
    if (
        executed_backend != "cuda"
        or not isinstance(name, str)
        or not isinstance(architecture, str)
        or CudaDeviceIdentity(name, architecture) not in SINGLE_PASS_W_FORCE_DEVICES
    ):
        raise ValueError(
            "single-pass FP32 triples W forces enter production focal validation or "
            "assembly only from an NVIDIA L20 (sm_89) CUDA backend. This component "
            f"records backend={executed_backend!r}, device_name={name!r}, "
            f"device_architecture={architecture!r}"
        )
