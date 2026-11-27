"""Single-pass FP32 triples W forces: device admission, diagnostic and production assembly."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import numpy as np
import pytest
from generativeqc import Atom
from generativeqc import _f12_focal_ucc_triples as kernels
from generativeqc import _f12_triples_w_policy as policy
from generativeqc._cuda_device import CudaDeviceIdentity
from generativeqc._f12_focal_molecular import MolecularFocalProvider
from generativeqc._f12_focal_reference import FocalReference
from generativeqc._f12_focal_tplus_sources import OccupiedMP2F12
from generativeqc._f12_frozen_core import (
    assemble_production_focal_energy_and_forces,
    validate_production_focal_recipe,
)
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalEnergyComponent,
    FocalEnergyResult,
    FocalEvaluation,
    FocalLowOrderEnergy,
    FocalLowOrderForces,
    FocalMolecularResult,
    FocalRunResult,
    FocalSeries,
    FocalTerm,
    TriplesWPrecisionError,
    plan_singlepoint,
    singlepoint,
)

if TYPE_CHECKING:
    from generativeqc.experimental.f12_focal import FocalRequest

_BASIS = FocalBasis(
    2,
    "single-pass-w-force-policy",
    "sto-3g",
    jk_fit="def2-svp",
    mp2_fit="def2-svp",
    f12_fit="def2-svp",
    cabs="def2-svp",
    beta=1.0,
)
_WATER = (
    Atom("O", (0.0, 0.0, 0.0)),
    Atom("H", (1.43, 0.0, 1.11)),
    Atom("H", (-1.43, 0.0, 1.11)),
)
_HYDROXYL = (Atom("O", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 1.83)))
_ENTRIES = ("rhf_tplus", "uhf_tplus", "conventional")
_HOST_ENTRIES = ("uhf_tplus", "conventional")
_PRECISION = {
    "fp64": {},
    "fp32_split3": {"triples_w_schedule": "fp32_split3"},
    "fp32_single": {"triples_fp32_w": True},
}
_L20 = CudaDeviceIdentity("NVIDIA L20", "sm_89")
_A100 = CudaDeviceIdentity("NVIDIA A100-SXM4-80GB", "sm_80")
# Shares the L20 compute capability without its single-pass acceptance evidence.
_L40S = CudaDeviceIdentity("NVIDIA L40S", "sm_89")
_UNQUALIFIED = pytest.mark.parametrize("device", [_A100, _L40S], ids=["a100", "l40s"])


def _request(root_id: str, role: str, backend: str) -> FocalRequest:
    atoms, multiplicity = (_WATER, 1) if root_id == "rhf" else (_HYDROXYL, 2)
    plan = plan_singlepoint(
        atoms,
        FocalSeries(_BASIS),
        FocalSeries(_BASIS),
        None,
        charge=0,
        multiplicity=multiplicity,
        root_id=root_id,
        correlation_space_id="all-active",
        backend=backend,
    )
    pair = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    return replace(pair, role=role)


def _reference(request: FocalRequest, occupied: tuple[int, ...]) -> FocalReference:
    n = 7
    spin = len(occupied)
    eye = np.broadcast_to(np.eye(n), (spin, n, n)).copy()
    eps = np.broadcast_to(np.arange(1, n + 1, dtype=np.float64), (spin, n)).copy()
    return FocalReference(
        -75.0,
        None,
        np.eye(n),
        np.eye(n),
        eye,
        eps,
        eye,
        eye,
        occupied,
        3,
        request.backend,
        request.surface_id,
        request.basis.basis_id,
        request.basis.orbital.identity,
        request.basis.jk_fit.identity,
    )


def _low_order(
    reference: FocalReference, occupied: tuple[int, int]
) -> FocalLowOrderEnergy:
    partitions = OccupiedMP2F12(
        (np.full(occupied[0], -0.015), np.full(occupied[1], -0.015)),
        (np.full(occupied[0], -0.005), np.full(occupied[1], -0.005)),
    )
    return FocalLowOrderEnergy(
        reference, 2, (0.0, 0.0), (-0.05, -0.05, -0.05), -0.05, partitions
    )


def _never_run(*_args: object) -> None:
    raise AssertionError("no TensorIR graph is expected")


def _provider(
    backend: str,
    precision: str,
    device: CudaDeviceIdentity | None = _L20,
    **settings: Any,
) -> MolecularFocalProvider:
    """A provider whose CUDA device query returns device; None keeps the driver query."""
    settings.update(_PRECISION[precision])
    if backend == "cuda":
        settings.update(
            run=_never_run,
            metric_library=Path("fake-metric.so"),
            response_library=SimpleNamespace(),
        )
        if device is not None:
            settings["_cuda_device"] = lambda: device
    return MolecularFocalProvider(shell_library=Path("fake-shell.so"), **settings)


def _native_telemetry(kwargs: dict[str, Any]) -> dict[str, object]:
    """The executed W schedule, as the v2 native triples report it."""
    forces = kwargs.get("forces")
    if kwargs.get("triples_fp32_w"):
        schedule, forward, reverse = "fp32_single", "fp32", "fp64"
    elif kwargs.get("triples_w_schedule") == "fp32_split3":
        schedule, forward, reverse = "fp32_split3", "fp32", "fp32"
    else:
        schedule, forward, reverse = "fp64", "fp64", "fp64"
    return {
        "w_precision_schedule": schedule,
        "forward_w_precision": forward,
        "reverse_w_precision": reverse if forces else "n/a",
        "preconvergence": kwargs.get("preconvergence", "auto"),
    }


def _rounding(kwargs: dict[str, Any]) -> float:
    """Single-pass FP32 W values differ from split/FP64 ones by a rounding-sized amount."""
    return 1e-7 if kwargs.get("triples_fp32_w") else 0.0


_NATIVE_OWNERS = {
    "rhf_tplus": "spatial_rhf_triples_v2",
    "uhf_tplus": "spin_blocked_uhf_triples_v2",
    "conventional": "molecular_triples_on_reference",
}


@pytest.fixture
def native_triples(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the native triples owners and record every contraction they receive."""
    calls: list[dict[str, Any]] = []
    weighted: dict[str, float] = {}

    def spatial(request: FocalRequest, reference: FocalReference, **kwargs: Any) -> Any:
        calls.append(kwargs)
        occupied = np.full(reference.occupied[0], -0.002 + _rounding(kwargs))
        weighted["energy"] = float(np.dot(kwargs["occupied_weights"], occupied))
        return SimpleNamespace(
            energy=weighted["energy"],
            occupied=occupied,
            forces=np.full((len(request.atoms), 3), 0.01 + _rounding(kwargs)),
            telemetry=_native_telemetry(kwargs),
        )

    def spin_blocked(
        request: FocalRequest, reference: FocalReference, **kwargs: Any
    ) -> Any:
        calls.append(kwargs)
        alpha = np.full(reference.occupied[0], -0.002 + _rounding(kwargs))
        beta = np.full(reference.occupied[1], -0.001)
        wa, wb = kwargs["occupied_weights"]
        weighted["energy"] = float(np.dot(wa, alpha) + np.dot(wb, beta))
        return SimpleNamespace(
            energy=weighted["energy"],
            occupied_alpha=alpha,
            occupied_beta=beta,
            forces=np.full((len(request.atoms), 3), 0.01 + _rounding(kwargs)),
            telemetry=_native_telemetry(kwargs),
        )

    def conventional(
        request: FocalRequest, reference: FocalReference, **kwargs: Any
    ) -> tuple[float, np.ndarray | None]:
        calls.append(kwargs)
        if kwargs.get("telemetry") is not None:
            kwargs["telemetry"].update(_native_telemetry(kwargs))
        shift = _rounding(kwargs)
        forces = (
            np.full((len(request.atoms), 3), 0.01 + shift) if kwargs["forces"] else None
        )
        return -0.01 + shift, forces

    def tplus_from_occupied(
        self: MolecularFocalProvider,
        request: FocalRequest,
        low: FocalLowOrderEnergy,
        *_: Any,
        **__: Any,
    ) -> SimpleNamespace:
        assert low.occupied_mp2_f12 is not None
        return SimpleNamespace(
            energy=weighted["energy"],
            mp2=low.occupied_mp2_f12.mp2,
            f12_increment=low.occupied_mp2_f12.f12_increment,
        )

    def tplus_scaling_forces(
        self: MolecularFocalProvider, request: FocalRequest, *_: Any, **__: Any
    ) -> np.ndarray:
        return np.zeros((len(request.atoms), 3))

    monkeypatch.setattr(kernels, "spatial_rhf_triples_v2", spatial)
    monkeypatch.setattr(kernels, "spin_blocked_uhf_triples_v2", spin_blocked)
    monkeypatch.setattr(kernels, "molecular_triples_on_reference", conventional)
    monkeypatch.setattr(
        MolecularFocalProvider, "tplus_from_occupied", tplus_from_occupied
    )
    monkeypatch.setattr(
        MolecularFocalProvider, "tplus_scaling_forces", tplus_scaling_forces
    )
    return calls


def _force_entry(
    entry: str,
    precision: str,
    *,
    backend: str = "cuda",
    provider: MolecularFocalProvider | None = None,
    device: CudaDeviceIdentity | None = _L20,
    **diagnostic: bool,
) -> tuple[float, np.ndarray, dict[str, object]]:
    """Evaluate one analytic triples force entry; return energy, forces and telemetry."""
    if provider is None:
        provider = _provider(backend, precision, device)
    if entry == "conventional":
        request = _request("rhf", "triples", backend)
        reference = _reference(request, (5,))
        telemetry: dict[str, object] = {}
        energy, forces = provider.triples_energy_and_forces(
            request, reference=reference, telemetry=telemetry, **diagnostic
        )
        return energy, forces, telemetry
    root_id, occupied = ("rhf", (5, 5)) if entry == "rhf_tplus" else ("uhf", (5, 4))
    request = _request(root_id, "triples_plus", backend)
    reference = _reference(request, occupied[:1] if root_id == "rhf" else occupied)
    method = (
        provider.restricted_tplus_energy_and_forces_on_reference
        if root_id == "rhf"
        else provider.unrestricted_tplus_energy_and_forces_on_reference
    )
    _, evaluation, returned = method(
        request,
        low_order=_low_order(reference, occupied),
        expected_cabs_rank=2,
        **diagnostic,
    )
    assert evaluation.forces is not None
    assert evaluation.telemetry == returned
    return evaluation.energy, evaluation.forces, dict(evaluation.telemetry)


def _composed(
    components: tuple[FocalEvaluation, ...], weights: tuple[float, ...]
) -> FocalRunResult:
    """Compose weighted components the way focal composition does."""
    return FocalRunResult(
        math.fsum(c.term.energy * w for c, w in zip(components, weights, strict=True)),
        np.sum(
            [c.term.forces * w for c, w in zip(components, weights, strict=True)],
            axis=0,
        ),
        tuple(
            (c.term.role, c.term.basis_id, c.term.cardinal, w)
            for c, w in zip(components, weights, strict=True)
        ),
        components,
        valence_correlation_space="all-active",
        frozen_core_definition_version=2,
    )


def _evaluation(
    role: str, energy: float, forces: np.ndarray, telemetry: dict[str, object]
) -> FocalEvaluation:
    term = FocalTerm(
        "surface",
        "series",
        f"basis-{role}",
        2,
        role,
        energy,
        np.asarray(forces, dtype=np.float64),
    )
    return FocalEvaluation(term, "cuda", True, telemetry=dict(telemetry))


def _valence_with_triples(
    entry: str, energy: float, forces: np.ndarray, telemetry: dict[str, object]
) -> FocalRunResult:
    """A composed valence result whose triples component is the given evaluation."""
    role = "triples" if entry == "conventional" else "triples_plus"
    low_order = _evaluation("hf_cabs", -76.0, np.full(forces.shape, 0.125), {})
    return _composed(
        (low_order, _evaluation(role, energy, forces, telemetry)), (1.0, 1.0)
    )


def _assemble(valence: FocalRunResult | FocalMolecularResult) -> Any:
    return assemble_production_focal_energy_and_forces(
        valence=valence,
        valence_correlation_space="all-active",
        require_production_defaults=False,
    )


@pytest.mark.parametrize("entry", _ENTRIES)
def test_l20_runs_requested_single_pass_forces_for_production(
    entry: str, native_triples: list[dict[str, Any]]
) -> None:
    energy, forces, telemetry = _force_entry(entry, "fp32_single")
    assert len(native_triples) == 1
    assert native_triples[0]["triples_fp32_w"] is True
    assert native_triples[0]["forces"] is True
    assert telemetry["w_precision_schedule"] == "fp32_single"
    assert telemetry["reverse_w_precision"] != "n/a"
    assert (telemetry["device_name"], telemetry["device_architecture"]) == (
        "NVIDIA L20",
        "sm_89",
    )
    assert telemetry["preconvergence"] == "auto"
    assert "production_eligible" not in telemetry
    valence = _valence_with_triples(entry, energy, forces, telemetry)
    validate_production_focal_recipe(
        valence_correlation_space="all-active",
        require_production_defaults=False,
        components=valence.components,
    )
    assembled = _assemble(valence)
    np.testing.assert_array_equal(assembled.total_forces_Eh_per_bohr, forces + 0.125)
    assert assembled.total_energy_Eh == math.fsum((-76.0, energy))


@_UNQUALIFIED
@pytest.mark.parametrize("entry", _ENTRIES)
def test_single_pass_forces_on_other_devices_are_refused_before_native_work(
    entry: str, device: CudaDeviceIdentity, native_triples: list[dict[str, Any]]
) -> None:
    with pytest.raises(TriplesWPrecisionError, match="only on an NVIDIA L20") as error:
        _force_entry(entry, "fp32_single", device=device)
    assert isinstance(error.value, ValueError)
    assert f"{device.name!r} ({device.architecture})" in str(error.value)
    assert native_triples == []


@pytest.mark.parametrize("entry", _HOST_ENTRIES)
def test_single_pass_forces_on_the_host_backend_are_refused_before_native_work(
    entry: str, native_triples: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        policy, "device_identity", Mock(side_effect=AssertionError("CUDA device query"))
    )
    with pytest.raises(TriplesWPrecisionError, match="host CPU backend"):
        _force_entry(entry, "fp32_single", backend="cpu_reference")
    assert native_triples == []


def test_force_admission_queries_cuda_ordinal_zero_once_by_default(
    native_triples: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = Mock(return_value=_L20)
    monkeypatch.setattr(policy, "device_identity", probe)
    _, _, telemetry = _force_entry(
        "rhf_tplus", "fp32_single", provider=_provider("cuda", "fp32_single", None)
    )
    probe.assert_called_once_with()
    assert telemetry["device_name"] == "NVIDIA L20"
    probe.return_value = _A100
    with pytest.raises(TriplesWPrecisionError, match="A100"):
        _force_entry(
            "rhf_tplus", "fp32_single", provider=_provider("cuda", "fp32_single", None)
        )
    assert len(native_triples) == 1


def test_a_failed_device_query_stops_forces_before_native_work(
    native_triples: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        policy, "device_identity", Mock(side_effect=RuntimeError("cuDeviceGet failed"))
    )
    for precision in ("fp64", "fp32_single"):
        with pytest.raises(RuntimeError, match="cuDeviceGet"):
            _force_entry(
                "conventional", precision, provider=_provider("cuda", precision, None)
            )
    assert native_triples == []
    with pytest.raises(TypeError, match="CudaDeviceIdentity"):
        policy.admit_triples_w_forces(
            "cuda",
            triples_fp32_w=True,
            triples_w_schedule=None,
            preconvergence="auto",
            cuda_device=lambda: "NVIDIA L20",  # type: ignore[arg-type,return-value]
            diagnostic_fp32_single_w_forces=False,
        )


@pytest.mark.parametrize("device", [_L20, _A100], ids=["l20", "a100"])
@pytest.mark.parametrize("precision", ["fp64", "fp32_split3"])
@pytest.mark.parametrize("entry", _ENTRIES)
def test_default_and_split_schedules_run_on_any_device_with_provenance(
    entry: str,
    precision: str,
    device: CudaDeviceIdentity,
    native_triples: list[dict[str, Any]],
) -> None:
    energy, forces, telemetry = _force_entry(entry, precision, device=device)
    assert not native_triples[0].get("triples_fp32_w", False)
    assert telemetry["w_precision_schedule"] == precision
    assert (telemetry["device_name"], telemetry["device_architecture"]) == (
        device.name,
        device.architecture,
    )
    assert telemetry["preconvergence"] == "auto"
    assert "production_eligible" not in telemetry
    assembled = _assemble(_valence_with_triples(entry, energy, forces, telemetry))
    np.testing.assert_array_equal(assembled.total_forces_Eh_per_bohr, forces + 0.125)


@pytest.mark.parametrize("preconvergence", ["fp64", "mixed"])
@pytest.mark.parametrize("entry", _ENTRIES)
def test_forces_record_the_requested_preconvergence(
    entry: str, preconvergence: str, native_triples: list[dict[str, Any]]
) -> None:
    provider = _provider("cuda", "fp32_single", _L20, preconvergence=preconvergence)
    _, _, telemetry = _force_entry(entry, "fp32_single", provider=provider)
    assert native_triples[0]["preconvergence"] == preconvergence
    assert telemetry["preconvergence"] == preconvergence


def test_admission_records_provenance_of_an_empty_triples_space() -> None:
    admission = policy.admit_triples_w_forces(
        "cuda",
        triples_fp32_w=True,
        triples_w_schedule=None,
        preconvergence="auto",
        cuda_device=lambda: _L20,
        diagnostic_fp32_single_w_forces=False,
    )
    telemetry: dict[str, object] = {
        "w_precision_schedule": "n/a",
        "reverse_w_precision": "n/a",
    }
    admission.record(telemetry)
    assert telemetry == {
        "w_precision_schedule": "n/a",
        "reverse_w_precision": "n/a",
        "preconvergence": "auto",
        "device_name": "NVIDIA L20",
        "device_architecture": "sm_89",
    }


@pytest.mark.parametrize("executed", ["fp64", "fp32_split3", "unreported"])
@pytest.mark.parametrize("entry", _ENTRIES)
def test_requested_single_pass_forces_refuse_another_executed_schedule(
    entry: str,
    executed: str,
    native_triples: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _NATIVE_OWNERS[entry]
    native = getattr(kernels, owner)

    def fallback(*args: Any, **kwargs: Any) -> Any:
        result = native(*args, **kwargs)
        telemetry = kwargs["telemetry"] if entry == "conventional" else result.telemetry
        telemetry["w_precision_schedule"] = executed
        return result

    monkeypatch.setattr(kernels, owner, fallback)
    with pytest.raises(
        TriplesWPrecisionError, match=f"executed w_precision_schedule={executed!r}"
    ):
        _force_entry(entry, "fp32_single")
    assert len(native_triples) == 1


@pytest.mark.parametrize("entry", ("rhf_tplus", "conventional"))
def test_single_pass_production_refuses_the_bounded_rhf_fallback(
    entry: str,
    native_triples: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _NATIVE_OWNERS[entry]
    native = getattr(kernels, owner)

    def bounded(*args: Any, **kwargs: Any) -> Any:
        result = native(*args, **kwargs)
        telemetry = kwargs["telemetry"] if entry == "conventional" else result.telemetry
        # One forward FP32 count without reverse FP32 from the bounded owner.
        slots = np.array(
            [0, 0, 2, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 2, 0, 4096], dtype=float
        )
        telemetry.update(
            kernels._triples_execution_telemetry(
                slots,
                forces=True,
                fp32_requested=True,
                spatial_cuda=True,
                cpu=False,
            )
        )
        return result

    monkeypatch.setattr(kernels, owner, bounded)
    with pytest.raises(TriplesWPrecisionError, match="used a resource fallback"):
        _force_entry(entry, "fp32_single")
    _, forces, telemetry = _force_entry(
        entry, "fp32_single", diagnostic_fp32_single_w_forces=True
    )
    assert np.isfinite(forces).all()
    assert telemetry["w_precision_schedule"] == "fp32_single"
    assert telemetry["forward_w_precision"] == "fp32"
    assert telemetry["reverse_w_precision"] == "fp64"
    assert telemetry["triples_fp32_w_resource_fallback"] is True
    assert telemetry["production_eligible"] is False


@pytest.fixture
def water_singlepoint(
    native_triples: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Run RHF water `singlepoint` forces through stubbed low-order and doubles stages."""

    def low_energy(
        self: MolecularFocalProvider, request: FocalRequest, **kwargs: Any
    ) -> FocalLowOrderEnergy:
        reference = kwargs.get("reference") or _reference(request, (5,))
        return _low_order(reference, (5, 5))

    def low_forces(
        self: MolecularFocalProvider, request: FocalRequest, **kwargs: Any
    ) -> FocalLowOrderForces:
        return FocalLowOrderForces(
            low_energy(self, request), np.zeros((3, 3)), np.full((3, 3), 0.1)
        )

    monkeypatch.setattr(MolecularFocalProvider, "low_order_energy", low_energy)
    monkeypatch.setattr(
        MolecularFocalProvider, "low_order_energy_and_forces", low_forces
    )
    monkeypatch.setattr(
        MolecularFocalProvider,
        "f12_doubles_energy_and_forces",
        lambda *_a, **_kw: (-0.02, np.full((3, 3), 0.02)),
    )

    def run(precision: str, triples: object, device: CudaDeviceIdentity = _L20) -> Any:
        return singlepoint(
            _WATER,
            FocalSeries(_BASIS),
            FocalSeries(_BASIS),
            None,
            provider=_provider("cuda", precision, device),
            root_id="rhf",
            correlation_space_id="all-active",
            f12_doubles=_BASIS,
            triples=triples,
            backend="cuda",
        )

    return run


_SINGLEPOINT_TRIPLES = pytest.mark.parametrize(
    ("triples", "entry"),
    [({"tplus": _BASIS}, "rhf_tplus"), (_BASIS, "conventional")],
    ids=["tplus", "conventional"],
)


@_SINGLEPOINT_TRIPLES
def test_singlepoint_single_pass_forces_follow_the_device(
    triples: object,
    entry: str,
    water_singlepoint: Any,
    native_triples: list[dict[str, Any]],
) -> None:
    result = water_singlepoint("fp32_single", triples)
    telemetry = result.components[-1].telemetry
    assert telemetry["w_precision_schedule"] == "fp32_single"
    assert telemetry["device_name"] == "NVIDIA L20"
    assert "production_eligible" not in telemetry
    assert len(native_triples) == 1
    _assemble(result)
    with pytest.raises(TriplesWPrecisionError, match="only on an NVIDIA L20"):
        water_singlepoint("fp32_single", triples, _A100)
    assert len(native_triples) == 1
    result = water_singlepoint("fp32_split3", triples, _A100)
    assert result.components[-1].telemetry["device_name"] == _A100.name
    assert len(native_triples) == 2


@pytest.mark.parametrize("precision", ["fp32_split3", "fp32_single"])
@_SINGLEPOINT_TRIPLES
def test_production_assembly_takes_the_composed_singlepoint_result(
    triples: object, entry: str, precision: str, water_singlepoint: Any
) -> None:
    result = water_singlepoint(precision, triples)
    assert isinstance(result, FocalMolecularResult)
    delta_sr = np.full(result.forces.shape, 0.001)
    assembled = assemble_production_focal_energy_and_forces(
        valence=result,
        valence_correlation_space="all-active",
        delta_sr_energy=-0.25,
        delta_sr_forces=delta_sr,
        delta_sr_correlation_space="all-active",
        delta_sr_frozen_core_definition_version=2,
        require_production_defaults=False,
    )
    assert assembled.valence_energy_Eh == result.energy
    np.testing.assert_array_equal(assembled.valence_forces_Eh_per_bohr, result.forces)
    assert assembled.total_energy_Eh == math.fsum((result.energy, -0.25))
    np.testing.assert_array_equal(
        assembled.total_forces_Eh_per_bohr, result.forces + delta_sr
    )


@_SINGLEPOINT_TRIPLES
def test_production_assembly_refuses_a_real_diagnostic_result_and_its_bare_values(
    triples: object, entry: str, water_singlepoint: Any
) -> None:
    result = water_singlepoint("fp32_split3", triples)
    energy, forces, telemetry = _force_entry(
        entry, "fp32_single", device=_A100, diagnostic_fp32_single_w_forces=True
    )
    assert telemetry["production_eligible"] is False
    production = result.components[-1]
    assert production.term.energy != energy
    diagnostic = replace(
        production,
        term=replace(
            production.term, energy=energy, forces=np.asarray(forces, dtype=np.float64)
        ),
        telemetry=telemetry,
    )
    weights = tuple(w for *_, w in result.weights)
    composed = _composed((*result.components[:-1], diagnostic), weights)
    with pytest.raises(
        ValueError, match="diagnostic focal result 'fp32_single_w_forces'"
    ):
        _assemble(
            replace(
                result,
                energy=composed.energy,
                forces=composed.forces,
                components=composed.components,
            )
        )
    with pytest.raises(TypeError, match="valence_energy"):
        assemble_production_focal_energy_and_forces(  # type: ignore[call-arg]
            valence_energy=composed.energy,
            valence_forces=composed.forces,
            valence_components=(),
        )
    with pytest.raises(ValueError, match="requires the valence result components"):
        assemble_production_focal_energy_and_forces(
            valence=replace(
                result,
                energy=composed.energy,
                forces=composed.forces,
                weights=(),
                components=(),
            ),
        )
    with pytest.raises(ValueError, match="do not equal the weighted sum"):
        _assemble(replace(result, energy=composed.energy, forces=composed.forces))


@pytest.mark.parametrize("device", [_L20, _A100], ids=["l20", "a100"])
@pytest.mark.parametrize("entry", _ENTRIES)
def test_explicit_diagnostic_runs_single_pass_forces_and_tags_them(
    entry: str, device: CudaDeviceIdentity, native_triples: list[dict[str, Any]]
) -> None:
    energy, forces, telemetry = _force_entry(
        entry, "fp32_single", device=device, diagnostic_fp32_single_w_forces=True
    )
    assert len(native_triples) == 1
    assert native_triples[0]["triples_fp32_w"] is True
    assert native_triples[0]["forces"] is True
    assert np.isfinite(energy) and np.isfinite(forces).all()
    assert telemetry["w_precision_schedule"] == "fp32_single"
    assert telemetry["device_name"] == device.name
    assert telemetry["production_eligible"] is False
    assert telemetry["diagnostic_mode"] == "fp32_single_w_forces"


@pytest.mark.parametrize("entry", _HOST_ENTRIES)
def test_explicit_diagnostic_runs_single_pass_forces_on_the_host_backend(
    entry: str, native_triples: list[dict[str, Any]]
) -> None:
    _, _, telemetry = _force_entry(
        entry,
        "fp32_single",
        backend="cpu_reference",
        diagnostic_fp32_single_w_forces=True,
    )
    assert len(native_triples) == 1
    assert (telemetry["device_name"], telemetry["device_architecture"]) == (
        "cpu",
        "n/a",
    )
    assert telemetry["production_eligible"] is False


@pytest.mark.parametrize("precision", ["fp64", "fp32_split3"])
@pytest.mark.parametrize("entry", _ENTRIES)
def test_diagnostic_flag_requires_the_single_pass_schedule(
    entry: str, precision: str, native_triples: list[dict[str, Any]]
) -> None:
    with pytest.raises(ValueError, match="requires the single-pass FP32 W schedule"):
        _force_entry(entry, precision, diagnostic_fp32_single_w_forces=True)
    with pytest.raises(TypeError, match="must be a boolean"):
        _force_entry(entry, precision, diagnostic_fp32_single_w_forces=1)  # type: ignore[arg-type]
    assert native_triples == []


@pytest.mark.parametrize("diagnostic", [False, True])
def test_single_pass_conventional_forces_must_record_their_provenance(
    diagnostic: bool, native_triples: list[dict[str, Any]]
) -> None:
    request = _request("rhf", "triples", "cuda")
    with pytest.raises(ValueError, match="must record its provenance"):
        _provider("cuda", "fp32_single").triples_energy_and_forces(
            request,
            reference=_reference(request, (5,)),
            diagnostic_fp32_single_w_forces=diagnostic,
        )
    assert native_triples == []


@pytest.mark.parametrize("device", [_L20, _A100], ids=["l20", "a100"])
@pytest.mark.parametrize("entry", _ENTRIES)
def test_production_assembly_refuses_the_diagnostic_result(
    entry: str, device: CudaDeviceIdentity, native_triples: list[dict[str, Any]]
) -> None:
    energy, forces, telemetry = _force_entry(
        entry, "fp32_single", device=device, diagnostic_fp32_single_w_forces=True
    )
    valence = _valence_with_triples(entry, energy, forces, telemetry)
    with pytest.raises(
        ValueError, match="diagnostic focal result 'fp32_single_w_forces'"
    ):
        validate_production_focal_recipe(components=valence.components)
    with pytest.raises(
        ValueError, match="diagnostic focal result 'fp32_single_w_forces'"
    ):
        assemble_production_focal_energy_and_forces(valence=valence)
    production = _valence_with_triples(entry, -0.01, np.zeros_like(forces), {})
    with pytest.raises(ValueError, match="do not equal the weighted sum"):
        assemble_production_focal_energy_and_forces(
            valence=replace(production, energy=valence.energy, forces=valence.forces)
        )


@pytest.mark.parametrize(
    "device",
    [
        {},
        {"device_name": _A100.name, "device_architecture": _A100.architecture},
        {"device_name": _L40S.name, "device_architecture": _L40S.architecture},
        {"device_name": _L20.name, "device_architecture": "sm_80"},
        {"device_name": "cpu", "device_architecture": "n/a"},
        {"device_name": None, "device_architecture": None},
    ],
    ids=["unrecorded", "a100", "l40s", "mismatched-architecture", "host", "null"],
)
@pytest.mark.parametrize(
    "reverse", ["fp32", "fp64", "n/a", pytest.param(None, id="missing")]
)
def test_production_assembly_refuses_single_pass_forces_without_an_l20_record(
    device: dict[str, object], reverse: str | None
) -> None:
    forces = np.zeros((3, 3))
    telemetry = {
        "w_precision_schedule": "fp32_single",
        **device,
    }
    if reverse is not None:
        telemetry["reverse_w_precision"] = reverse
    valence = _valence_with_triples("conventional", -1.0, forces, telemetry)
    with pytest.raises(ValueError, match="only from an NVIDIA L20"):
        validate_production_focal_recipe(components=valence.components)
    with pytest.raises(ValueError, match="only from an NVIDIA L20"):
        _assemble(valence)
    recorded = {
        **telemetry,
        "device_name": "NVIDIA L20",
        "device_architecture": "sm_89",
    }
    assert _assemble(
        _valence_with_triples("conventional", -1.0, forces, recorded)
    ).total_energy_Eh == math.fsum((-76.0, -1.0))


@pytest.mark.parametrize("executed", ["fp32_single", "fp64"])
def test_production_assembly_refuses_an_l20_force_result_with_a_resource_fallback(
    executed: str,
) -> None:
    telemetry = {
        "requested_w_precision_schedule": "fp32_single",
        "w_precision_schedule": executed,
        "triples_fp32_w_resource_fallback": True,
        "device_name": _L20.name,
        "device_architecture": _L20.architecture,
    }
    valence = _valence_with_triples("conventional", -1.0, np.zeros((3, 3)), telemetry)
    with pytest.raises(
        TriplesWPrecisionError, match="resource fallback|matching W telemetry"
    ):
        validate_production_focal_recipe(components=valence.components)
    with pytest.raises(
        TriplesWPrecisionError, match="resource fallback|matching W telemetry"
    ):
        _assemble(valence)


def test_production_assembly_refuses_a_cpu_force_result_with_an_l20_record() -> None:
    telemetry = {
        "w_precision_schedule": "fp32_single",
        "device_name": _L20.name,
        "device_architecture": _L20.architecture,
    }
    valence = _valence_with_triples("conventional", -1.0, np.zeros((3, 3)), telemetry)
    hf, triples = valence.components
    valence = replace(
        valence, components=(hf, replace(triples, executed_backend="cpu_reference"))
    )
    with pytest.raises(ValueError, match="backend='cpu_reference'"):
        _assemble(valence)


def test_production_assembly_refuses_unreported_legacy_single_pass_forces() -> None:
    telemetry = {"triples_fp32_w": True, "w_precision_schedule": "unreported"}
    valence = _valence_with_triples("conventional", -1.0, np.zeros((3, 3)), telemetry)
    with pytest.raises(
        TriplesWPrecisionError, match="w_precision_schedule='unreported'"
    ):
        _assemble(valence)


@pytest.mark.parametrize("device", [None, _A100], ids=["unrecorded", "a100"])
def test_production_assembly_admits_single_pass_energy(
    device: CudaDeviceIdentity | None,
) -> None:
    request = _request("rhf", "triples", "cuda")
    telemetry = {"w_precision_schedule": "fp32_single", "reverse_w_precision": "n/a"}
    if device is not None:
        telemetry.update(
            device_name=device.name, device_architecture=device.architecture
        )
    energy_only = FocalEnergyComponent(
        request,
        -1.0,
        1.0,
        "cuda",
        True,
        telemetry=telemetry,
    )
    assembled = assemble_production_focal_energy_and_forces(
        valence=FocalEnergyResult(
            -1.0, (energy_only,), {}, {}, valence_correlation_space="all-active",
            frozen_core_definition_version=2,
        ),
        valence_correlation_space="all-active",
        require_production_defaults=False,
    )
    assert assembled.total_energy_Eh == -1.0
    assert assembled.total_forces_Eh_per_bohr is None
    with pytest.raises(TypeError, match="telemetry mapping"):
        validate_production_focal_recipe(components=(object(),))


@pytest.mark.parametrize("entry", _ENTRIES)
def test_a100_energy_entry_keeps_the_callers_single_pass_choice(
    entry: str,
    native_triples: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider("cuda", "fp32_single", _A100)
    telemetry: dict[str, object] = {}
    if entry == "conventional":
        request = _request("rhf", "triples", "cuda")
        energy = provider.triples_energy(
            request, reference=_reference(request, (5,)), telemetry=telemetry
        )
    else:
        root_id, occupied = ("rhf", (5, 5)) if entry == "rhf_tplus" else ("uhf", (5, 4))
        request = _request(root_id, "triples_plus", "cuda")
        reference = _reference(request, occupied[:1] if root_id == "rhf" else occupied)

        def forward(
            request: FocalRequest, reference: FocalReference, **options: Any
        ) -> tuple[float, SimpleNamespace]:
            native_triples.append({"forces": False, **options})
            return -0.01, SimpleNamespace(
                reference=reference, telemetry=_native_telemetry(options)
            )

        monkeypatch.setattr(kernels, "molecular_occupied_triples_on_reference", forward)
        monkeypatch.setattr(
            MolecularFocalProvider,
            "tplus_from_occupied",
            lambda *_args: SimpleNamespace(energy=-0.02),
        )
        evaluate = (
            provider.restricted_tplus_energy_on_reference
            if root_id == "rhf"
            else provider.unrestricted_tplus_energy_on_reference
        )
        energy, evaluated = evaluate(request, low_order=_low_order(reference, occupied))
        assert evaluated.forces is None
        telemetry = evaluated.telemetry
    assert math.isfinite(energy)
    assert native_triples
    assert all(
        call["triples_fp32_w"] is True and call["forces"] is False
        for call in native_triples
    )
    assert telemetry["w_precision_schedule"] == "fp32_single"
    assert telemetry["reverse_w_precision"] == "n/a"


def test_production_assembly_requires_a_consistent_typed_valence_result() -> None:
    with pytest.raises(TypeError, match="valence"):
        assemble_production_focal_energy_and_forces()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="FocalRunResult"):
        assemble_production_focal_energy_and_forces(valence=-1.0)  # type: ignore[arg-type]
    component = _evaluation("hf_cabs", -1.0, np.zeros((3, 3)), {})
    with pytest.raises(TypeError, match="FocalRunResult"):
        assemble_production_focal_energy_and_forces(
            valence=SimpleNamespace(energy=-1.0, components=(component,))  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="FocalEvaluation"):
        assemble_production_focal_energy_and_forces(
            valence=FocalRunResult(
                -1.0,
                np.zeros((3, 3)),
                (("hf_cabs", "b", 2, 1.0),),
                (SimpleNamespace(telemetry={}),),  # type: ignore[arg-type]
            )
        )
    valence = _composed((component,), (1.0,))
    with pytest.raises(ValueError, match="frozen-core definition version 2"):
        assemble_production_focal_energy_and_forces(
            valence=replace(valence, frozen_core_definition_version=1),
            require_production_defaults=False,
        )
    with pytest.raises(ValueError, match="describe different terms"):
        assemble_production_focal_energy_and_forces(
            valence=replace(
                valence, weights=(("mp2_f12_correlation", "basis-hf_cabs", 2, 1.0),)
            )
        )
    with pytest.raises(ValueError, match="must be converged"):
        assemble_production_focal_energy_and_forces(
            valence=replace(valence, components=(replace(component, converged=False),))
        )
    request = _request("rhf", "triples", "cuda")
    part = FocalEnergyComponent(request, -1.0, 0.5, "cuda", True)
    with pytest.raises(ValueError, match="does not equal the weighted sum"):
        assemble_production_focal_energy_and_forces(
            valence=FocalEnergyResult(-1.0, (part,), {}, {}),
            valence_correlation_space="all-active",
            require_production_defaults=False,
        )
    with pytest.raises(
        ValueError, match="require a valence FocalRunResult with forces"
    ):
        assemble_production_focal_energy_and_forces(
            valence=FocalEnergyResult(
                -0.5, (part,), {}, {}, valence_correlation_space="all-active",
                frozen_core_definition_version=2,
            ),
            valence_correlation_space="all-active",
            delta_cv_energy=-0.1,
            delta_cv_forces=np.zeros((3, 3)),
            delta_cv_valence_space="all-active",
            delta_cv_frozen_core_definition_version=2,
            require_production_defaults=False,
        )
