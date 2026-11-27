"""Canonical UCCSD with one host-controlled CPU/CUDA iteration policy.

The shared TensorIR graph owns all scientific contractions. This conventional
owner borrows a validated UHF snapshot and spin-capable integral provider,
materializes owned spin-resolved MO feeds, and maintains packed excitation-state
DIIS coordinates. CUDA transforms/residuals never fall back to CPU evaluation;
host Fock transforms, checkpoints, denominators and DIIS are explicit parts of
this contract.
"""

from __future__ import annotations

import math
import time
from hashlib import sha256
from types import MappingProxyType

import numpy as np
from generativeqc._ucc.native_tensor_cpu import NativeCCTensorExecutor
from generativeqc._ucc.native_tensor_cuda import CudaCCTensorExecutor
from generativeqc._ucc.uccsd_solver import (
    UCCSDIteration,
    UCCSDResult,
    _denominator_minimum,
    _denominators,
    _iteration_bytes,
    _PackedUCCSD,
    _representatives,
)
from generativeqc_compiler.cc.uccsd_coulomb_blocks import (
    antisymmetrized_integral_requests as _antisymmetrized_integral_requests,
)
from generativeqc_compiler.cc.uccsd_equations import build_program
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.resources import checked_bytes
from generativeqc_compiler.tensor.cpu import emit_cpu
from generativeqc_compiler.tensor.cuda_plan import plan_cuda

from tools.generativeqc_posthf.providers import ConventionalProvider
from tools.generativeqc_response.uhf import UHFReferenceSnapshot

from .solver import SolverOptions


def _integral_requests(name: str, snapshot: UHFReferenceSnapshot) -> tuple:
    domains, labels = name[1:].split("_")
    spins = tuple("alpha" if label == "a" else "beta" for label in labels)
    slots = tuple(
        tuple(range(snapshot.nocc(spin)))
        if domain == "o"
        else tuple(range(snapshot.nocc(spin), snapshot.nmo))
        for domain, spin in zip(domains, spins, strict=True)
    )
    return _antisymmetrized_integral_requests(slots, spins)


class PreparedUCCSD(_PackedUCCSD):
    """Own conventional MO feeds, with bounded packed host iteration state.

    Provider/snapshot identities and the complete provider cache transition are
    admitted before AO reads. Provider and solver numeric budgets are separate
    explicit bounds, as in existing correlated owners. Prepared feeds own host
    storage and survive provider clearing/closing. CUDA is explicitly staged:
    four-center integral transforms and CC contractions use CUDA, while Fock
    transforms, feeds, checkpoints and excitation-state DIIS remain on the host.
    No PySCF or CPU equation fallback.
    """

    def __init__(
        self,
        snapshot: UHFReferenceSnapshot,
        provider: ConventionalProvider,
        *,
        options: SolverOptions | None = None,
        executor: NativeCCTensorExecutor | CudaCCTensorExecutor | None = None,
    ) -> None:
        started = time.perf_counter()
        if (
            not isinstance(snapshot, UHFReferenceSnapshot)
            or snapshot.algorithm != "UHF"
        ):
            raise TypeError(
                "UCCSD requires a validated canonical UHF reference, not RHF/UKS"
            )
        if (
            not isinstance(provider, ConventionalProvider)
            or provider.snapshot.identity != snapshot.identity
        ):
            raise ValueError(
                "UCCSD requires an identity-matched conventional integral provider"
            )
        if options is None:
            options = SolverOptions()
        if not isinstance(options, SolverOptions):
            raise TypeError("UCCSD options must be SolverOptions")
        if executor is None:
            if provider.backend != "cpu":
                raise ValueError("CUDA UCCSD requires an explicit CUDA tensor executor")
            executor = NativeCCTensorExecutor(max_bytes=options.max_bytes)
        if not isinstance(executor, (NativeCCTensorExecutor, CudaCCTensorExecutor)):
            raise TypeError("UCCSD requires a native CPU or CUDA TensorIR executor")
        cuda = isinstance(executor, CudaCCTensorExecutor)
        if provider.backend != ("cuda" if cuda else "cpu"):
            raise ValueError(
                "UCCSD integral and tensor backends must match; no CPU fallback"
            )
        if isinstance(executor, CudaCCTensorExecutor) and (
            provider.cuda_artifact is None or provider.device_id != executor.device
        ):
            raise ValueError(
                "CUDA UCCSD requires a transform artifact and matching device ordinal"
            )
        self.reference, self.options, self.executor = snapshot, options, executor
        self.layout = UCCSDLayout(
            snapshot.nocc("alpha"),
            snapshot.nocc("beta"),
            snapshot.nmo - snapshot.nocc("alpha"),
            snapshot.nmo - snapshot.nocc("beta"),
        )
        self.minimum_absolute_denominator = _denominator_minimum(
            snapshot, self.layout, options
        )
        self.program = build_program(
            self.layout.oa, self.layout.ob, self.layout.va, self.layout.vb
        )
        inputs = {
            node.attrs["name"]: node.spec
            for node in self.program.live_nodes
            if node.op == "input"
        }
        conversions = {
            name: _integral_requests(name, snapshot)
            for name in sorted(inputs)
            if name.startswith("g")
        }
        requests = tuple(
            dict.fromkeys(
                block
                for contributions in conversions.values()
                for _, block, _ in contributions
            )
        )
        feed_bytes = sum(
            spec.size * 8 for name, spec in inputs.items() if not name.startswith("t")
        )
        raw_bytes = sum(math.prod(block.shape) * 8 for block in requests)
        largest = max(
            (spec.size * 8 for name, spec in inputs.items() if name.startswith("g")),
            default=0,
        )
        self.host_iteration_bytes = _iteration_bytes(self.layout, options, feed_bytes)
        host_preparation_bytes = checked_bytes(
            feed_bytes + raw_bytes + 4 * largest + 24 * snapshot.nmo**2,
            "UCCSD preparation storage",
        )
        if isinstance(executor, CudaCCTensorExecutor):
            self.execution_peak_bytes = plan_cuda(
                self.program, executor.compiler.target, max_bytes=executor.max_bytes
            ).peak_bytes
        else:
            _, resources = emit_cpu(
                self.program,
                max_bytes=executor.max_bytes,
                max_work=executor.max_work,
                max_nodes=executor.max_nodes,
            )
            self.execution_peak_bytes = resources["required_bytes"]
        self.logical_required_bytes = checked_bytes(
            max(
                host_preparation_bytes,
                self.host_iteration_bytes + self.execution_peak_bytes,
            ),
            "UCCSD numeric storage",
        )
        if self.logical_required_bytes > options.max_bytes:
            raise MemoryError(
                f"UCCSD solver budget needs {self.logical_required_bytes} numeric bytes before integral conversion"
            )
        self.denominators = _denominators(snapshot, self.layout, options.level_shift)
        feeds = {}
        for spin, suffix in (("alpha", "aa"), ("beta", "bb")):
            coefficients = getattr(snapshot, f"coefficients_{spin}")
            fock = coefficients.T @ getattr(snapshot, f"fock_{spin}") @ coefficients
            occupied = snapshot.nocc(spin)
            for domains, axes in (
                ("oo", (slice(0, occupied), slice(0, occupied))),
                ("ov", (slice(0, occupied), slice(occupied, None))),
                ("vv", (slice(occupied, None), slice(occupied, None))),
            ):
                feeds[f"f{domains}_{suffix}"] = immutable(fock[axes])
        with provider._lock:
            self.provider_peak_bytes = provider.admit(requests)
            previous_statistics = dict(provider.statistics)
            raw = {}
            for block in requests:
                result = provider.get(block)
                if (
                    result.reference_id != snapshot.identity
                    or result.hamiltonian_id != snapshot.hamiltonian_id
                ):
                    raise ValueError(
                        "UCCSD integral block identity does not match reference"
                    )
                raw[block] = immutable(result.to_host(), shape=block.shape)
            for name, contributions in conversions.items():
                value = np.zeros(inputs[name].shape)
                for coefficient, block, axes in contributions:
                    value += coefficient * raw[block].transpose(axes)
                feeds[name] = immutable(value)
            self.provider_statistics = MappingProxyType(
                {
                    name: (
                        self.provider_peak_bytes
                        if name == "peak_bytes"
                        else value - previous_statistics[name]
                    )
                    for name, value in provider.statistics.items()
                }
            )
        self.feeds = MappingProxyType(feeds)
        self.integral_hash = canonical_hash(
            {
                name: sha256(value.astype("<f8", copy=False).tobytes()).hexdigest()
                for name, value in feeds.items()
            }
        )
        self.integral_backend = provider.backend
        self.source_backend = provider.source.backend
        self.integral_source_id = provider.source.identity
        self.integral_request_count = len(requests)
        self._executor_configuration = self._execution_configuration()
        self._closed = False
        self.preparation_seconds = time.perf_counter() - started


__all__ = [
    "PreparedUCCSD",
    "UCCSDIteration",
    "UCCSDResult",
    "_PackedUCCSD",
    "_denominator_minimum",
    "_denominators",
    "_iteration_bytes",
    "_representatives",
]
