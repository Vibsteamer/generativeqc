"""Backend-neutral dependency proof for reuse between iterative evaluations.

This is a schedule analysis over existing immutable TensorIR, not a cache owner
or a new solver. Callers must hold every declared invariant input immutable for
one execution epoch. Changing reference/geometry/basis/parameters ends that epoch;
matching shapes or pointer addresses are never a validity proof. Native owners
remain responsible for retained-storage admission, failure and publication.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from generativeqc_compiler.common.liveness import EffectKind
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.resources import checked_bytes

from .ir import Node, input_tensor
from .optimize import _tensor_effect
from .program import Program, node_hashes


@dataclass(frozen=True)
class IterationReusePlan:
    """Proven invariant operations in the original deterministic graph order.

    Every retained intermediate requires distinct live storage throughout replay.
    ``retained_bytes`` is their logical numeric capacity, not a complete endpoint
    peak: native planners must also charge inputs, dynamic scratch, outputs and
    preparation overlap. Pure TensorIR views are logical values; a backend that
    aliases their storage must retain the entire underlying allocation lease.
    """

    invariant_nodes: tuple[Node, ...]
    dynamic_nodes: tuple[Node, ...]
    input_dependencies: tuple[tuple[Node, tuple[str, ...]], ...]
    invariant_inputs: tuple[str, ...]
    retained_bytes: int
    identity: str


@dataclass(frozen=True)
class IterationReusePrograms:
    """Executable preparation and replay for one immutable set of inputs."""

    prepare: Program
    replay: Program
    retained_bytes: int


def materialize_iteration_frontier(
    program: Program, plan: IterationReusePlan, *, max_retained_bytes: int
) -> IterationReusePrograms:
    """Bind dependency-proven retained values as explicit replay inputs.

    Supports the plain arithmetic graph used by stationary CC response. The
    caller runs preparation once, retains its owned outputs, and charges
    simultaneous execution storage separately from the logical frontier.
    """
    if type(max_retained_bytes) is not int or max_retained_bytes < 0:
        raise ValueError("frontier storage budget must be a nonnegative integer")
    if "precision_execution" in program.provenance:
        raise ValueError("frontier materialization requires a plain arithmetic graph")
    frontier = invariant_frontier(program, plan)
    if not frontier:
        raise ValueError("iteration program has no retained frontier values")
    if plan != analyze_iteration_reuse(program, invariant_inputs=plan.invariant_inputs):
        raise ValueError("frontier materialization requires this graph's default proof")
    retained = checked_bytes(
        sum(node.spec.size * node.spec.itemsize for node in frontier),
        "iteration frontier retained bytes",
    )
    if retained > max_retained_bytes:
        raise MemoryError("iteration frontier retained storage budget exceeded")
    name_by_node = {
        node: f"iteration_frontier_{number}" for number, node in enumerate(frontier)
    }
    if set(name_by_node.values()) & {
        node.attrs["name"] for node in program.live_nodes if node.op == "input"
    }:
        raise ValueError("frontier replay input names collide with the original graph")
    provenance = {"iteration_reuse": plan.identity}
    prepared = Program(
        {name: node for node, name in name_by_node.items()},
        provenance={**provenance, "phase": "prepare"},
    )
    replacements = {
        node: input_tensor(name, replace(node.spec, role="input"))
        for node, name in name_by_node.items()
    }
    for node in program.live_nodes:
        if node.op == "input":
            replacements[node] = node
    for node in plan.dynamic_nodes:
        inputs = tuple(replacements[child] for child in node.inputs)
        replacements[node] = Node(node.op, inputs, node.spec, node.attributes)
    replay = Program(
        {name: replacements[node] for name, node in program.outputs.items()},
        provenance={**provenance, "phase": "replay"},
    )
    return IterationReusePrograms(prepared, replay, retained)


def invariant_frontier(program: Program, plan: IterationReusePlan) -> tuple[Node, ...]:
    """Return immutable values read by dynamic operations or published outputs.

    All invariant operations still execute during preparation. Only values that
    cross its boundary need persistent storage; interior preparation temporaries
    can share the caller's existing scratch. This is a liveness cut over the
    existing purity/dependency proof, not a new validity or mathematical proof.
    """
    operations = {node for node in program.live_nodes if node.op != "input"}
    invariant = set(plan.invariant_nodes)
    dynamic = set(plan.dynamic_nodes)
    if invariant & dynamic or invariant | dynamic != operations:
        raise ValueError(
            "invariant frontier requires this program's complete reuse plan"
        )
    readers = set(program.outputs.values()) | {
        source for node in plan.dynamic_nodes for source in node.inputs
    }
    return tuple(node for node in plan.invariant_nodes if node in readers)


def analyze_iteration_reuse(
    program: Program,
    *,
    invariant_inputs: Sequence[str],
    effects: Mapping[str, EffectKind] | None = None,
) -> IterationReusePlan:
    """Classify transitive dependencies before any CPU/CUDA lowering.

    No shape, method name, numerical value or backend is used to infer reuse.
    Effects may conservatively downgrade an existing primitive's PURE proof;
    unknown operations can never be promoted to PURE by this argument. A node
    depending on an opaque/effectful operation stays dynamic even when all of its
    named inputs are invariant. The analysis neither reorders nor drops nodes.
    """
    if not isinstance(program, Program):
        raise TypeError("iteration reuse requires a TensorIR Program")
    if isinstance(invariant_inputs, (str, bytes)) or not isinstance(
        invariant_inputs, Sequence
    ):
        raise TypeError("invariant inputs must be a sequence of names")
    names = tuple(invariant_inputs)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("invariant input names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("duplicate invariant input")
    names = tuple(sorted(names))
    if effects is None:
        effects = {}
    if not isinstance(effects, Mapping) or any(
        not isinstance(op, str) or not isinstance(effect, EffectKind)
        for op, effect in effects.items()
    ):
        raise TypeError("iteration effects must map operation names to EffectKind")
    nodes = program.live_nodes
    available_inputs = {node.attrs["name"] for node in nodes if node.op == "input"}
    if set(names) - available_inputs:
        raise ValueError("invariant input is absent from the live program")
    declared = set(names)
    dependencies: dict[Node, frozenset[str]] = {}
    reusable: dict[Node, bool] = {}
    invariant, dynamic = [], []
    hashes = node_hashes(nodes)
    effective = []
    retained = 0
    for node in nodes:
        proof = _tensor_effect(node)
        effect = effects.get(node.op, proof)
        pure = proof is EffectKind.PURE and effect is EffectKind.PURE
        effective.append((hashes[node], pure))
        if node.op == "input":
            dependencies[node] = frozenset((node.attrs["name"],))
            reusable[node] = pure and node.attrs["name"] in declared
            continue
        dependencies[node] = frozenset().union(
            *(dependencies[source] for source in node.inputs)
        )
        reusable[node] = pure and all(reusable[source] for source in node.inputs)
        if reusable[node]:
            invariant.append(node)
            retained = checked_bytes(
                retained + node.spec.size * node.spec.itemsize,
                "iteration invariant retained bytes",
            )
        else:
            dynamic.append(node)
    identity = canonical_hash(
        {
            "schema": "generativeqc.tensor.iteration-reuse.v1",
            "program": program.logical_hash,
            "invariant_inputs": names,
            "pure_proofs": effective,
        }
    )
    return IterationReusePlan(
        tuple(invariant),
        tuple(dynamic),
        tuple((node, tuple(sorted(dependencies[node]))) for node in nodes),
        names,
        retained,
        identity,
    )
