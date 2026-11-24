"""Prepare a native ECP tensor program and project molecular ECP derivative weights."""
from __future__ import annotations

import os
from pathlib import Path

from generativeqc_compiler.tensor.ir import add, einsum, input_tensor, transpose
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec


def projected_inputs(integrals, weights, frame=None, motion=None):
    """Build only the arrays required by the AO, frozen-frame or moving-frame case."""
    natom, _, nao, _ = integrals.local_derivative.shape
    atom = Index("atom", IndexSpace("atom", "batch", natom))
    xyz = Index("xyz", IndexSpace("xyz", "batch", 3))
    aos = IndexSpace("ao", "ao", nao)
    p, q = Index("p", aos), Index("q", aos)
    feeds = {}

    def tensor(name, value, *indices):
        feeds[name] = value
        return input_tensor(name, TensorSpec(indices, role="input"))

    derivative = add(
        tensor("dlocal", integrals.local_derivative, atom, xyz, p, q),
        tensor("dnonlocal", integrals.nonlocal_derivative, atom, xyz, p, q),
    )
    if frame is None:
        ao_weight = tensor("weights", weights, p, q)
    else:
        orbitals = IndexSpace("projected", "orbital", frame.shape[1])
        i, j = Index("i", orbitals), Index("j", orbitals)
        weight = tensor("weights", weights, i, j)
        coeff = tensor("frame", frame, p, i)
        cw = einsum("pi,ij->pj", coeff, weight)
        ao_weight = einsum("pj,qj->pq", cw, coeff)
    gradient = einsum("axpq,pq->ax", derivative, ao_weight)
    if motion is not None:
        potential = add(tensor("local", integrals.local, p, q),
                        tensor("nonlocal_", integrals.nonlocal_, p, q))
        velocity = tensor("motion", motion, atom, xyz, p, i)
        vc = einsum("pq,qi->pi", potential, coeff)
        sym_weight = add(weight, transpose(weight, (1, 0)))
        frame_weight = einsum("pi,ij->pj", vc, sym_weight)
        gradient = add(gradient, einsum("axpi,pi->ax", velocity, frame_weight))
    program = Program({"gradient": gradient}, provenance={"operation": "projected ECP gradient"})
    return program, feeds


def prepare(program, *, device=0, cache=None):
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from generativeqc_compiler.tensor.cuda_plan import plan_cuda

    from ._cuda_device import device_architecture
    from ._cuda_runtime import preload_cuda_runtime_libraries
    from .profiles import find_nvcc

    target = cuda_target_info(device_architecture(device))
    plan = plan_cuda(program, target)
    nvcc = find_nvcc()
    if nvcc is None:
        raise RuntimeError("native CUDA ECP gradients require the NVCC toolchain")
    if cache is None:
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "generativeqc" / "ecp"
    preload_cuda_runtime_libraries()
    artifact = compile_cuda(plan, CudaCompilerAdapter(nvcc, target), cache)
    return PreparedCuda(plan, artifact, device=device)
