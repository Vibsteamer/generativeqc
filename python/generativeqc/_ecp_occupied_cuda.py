"""Native tensor program for the relaxed occupied scalar ECP expectation."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from generativeqc_compiler.tensor.ir import add, einsum, input_tensor
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec


def occupied_program(natom, nao, nspin, nocc, nvir, occupation):
    """Batch spin frames and Cartesian directions, sharing the AO integral inputs."""
    def axis(name, kind, size):
        return Index(name, IndexSpace(name, kind, size))

    atom, xyz = axis("atom", "batch", natom), axis("xyz", "batch", 3)
    spin = axis("spin", "batch", nspin)
    ao = IndexSpace("ao", "ao", nao)
    p, q = Index("p", ao), Index("q", ao)
    i, v = axis("i", "orbital", nocc), axis("v", "orbital", nvir)

    def tensor(name, *indices):
        return input_tensor(name, TensorSpec(indices, role="input"))

    potential = add(tensor("local", p, q), tensor("nonlocal_", p, q))
    derivative = add(tensor("dlocal", atom, xyz, p, q), tensor("dnonlocal", atom, xyz, p, q))
    ds = tensor("ds", atom, xyz, p, q)
    co, cv = tensor("occupied", spin, p, i), tensor("virtual", spin, p, v)
    uvo = tensor("response", spin, atom, xyz, v, i)
    vco = einsum("pq,sqi->spi", potential, co)
    density = einsum("spi,sqi->spq", co, co)
    projected = einsum("spi,spj->sij", co, vco)
    weighted = einsum("spi,sij->spj", co, projected)
    metric_weight = einsum("spj,sqj->spq", weighted, co)
    response_weight = einsum("spv,spi->svi", cv, vco)
    gradient = add(
        einsum("axpq,spq->sax", derivative, density),
        einsum("axpq,spq->sax", ds, metric_weight),
        einsum("saxvi,svi->sax", uvo, response_weight),
        coefficients=(occupation, -occupation, 2 * occupation),
    )
    return Program({"gradient": gradient}, provenance={"operation": "relaxed occupied ECP expectation"})


def inputs(integrals, ds, frames, occupation):
    nspin = len(frames)
    natom, _, nao, _ = ds.shape
    # Zero padding permits different UHF occupations in one graph. An empty
    # spin or virtual space has a single zero column and contributes zero.
    nocc = max(1, *(co.shape[1] for co, _, _ in frames))
    nvir = max(1, *(cv.shape[1] for _, cv, _ in frames))
    if nspin == 1 and frames[0][0].shape[1] and frames[0][1].shape[1]:
        co, cv, uvo = (v[None] for v in frames[0])
    else:
        co = np.zeros((nspin, nao, nocc))
        cv = np.zeros((nspin, nao, nvir))
        uvo = np.zeros((nspin, natom, 3, nvir, nocc))
        for spin, (occ, virt, response) in enumerate(frames):
            no, nv = occ.shape[1], virt.shape[1]
            co[spin, :, :no] = occ
            cv[spin, :, :nv] = virt
            uvo[spin, :, :, :nv, :no] = response
    feeds = dict(local=integrals.local, nonlocal_=integrals.nonlocal_, dlocal=integrals.local_derivative,
                 dnonlocal=integrals.nonlocal_derivative, ds=ds, occupied=co, virtual=cv, response=uvo)
    return occupied_program(natom, nao, nspin, nocc, nvir, occupation), feeds


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
        raise RuntimeError("native CUDA ECP expectation requires the NVCC toolchain")
    if cache is None:
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "generativeqc" / "ecp"
    preload_cuda_runtime_libraries()
    artifact = compile_cuda(plan, CudaCompilerAdapter(nvcc, target), cache)
    return PreparedCuda(plan, artifact, device=device)
