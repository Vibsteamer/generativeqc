"""Canonical same-space multipliers for a real orbital-frame cotangent.

This completes canonical constraints before the existing physical RHF Z solve.
It neither solves the occupied/virtual response nor supplies nuclear derivatives.
"""

from fractions import Fraction

from ..tensor import Index, IndexSpace, Program, TensorSpec, add, einsum, input_tensor
from ..tensor.ir import divide
from ..tensor.scalar_cpp import emit_scalar_cpp


def orbital_rotation_cotangent(atomic: int, orbitals: int) -> Program:
    """Project the AO coefficient cotangent with the same physical OBS frame."""
    if any(type(n) is not int or n < 1 for n in (atomic, orbitals)):
        raise ValueError("Positive AO and orbital dimensions are required")
    if orbitals > atomic:
        raise ValueError("Orbital frame exceeds its AO dimension")
    a = Index("a", IndexSpace("canonical_frame_ao", "orbital", atomic))
    orbital = IndexSpace("canonical_frame_orbital", "orbital", orbitals)
    p, q = Index("p", orbital), Index("q", orbital)
    c = input_tensor("coefficients", TensorSpec((a, p), role="input"))
    bar = input_tensor("frame_bar", TensorSpec((a, q), role="input"))
    return Program(
        {"rotation_gradient": einsum("ap,aq->pq", c, bar)},
        provenance={"contract": "M=C.T barC; full-Frobenius canonical constraints"},
    )


def canonical_pair_terms() -> tuple[Program, Program]:
    """Return the finite pair diagnostics and the nonzero-gap multiplier.

    For p<q, dL/dkappa_pq=M_pq-M_qp and dF_pq=(eps_p-eps_q)kappa_pq.
    A symmetric full-Frobenius Fock seed supplies two copies of the latter.
    The owner must qualify the gap before executing the second program.
    """
    spec = TensorSpec(role="input")
    mpq, mqp, ep, eq = (input_tensor(k, spec) for k in ("mpq", "mqp", "ep", "eq"))
    gap = add(ep, eq, coefficients=(1, -1))
    torque = add(mpq, mqp, coefficients=(1, -1))
    numerator = add(mpq, mqp, coefficients=(Fraction(-1, 2), Fraction(1, 2)))
    return (
        Program({"gap": gap, "torque": torque}),
        Program({"seed": divide(numerator, gap)}),
    )


def emit_canonical_frame_cuda() -> str:
    """Emit pair rules and a guarded device schedule; no runtime imports."""
    diagnostics, multiplier = canonical_pair_terms()
    helpers = "\n".join(
        emit_scalar_cpp(
            graph,
            function_name=name,
            input_order=("mpq", "mqp", "ep", "eq"),
            output_order=outputs,
        ).replace("inline bool", "__device__ inline bool")
        for graph, name, outputs in (
            (diagnostics, "canonical_frame_pair_diagnostics", ("gap", "torque")),
            (multiplier, "canonical_frame_pair_multiplier", ("seed",)),
        )
    )
    return (
        "#ifndef GENERATIVEQC_GENERATED_CANONICAL_FRAME_CUH\n"
        "#define GENERATIVEQC_GENERATED_CANONICAL_FRAME_CUH\n"
        "#include <cuda_runtime.h>\n#include <cstddef>\n"
        + helpers
        + r"""
namespace generativeqc::tensor {
static __global__ void canonical_frame_seed_kernel(
    std::size_t n, std::size_t occupied, const double* energies, const double* rotation,
    double gap_threshold, double stationarity_threshold, double* seed, int* flags) {
  for (std::size_t k=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x; k<n*n;
       k+=std::size_t(blockDim.x)*gridDim.x) {
    const auto p=k/n, q=k%n;
    seed[k]=0.0;
    double gap, torque;
    if (!canonical_frame_pair_diagnostics(rotation[k], rotation[q*n+p],
                                         energies[p], energies[q], gap, torque)) {
      atomicExch(flags,1); continue;
    }
    if (p==q || ((p<occupied)!=(q<occupied))) continue;
    if (fabs(gap)<=gap_threshold) {
      if (fabs(torque)>stationarity_threshold) atomicExch(flags+1,1);
      continue;
    }
    double value;
    if (!canonical_frame_pair_multiplier(rotation[k], rotation[q*n+p],
                                        energies[p], energies[q], value)) {
      atomicExch(flags,1); continue;
    }
    seed[k]=value;
  }
}
inline void launch_canonical_frame_seed(
    std::size_t n, std::size_t occupied, const double* energies, const double* rotation,
    double gap_threshold, double stationarity_threshold, double* seed, int* flags,
    cudaStream_t stream) {
  const auto count=(n*n+255)/256;
  const auto blocks=static_cast<unsigned>(count<65535 ? count : 65535);
  canonical_frame_seed_kernel<<<blocks,256,0,stream>>>(
      n,occupied,energies,rotation,gap_threshold,stationarity_threshold,seed,flags);
}
}  // namespace generativeqc::tensor
#endif
"""
    )
