# Native DF-UCC v2 CUDA working-set admission

Status: implemented
Date: 2026-11-26

The source, compact four-index core and FP64 amplitudes stay live across the CUDA CCSD solve. The one-Q virtual ladder tile describes only its own GEMM temporaries. The residual also retains tau, half-tau, DF panels, Fock permutations and ring blocks at different points. Optional FP32 preconvergence retains the FP64 seed while casting the source, Fock blocks, orbital energies and core.

The CCSD planner takes the maximum simultaneous requirement across these phases before allocating solve buffers. The PPL phase includes the packed same-spin input and output. Packed DIIS history can span every phase. Its capacity leaves room for the extrapolated combination, and pushing a new vector retires the oldest pair first. An unavailable FP32 workspace leaves the ordinary FP64 solve available. FP64 projection follows destruction of the FP32 stage.

The endpoint charges subsequent UCC-owned device buffers as they allocate and releases each charge with its owner. The DF source contributes its separately admitted plan residency. This also bounds Lambda and triples intermediates whose lifetimes depend on the selected path. The Lambda VJP selects a Q tile after accounting for both transposed panels. Triples first allocates its fixed channel workspace, then chooses resident or streamed slabs and fits both vectors and Gram matrices in the remaining page space.

The native capacity regression exercises exact and one-byte-short boundaries, empty-spin cases, overflow and moved buffer ownership. A host replay of the residual buffer lifetimes matched the planner on 12,960 shape and precision combinations. The per-allocation cap covers UCC-owned numeric device buffers. Source and force-pullback admission also accounts for their modeled host storage. Host-side DIIS matrices and triples staging need separate accounting for a combined endpoint limit. Runtime device residency, driver and allocator overhead, convergence and endpoint timing require device measurements.

The borrowed source cuBLAS handle has a 4 MiB workspace, the Ampere/Ada default pool, counted in the source panel peak and its resident bytes (2026-11-27, see `kUccBlasWorkspaceBytes` and `ucc_source_panel_peak_bytes` in `src/cc/df_ucc_v2_source_capacity.hpp`). The earlier 16 KiB covered only the DIIS dot reductions. Workspaces that small force non-split-K kernels, and a microprobe measured them up to 84x slower on an L20 for small-output, long-reduction GEMMs. No complete open-shell endpoint timing yet isolates this change.
