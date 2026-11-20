#pragma once

#include <vector>

namespace {
struct Block {
  int la, lb, a, b, pa, pb, na, nb;
  Size cart_offset, sphere_offset;
};
GENERATIVEQC_ERI_SHARED_HD int ncart(int l) { return (l + 1) * (l + 2) / 2; }
GENERATIVEQC_ERI_SHARED_HD int transform_begin(int l) {
  int p = 0;
  for (int i = 0; i < l; ++i) p += (2 * i + 1) * ncart(i);
  return p;
}
struct Batch {
  int natom;
  const Block* blocks;
  const double *exp, *radial, *xyz, *transforms;
  const int* charges;
  double *cart, *sphere;
  Size nsphere, rows;
};
template <bool Response>
GENERATIVEQC_ERI_DEVICE void block_elements(const Batch& batch, int B, int lane, int lanes) {
  const Block& q = batch.blocks[B];
  Pair pair{q.la, q.lb, q.a,       q.b,          q.na,      q.nb,         batch.natom,
            q.pa, q.pb, batch.exp, batch.radial, batch.xyz, batch.charges};
  const int nc = ncart(q.la) * ncart(q.lb);
  double* dst = batch.cart + q.cart_offset * 3 * batch.rows;
  for (int k = lane; k < nc; k += lanes) element<Response>(pair, k, nc, dst);
}
GENERATIVEQC_ERI_DEVICE void project(const Batch& batch, int B, int lane, int lanes) {
  const Block& q = batch.blocks[B];
  const int ca = ncart(q.la), cb = ncart(q.lb), nb = 2 * q.lb + 1, n = (2 * q.la + 1) * nb;
  const Size cart = ca * cb;
  const double *Ta = batch.transforms + transform_begin(q.la),
               *Tb = batch.transforms + transform_begin(q.lb);
  const double* data = batch.cart + q.cart_offset * 3 * batch.rows;
  for (Size k = lane; k < batch.rows * n; k += lanes) {
    const Size row = k / n;
    const int ij = k % n;
    const double *A = Ta + (ij / nb) * ca, *B = Tb + (ij % nb) * cb;
    const double* S = data + row * cart;
    const double* T = S + batch.rows * cart;
    const double* V = T + batch.rows * cart;
    double valS = 0, valH = 0;
    for (int x = 0; x < ca; ++x) {
      if (A[x] == 0) continue;
      for (int y = 0; y < cb; ++y) {
        if (B[y] == 0) continue;
        const double w = A[x] * B[y];
        valS += w * S[x * cb + y];
        valH += w * (T[x * cb + y] + V[x * cb + y]);
      }
    }
    const Size offset = row * 2 * batch.nsphere + q.sphere_offset + ij;
    batch.sphere[offset] = valS;
    batch.sphere[offset + batch.nsphere] = valH;
  }
}
#ifdef __CUDACC__
template <bool Response>
__global__ void blocks_kernel(Batch b) {
  block_elements<Response>(b, blockIdx.x, threadIdx.x, blockDim.x);
}
__global__ void project_kernel(Batch b) { project(b, blockIdx.x, threadIdx.x, blockDim.x); }
// Each lane sums a fixed ordered AO subsection. The warp reduces a physical coordinate
// without atomics so repeated nuclear centers and nonsymmetric cotangents are reproducible.
__global__ void weight_kernel(const double* data, const double* weights, Size cols, double* out) {
  double sum = 0;
  for (Size col = threadIdx.x; col < cols; col += 32)
    sum += weights[col] * data[blockIdx.x * cols + col];
  for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
  if (threadIdx.x == 0) out[blockIdx.x] = sum;
}
struct Allocation {
  void* pointer{};
  ~Allocation() {
    if (pointer) cudaFree(pointer);
  }
};
#define BATCH_EXPORT GQC_F12_CABS_ONE_ELECTRON_SYMBOL(_batch)
#define WEIGHTS_EXPORT GQC_F12_CABS_ONE_ELECTRON_SYMBOL(_weights)
#else
#define BATCH_EXPORT GQC_F12_CABS_ONE_ELECTRON_SYMBOL(_batch_host_reference)
#define WEIGHTS_EXPORT GQC_F12_CABS_ONE_ELECTRON_SYMBOL(_weights_host_reference)
#endif
}  // namespace

extern "C" int BATCH_EXPORT(int response, int ns, int nat, int nprim, const int* shell,
                            const double* exp, const double* radial, const double* xyz,
                            const int* charges, const double* transform, double* output,
                            Size output_count, Size max_bytes, Size* bytes) {
  if ((response != 0 && response != 1) || ns < 1 || nat < 1 ||
      nat > std::numeric_limits<int>::max() / 3 || nprim < 1 || !shell || !exp || !radial || !xyz ||
      !charges || !transform || !output || !bytes)
    return 1;
  const Size rows = 1 + Size(response) * 3 * nat;
  std::vector<Block> blocks;
  Size nc = 0, nsp = 0, primitive_end = 0;
  for (int i = 0; i < ns; ++i) {
    const int* s = shell + Size(i) * 4;
    if (s[0] < 0 || s[0] > MaxL || s[1] < 0 || s[1] >= nat || s[2] < 0 ||
        Size(s[2]) != primitive_end || s[3] <= s[2] || s[3] > nprim)
      return 1;
    primitive_end = s[3];
    for (int j = 0; j <= i; ++j) {
      const int* t = shell + Size(j) * 4;
      if (s[0] != MaxL && t[0] != MaxL) continue;
      if (blocks.size() >= Size(std::numeric_limits<int>::max())) return 2;
      blocks.push_back({s[0], t[0], s[1], t[1], s[2], t[2], s[3] - s[2], t[3] - t[2], nc, nsp});
      nc += ncart(s[0]) * ncart(t[0]);
      nsp += (2 * s[0] + 1) * (2 * t[0] + 1);
    }
  }
  if (blocks.empty() || primitive_end != Size(nprim)) return 1;
  // Enforce a byte budget before multiplication by the nuclear Jacobian dimension.
  if (nsp > max_bytes / sizeof(double) / rows / 2 || nc > max_bytes / sizeof(double) / rows / 3)
    return 2;
  if (output_count != 2 * rows * nsp) return 1;
  const Size nt = transform_begin(MaxL + 1), input = 2 * Size(nprim) + 3 * Size(nat) + nt;
  const Size sizeof_blocks = blocks.size() * sizeof(Block);
  const Size charge_bytes = Size(nat) * sizeof(int);
  // Staged data plus the complete Cartesian scratch and compact spherical output.
  const Size raw_count = 3 * rows * nc;
  const Size cap = max_bytes / sizeof(double);
  if (input > cap || output_count > cap - input || raw_count > cap - input - output_count) return 2;
  const Size doubles_bytes = (input + raw_count + output_count) * sizeof(double);
  if (sizeof_blocks > max_bytes - doubles_bytes ||
      charge_bytes > max_bytes - doubles_bytes - sizeof_blocks)
    return 2;
  *bytes = doubles_bytes + sizeof_blocks + charge_bytes;
  for (int i = 0; i < nprim; ++i)
    if (!(exp[i] > 0) || !std::isfinite(exp[i]) || !std::isfinite(radial[i])) return 1;
  for (int i = 0; i < nat; ++i)
    if (charges[i] <= 0 || !std::isfinite(xyz[3 * i]) || !std::isfinite(xyz[3 * i + 1]) ||
        !std::isfinite(xyz[3 * i + 2]))
      return 1;
  for (Size i = 0; i < nt; ++i)
    if (!std::isfinite(transform[i])) return 1;
#ifdef __CUDACC__
  Allocation owner;
  if (cudaMalloc(&owner.pointer, *bytes) != cudaSuccess) return 3;
  auto* de = static_cast<double*>(owner.pointer);
  double *dr = de + nprim, *dx = dr + nprim, *dt = dx + 3 * Size(nat), *cart = dt + nt,
         *sphere = cart + raw_count;
  auto* db = reinterpret_cast<Block*>(sphere + output_count);
  auto* dz = reinterpret_cast<int*>(db + blocks.size());
  auto upload = [](void* d, const void* h, Size n) {
    return cudaMemcpy(d, h, n, cudaMemcpyHostToDevice) == cudaSuccess;
  };
  if (!upload(de, exp, nprim * sizeof(double)) || !upload(dr, radial, nprim * sizeof(double)) ||
      !upload(dx, xyz, 3 * Size(nat) * sizeof(double)) ||
      !upload(dt, transform, nt * sizeof(double)) || !upload(db, blocks.data(), sizeof_blocks) ||
      !upload(dz, charges, charge_bytes) ||
      cudaMemset(cart, 0, raw_count * sizeof(double)) != cudaSuccess)
    return 3;
  Batch b{nat, db, de, dr, dx, dt, dz, cart, sphere, nsp, rows};
  if (response)
    blocks_kernel<true><<<blocks.size(), 32>>>(b);
  else
    blocks_kernel<false><<<blocks.size(), 32>>>(b);
  if (cudaGetLastError() != cudaSuccess) return 3;
  project_kernel<<<blocks.size(), 128>>>(b);
  if (cudaGetLastError() != cudaSuccess || cudaMemcpy(output, sphere, output_count * sizeof(double),
                                                      cudaMemcpyDeviceToHost) != cudaSuccess)
    return 3;
#else
  std::vector<double> cart(raw_count, 0);
  Batch b{nat, blocks.data(), exp, radial, xyz, transform, charges, cart.data(), output, nsp, rows};
  for (Size i = 0; i < blocks.size(); ++i) {
    if (response)
      block_elements<true>(b, i, 0, 1);
    else
      block_elements<false>(b, i, 0, 1);
    project(b, i, 0, 1);
  }
#endif
  return 0;
}
extern "C" int WEIGHTS_EXPORT(int nd, Size cols, const double* jac, const double* w, double* out,
                              Size max_bytes, Size* bytes) {
  if (nd <= 0 || cols == 0 || !jac || !w || !out || !bytes) return 1;
  const Size cap = max_bytes / sizeof(double);
  if (Size(nd) > cap || cols > (cap - nd) / (Size(nd) + 1)) return 2;
  *bytes = (cols * (Size(nd) + 1) + nd) * sizeof(double);
#ifdef __CUDACC__
  Allocation owner;
  if (cudaMalloc(&owner.pointer, *bytes) != cudaSuccess) return 3;
  auto* ds = static_cast<double*>(owner.pointer);
  double *dw = ds + cols * nd, *dg = dw + cols;
  if (cudaMemcpy(ds, jac, cols * nd * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dw, w, cols * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    return 3;
  weight_kernel<<<nd, 32>>>(ds, dw, cols, dg);
  if (cudaGetLastError() != cudaSuccess ||
      cudaMemcpy(out, dg, nd * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 3;
#else
  for (int axis = 0; axis < nd; ++axis) {
    double sum = 0;
    for (Size k = 0; k < cols; ++k) sum += w[k] * jac[axis * cols + k];
    out[axis] = sum;
  }
#endif
  return 0;
}
#undef BATCH_EXPORT
#undef WEIGHTS_EXPORT
