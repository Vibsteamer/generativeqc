"""Assemble OBS+CABS real-spherical one-electron sources with an opt-in h-shell cohort."""

import ctypes as ct
import time
from collections.abc import Mapping, Sequence
from math import prod
from pathlib import Path

import numpy as np
from generativeqc._one_electron_gradient import IntegralGradient
from generativeqc.elements import atomic_number

from benchmarks.f12_molecular_one_electron import native_contraction
from benchmarks.f12_molecular_one_electron_forward import combined_matrices


class CabsHShells:
    """One high-shell CUDA batch per geometry, with a compact Jacobian reused on the reverse pass."""

    def __init__(
        self,
        case: dict,
        native_to_source: Sequence[int],
        topology: Mapping[str, np.ndarray],
        library: Path,
        *,
        host_reference: bool = False,
    ) -> None:
        self.n = len(native_to_source)
        p = np.asarray(native_to_source)
        if (
            p.dtype.kind not in "iu"
            or p.shape != (self.n,)
            or not np.array_equal(np.sort(p), np.arange(self.n))
        ):
            raise ValueError("expected a bijective orbital AO source mapping")
        z = case["elements"]
        self.charges = np.asarray([atomic_number(i) for i in z], dtype=np.int32)
        self.xyz = np.asarray(case["coordinates_bohr"], dtype=np.float64, order="C")
        if self.xyz.shape != (len(z), 3) or not np.isfinite(self.xyz).all():
            raise ValueError("invalid geometry")
        ns = int(topology["orbital_shell_count"])
        ls = np.asarray(topology["angular"][:ns], dtype=np.int32)
        atoms = np.asarray(topology["atom"][:ns], dtype=np.int32)
        offsets = np.asarray(topology["prim_offsets"][: ns + 1], dtype=np.int32)
        if (
            ns != len(case["shells"])
            or len(ls) != ns
            or ns == 0
            or min(ls) < 0
            or max(ls) != 5
            or not np.array_equal(self.xyz[atoms], topology["centers"][:ns])
            or len(offsets) != ns + 1
            or offsets[0] != 0
            or min(np.diff(offsets)) < 1
        ):
            raise ValueError("incompatible CABS h shell topology")
        self.shells = np.ascontiguousarray(
            np.stack((ls, atoms, offsets[:-1], offsets[1:]), axis=1), dtype=np.int32
        )
        self.exps, self.radial = [
            np.ascontiguousarray(topology[k][: offsets[-1]], dtype=np.float64)
            for k in ("exponents", "radial")
        ]
        rows = [
            np.asarray(t[: 2 * l + 1], dtype=np.intp)
            for l, t in zip(ls, topology["spherical_rows"][:ns], strict=True)
        ]
        if not np.array_equal(np.sort(np.concatenate(rows)), np.arange(self.n)):
            raise ValueError(
                "orbital source rows must cover the complete public AO basis"
            )
        self.blocks = [
            (i, j, rows[i], rows[j])
            for i in range(ns)
            for j in range(i + 1)
            if ls[i] == 5 or ls[j] == 5
        ]
        self.size = sum(len(a) * len(b) for _, _, a, b in self.blocks)
        # Cross-check ownership by the complete ordered AO tuple, including shell and atom.
        # The packed basis may have a different shell order from the explicit Calculator input.
        signatures = {tuple(r): i for i, r in enumerate(rows)}
        low_rows = []
        low_shells = []
        off = 0
        for shell in case["shells"]:
            l = int(shell["angular_momentum"])
            atom = int(shell["atom_index"])
            sub = p[off : off + 2 * l + 1]
            off += len(sub)
            j = signatures.get(tuple(sub))
            if j is None or (ls[j], atoms[j]) != (l, atom):
                raise ValueError("native/source shell mapping mismatch")
            e, c = np.asarray(shell["primitives"], dtype=np.float64).T
            block = slice(offsets[j], offsets[j + 1])
            if not np.array_equal(e, self.exps[block]) or min(e) <= 0:
                raise ValueError(
                    "native and integral-layout primitive exponents disagree"
                )
            # Libint radial coefficients multiply the unnormalized x^l Cartesian Gaussian.
            overlap = (
                2 * np.sqrt(e[:, None] * e[None, :]) / (e[:, None] + e[None, :])
            ) ** (l + 1.5)
            radial = (
                c
                * (2 * e / np.pi) ** 0.75
                * (4 * e) ** (0.5 * l)
                / np.sqrt((c @ overlap @ c) * prod(range(1, 2 * l + 1, 2)))
            )
            if not np.allclose(radial, self.radial[block], rtol=3e-13, atol=1e-14):
                raise ValueError(
                    "native and integral-layout radial normalization disagree"
                )
            if l != 5:
                low_shells.append(shell)
                low_rows.extend(sub)
        if off != self.n or not low_shells:
            raise ValueError("a combined CABS h basis requires its orbital subspace")
        self.low_source = np.sort(low_rows).astype(np.intp)
        self.low_permutation = np.searchsorted(self.low_source, low_rows)
        self.low_case = {**case, "shells": low_shells}
        transforms = []
        for l in range(6):
            dims = (2 * l + 1, (l + 1) * (l + 2) // 2)
            t = topology.get(f"transform_{l}")
            if t is None:
                if l in ls:
                    raise ValueError(f"missing transform for l={l}")
                t = np.zeros(dims)
            t = np.asarray(t, dtype=np.float64)
            if t.shape != dims or not np.isfinite(t).all():
                raise ValueError("invalid real-spherical transform")
            transforms.append(t.ravel())
        self.transforms = np.concatenate(transforms)
        self.lib = ct.CDLL(str(Path(library).resolve()))
        suffix = "_host_reference" if host_reference else ""
        d = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        i = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        size = ct.c_size_t
        ptr = ct.POINTER(size)
        self.batch = getattr(self.lib, "gqc_f12_cabs_h_one_electron_batch" + suffix)
        self.batch.argtypes = [ct.c_int] * 4 + [i, d, d, d, i, d, d, size, size, ptr]
        self.batch.restype = ct.c_int
        self.contract = getattr(
            self.lib, "gqc_f12_cabs_h_one_electron_weights" + suffix
        )
        self.contract.argtypes = [ct.c_int, size, d, d, d, size, ptr]
        self.contract.restype = ct.c_int
        self.data = None
        self.backend = "native_host_control" if host_reference else "native_cuda_fp64"

    def evaluate(self, *, derivatives: bool, max_bytes: int) -> dict:
        n = len(self.charges)
        rows = 1 + 3 * n * int(derivatives)
        count = rows * 2 * self.size
        if count * 8 > max_bytes:
            raise MemoryError("compact combined-basis shell buffer exceeds budget")
        result = np.empty((rows, 2, self.size), dtype=np.float64)
        used = ct.c_size_t()
        t = time.monotonic()
        rc = self.batch(
            int(derivatives),
            len(self.shells),
            n,
            len(self.exps),
            self.shells,
            self.exps,
            self.radial,
            self.xyz,
            self.charges,
            self.transforms,
            result,
            result.size,
            max_bytes,
            ct.byref(used),
        )
        if rc or not np.isfinite(result).all():
            raise RuntimeError(f"h-shell batch failed (status {rc})")
        self.data = result
        return {
            "backend": self.backend,
            "device_bytes": used.value,
            "compact_host_bytes": result.nbytes,
            "high_shell_pairs": len(self.blocks),
            "spherical_elements_per_matrix": self.size,
            "native_seconds": time.monotonic() - t,
            "batch_calls": 1,
        }

    def assemble(
        self, low_overlap: np.ndarray, low_hcore: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.data is None:
            raise ValueError("missing high shell integrals")
        low = np.ix_(self.low_source, self.low_source)
        out = np.empty((2, self.n, self.n))
        out[0][low] = low_overlap
        out[1][low] = low_hcore
        pos = 0
        for i, j, a, b in self.blocks:
            stop = pos + len(a) * len(b)
            part = self.data[0, :, pos:stop].reshape(2, len(a), len(b))
            out[:, a[:, None], b[None, :]] = part
            if i != j:
                out[:, b[:, None], a[None, :]] = part.transpose(0, 2, 1)
            pos = stop
        return out[0], out[1]

    def high_gradient(
        self, ws: np.ndarray, wh: np.ndarray, max_bytes: int
    ) -> tuple[np.ndarray, dict]:
        if self.data is None or self.data.shape[0] != 1 + 3 * len(self.charges):
            raise ValueError("no retained high shell nuclear derivatives")
        weights = np.empty((2, self.size))
        pos = 0
        for w in (ws, wh):
            if (
                w.shape != (self.n, self.n)
                or w.dtype != np.float64
                or not np.isfinite(w).all()
            ):
                raise ValueError(
                    "expected complete finite FP64 weights in source AO order"
                )
        for i, j, a, b in self.blocks:
            stop = pos + len(a) * len(b)
            for channel, w in enumerate((ws, wh)):
                part = w[np.ix_(a, b)]
                if i != j:
                    part += w[np.ix_(b, a)].T
                weights[channel, pos:stop] = part.ravel()
            pos = stop
        out = np.empty_like(self.xyz)
        used = ct.c_size_t()
        t = time.monotonic()
        rc = self.contract(
            out.size,
            2 * self.size,
            self.data[1:],
            weights,
            out,
            max_bytes,
            ct.byref(used),
        )
        if rc or not np.isfinite(out).all():
            raise RuntimeError(f"h-shell weight contraction failed (status {rc})")
        return out, {
            "device_bytes": used.value,
            "native_seconds": time.monotonic() - t,
            "batch_calls": 1,
        }

    def matrices(
        self, session_library: Path, *, derivatives: bool, max_bytes: int
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        hip = self.evaluate(derivatives=derivatives, max_bytes=max_bytes)
        s, h, low = combined_matrices(
            self.low_case,
            self.low_permutation,
            session_library,
            maximum_bytes=max_bytes,
        )
        S, H = self.assemble(s, h)
        return (
            S,
            H,
            {
                "device_bytes": max(hip["device_bytes"], low["device_bytes"]),
                "backend": self.backend,
                "low": low,
                "high": hip,
            },
        )

    def gradient(
        self, ws: np.ndarray, wh: np.ndarray, max_bytes: int
    ) -> IntegralGradient:
        high, hi_resources = self.high_gradient(ws, wh, max_bytes)
        low = np.ix_(self.low_source, self.low_source)
        lo = native_contraction(self.low_case, ws[low], wh[low], self.low_permutation)
        return IntegralGradient(
            lo.gradient + high,
            {
                **lo.resources,
                "low_device_bytes": lo.resources["device_bytes"],
                "high_device_bytes": hi_resources["device_bytes"],
                "high_contraction_calls": hi_resources["batch_calls"],
                "high_microseconds": round(1e6 * hi_resources["native_seconds"]),
                "device_bytes": max(
                    lo.resources["device_bytes"], hi_resources["device_bytes"]
                ),
            },
        )
