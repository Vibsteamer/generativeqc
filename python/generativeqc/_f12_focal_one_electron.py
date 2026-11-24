"""One-electron data for the real-spherical orbital and complementary Gaussian bases."""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._f12_focal_shells import pack_shells, spherical_projections
from .elements import checked_integer

if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike

    from .experimental.f12_focal import Backend, FocalRequest


@dataclass(frozen=True)
class FocalOneElectron:
    """Matrices in OBS-first order, with positive nuclear derivatives when requested."""

    overlap: np.ndarray
    hcore: np.ndarray
    d_overlap: np.ndarray | None
    d_hcore: np.ndarray | None
    obs_ao_count: int
    surface_id: str
    basis_id: str
    executed_backend: Backend
    native_workspace_bytes: int

    def nuclear_gradient(
        self,
        overlap_weight: np.ndarray,
        hcore_weight: np.ndarray,
        *,
        run: Callable | None,
    ) -> np.ndarray:
        """Positive derivative of the two weighted one-electron operators on this backend."""
        from generativeqc_compiler.method.nuclear_integral_response import (
            overlap_hcore_gradient,
        )

        if self.d_overlap is None or self.d_hcore is None:
            raise ValueError("Nuclear contraction requires both one-electron Jacobians")
        for weight in (overlap_weight, hcore_weight):
            if weight.shape != self.overlap.shape or weight.dtype != np.float64:
                raise ValueError(
                    "One-electron weights must be full FP64 matrices in the source AO order"
                )
        if self.executed_backend == "cpu_reference":
            return np.einsum("axij,ij->ax", self.d_overlap, overlap_weight) + np.einsum(
                "axij,ij->ax", self.d_hcore, hcore_weight
            )
        if self.executed_backend != "cuda" or run is None:
            raise ValueError(
                "CUDA one-electron nuclear contraction requires its graph runner"
            )
        graph = overlap_hcore_gradient(len(self.d_overlap), len(self.overlap))
        result = run(
            graph,
            {
                "d_overlap": self.d_overlap,
                "d_hcore": self.d_hcore,
                "overlap": overlap_weight,
                "hcore": hcore_weight,
            },
        )
        if result.backend != "cuda-fp64-ordinary-stream":
            raise RuntimeError(
                "One-electron nuclear contraction returned the wrong backend"
            )
        return result.outputs["gradient"]


def one_electron_sources(
    request: FocalRequest,
    *,
    library: str | PathLike[str],
    derivatives: bool = True,
    orbital_only: bool = False,
    budget_mib: int = 512,
) -> FocalOneElectron:
    """Evaluate S and T+V on the geometry of an HF+CABS or MP2-F12 request.

    The supplied F12 integral library must implement the requested backend.
    Coordinates are in Bohr. Each matrix starts with all orbital AOs and then
    all atom-centered OptRI AOs, both in native real-spherical order. With
    orbital_only=True the matrices contain just the ordinary orbital basis. With
    derivatives=True, each Jacobian has shape (atoms, 3, AOs, AOs) and includes
    the motion of both Gaussian centers and the nuclear attraction centers.
    For scalar ECPs the requested backend also evaluates through-f semilocal
    projectors with a zero residual local channel. The point attraction then
    uses the effective ionic charges. The numeric budget bounds workspace.
    """
    if request.role not in ("hf_cabs", "mp2_f12_correlation"):
        raise ValueError("one-electron F12 sources require an HF+CABS or MP2-F12 role")
    if request.backend not in ("cpu_reference", "cuda"):
        raise ValueError("one-electron F12 sources require cpu_reference or cuda")
    if (
        not request.atoms
        or type(derivatives) is not bool
        or type(orbital_only) is not bool
    ):
        raise ValueError(
            "one-electron F12 sources require nuclei and a boolean derivative option"
        )
    maximum = (
        checked_integer(
            budget_mib, "integral budget in MiB", low=1, high=(2**63 - 1) >> 20
        )
        << 20
    )
    basis = request.basis
    groups = [("orbital", basis.orbital)]
    if not orbital_only:
        groups.append(("CABS", basis.cabs))
    shells = pack_shells(request.atoms, tuple(groups), maximum)
    nuclei = [basis.orbital.by_element[a.atomic_number] for a in request.atoms]
    terms = {}
    if any(n.ecp_core_electrons for n in nuclei):
        from ._f12_focal_ecp import projector_terms

        terms = projector_terms(nuclei)
    charges = np.asarray(
        [n.nuclear_charge - n.ecp_core_electrons for n in nuclei], dtype=np.int32
    )
    xyz = np.asarray([a.position for a in request.atoms], dtype=np.float64)
    ns = len(shells.angular)
    widths = (2 * shells.angular + 1).tolist()
    n = sum(widths)
    rows = 1 + 3 * len(nuclei) * int(derivatives)
    triangle = (n * n + sum(w * w for w in widths)) // 2
    packed_bytes, dense_bytes = 16 * rows * triangle, 16 * rows * n * n
    if packed_bytes + dense_bytes > maximum:
        raise MemoryError(
            "combined one-electron output and shell staging exceed the numeric budget"
        )

    path = Path(library).resolve()
    lib = ct.CDLL(str(path))
    suffix = "_host_reference" if request.backend == "cpu_reference" else ""
    name = "gqc_f12_cabs_i_one_electron_full_batch" + suffix
    try:
        native = getattr(lib, name)
    except AttributeError as err:
        raise RuntimeError(
            f"F12 library lacks its {request.backend} one-electron kernel"
        ) from err
    d = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    i = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    size = ct.c_size_t
    native.argtypes = [ct.c_int] * 4 + [
        i,
        d,
        d,
        d,
        i,
        d,
        d,
        size,
        size,
        ct.POINTER(size),
    ]
    native.restype = ct.c_int
    # The native full-basis ABI uses angular-indexed offsets in the through-i table.
    transforms = spherical_projections(path, set(range(7)))
    table = np.concatenate([transforms[l].ravel() for l in range(7)])
    packed_shells = np.column_stack(
        (shells.angular, shells.atom, shells.prim_offsets[:-1], shells.prim_offsets[1:])
    )
    retained = sum(
        arr.nbytes
        for arr in (packed_shells, shells.exponents, shells.radial, charges, xyz, table)
    )
    available = maximum - packed_bytes - retained
    if available < dense_bytes:
        raise MemoryError(
            "combined one-electron geometry and matrices exceed the numeric budget"
        )
    compact = np.empty((rows, 2, triangle), dtype=np.float64)
    used = size()
    rc = native(
        int(derivatives),
        ns,
        len(nuclei),
        len(shells.exponents),
        packed_shells,
        shells.exponents,
        shells.radial,
        xyz,
        charges,
        table,
        compact,
        compact.size,
        available,
        ct.byref(used),
    )
    if rc == 2:
        raise MemoryError("native one-electron batch exceeds the numeric budget")
    if rc or not 0 < used.value <= available or not np.isfinite(compact).all():
        raise RuntimeError(
            f"native {request.backend} one-electron batch failed (status {rc})"
        )
    matrices = np.empty((rows, 2, n, n), dtype=np.float64)
    offsets = np.r_[0, np.cumsum(widths)]
    pos = 0
    for first, a in enumerate(widths):
        left = slice(offsets[first], offsets[first + 1])
        for second in range(first + 1):
            b = widths[second]
            right = slice(offsets[second], offsets[second + 1])
            block = compact[:, :, pos : pos + a * b].reshape(rows, 2, a, b)
            matrices[:, :, left, right] = block
            if first != second:
                matrices[:, :, right, left] = block.swapaxes(-1, -2)
            pos += a * b
    if terms:
        from ._f12_focal_ecp import add_projector_residual

        del block, compact
        residual_budget = maximum - dense_bytes - retained
        if request.backend == "cpu_reference":
            add_projector_residual(
                matrices[:, 1],
                shells,
                transforms,
                xyz,
                terms,
                derivatives,
                residual_budget,
            )
        else:
            from ._f12_focal_ecp_cuda import add_cuda_projector_residual

            ecp_bytes = add_cuda_projector_residual(
                matrices[:, 1],
                path,
                shells,
                transforms,
                xyz,
                terms,
                derivatives,
                residual_budget,
            )
            used.value = max(used.value, ecp_bytes)
    jac_shape = (len(nuclei), 3, n, n)
    return FocalOneElectron(
        matrices[0, 0],
        matrices[0, 1],
        matrices[1:, 0].reshape(jac_shape) if derivatives else None,
        matrices[1:, 1].reshape(jac_shape) if derivatives else None,
        shells.ao_counts[0],
        request.surface_id,
        basis.basis_id,
        request.backend,
        used.value,
    )
