"""Native S and Hcore of the combined OBS+CABS in the source spherical AO order."""

import ctypes as ct
import time
from pathlib import Path

import numpy as np
from generativeqc import Calculator, Primitive, Shell
from generativeqc._api_types import Atom
from generativeqc._context_owner import ContextOwner


def combined_matrices(
    case: dict,
    native_to_source: np.ndarray,
    session_library_path: Path,
    *,
    maximum_bytes: int = 128 << 20,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build S and T+V_nuc on CUDA, retaining the native integral allocation ledger."""
    start = time.monotonic()
    shells = tuple(
        Shell(
            s["atom_index"],
            s["angular_momentum"],
            tuple(Primitive(*p) for p in s["primitives"]),
        )
        for s in case["shells"]
    )
    n = sum(2 * s.angular_momentum + 1 for s in shells)
    p = np.asarray(native_to_source)
    if (
        p.dtype.kind not in "iu"
        or p.shape != (n,)
        or not np.array_equal(np.sort(p), np.arange(n))
    ):
        raise ValueError(
            "native_to_source must be a bijection on the combined spherical AO basis"
        )
    if not 0 < maximum_bytes < (1 << (8 * ct.sizeof(ct.c_size_t))):
        raise ValueError("maximum_bytes must be a positive native size")
    if 2 * 8 * n * n > maximum_bytes:
        raise MemoryError("combined overlap and Hcore exceed the numeric budget")
    geometry = tuple(
        Atom.from_value(pair)
        for pair in zip(case["elements"], case["coordinates_bohr"], strict=True)
    )
    calc = Calculator(
        "rhf",
        device="cuda",
        precision="fp64",
        basis=shells,
        basis_representation="spherical",
    )
    library = calc._library
    bridge = ct.CDLL(str(Path(session_library_path).resolve()))
    call = bridge.gqc_f12_molecular_one_electron_forward
    doubles = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    counts = np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS")
    call.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        doubles,
        doubles,
        counts,
        ct.POINTER(ct.c_char),
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    owner = ContextOwner(library)
    system = ct.c_void_p()
    overlap = np.empty((n, n))
    hcore = np.empty_like(overlap)
    metrics = np.empty(5, dtype=np.uint64)
    error = ct.create_string_buffer(512)
    try:
        with owner.lock:
            context = owner.get(calc._context_descriptor())
            # The validated standalone one-electron g-shell system constructor
            # is shared with the corresponding fixed-weight derivative.
            system = calc._create_native_system(
                context,
                geometry,
                int(case.get("charge", 0)),
                int(case.get("multiplicity", 1)),
                operator="one_electron_gradient",
                derivative_order=1,
            )
            rc = call(
                context,
                system,
                n * n,
                maximum_bytes,
                overlap,
                hcore,
                metrics,
                error,
                len(error),
            )
            if rc:
                raise RuntimeError(
                    f"CUDA combined-basis S/Hcore returned {rc}: {error.value.decode()}"
                )
    finally:
        try:
            if system:
                library.generativeqc_system_destroy(system)
        finally:
            owner.clear()
    inverse = np.argsort(p)
    source_order = np.ix_(inverse, inverse)
    resources = dict(
        zip(
            (
                "device_bytes",
                "device_allocations",
                "device_requested_bytes",
                "cartesian_ao",
                "spherical_ao",
            ),
            map(int, metrics),
            strict=True,
        )
    )
    resources.update(backend="native_cuda_fp64", seconds=time.monotonic() - start)
    return overlap[source_order], hcore[source_order], resources
