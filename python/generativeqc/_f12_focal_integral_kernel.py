"""Native Gaussian channel bindings and Cartesian/spherical derivative contractions."""

import ctypes as ct
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

_OPCODES = {
    "coulomb": 0,
    "f12": 0,
    "f12_over_coulomb": 1,
    "f12_squared": 2,
    "double_commutator": 3,
}
SLATER_FIT_NORMALIZATIONS = {
    "negative_exponential": 0,
    "negative_exponential_over_beta": 1,
}
DEFAULT_SLATER_FIT_NORMALIZATION = "negative_exponential_over_beta"
# Non-_ex native entrypoints in f12_molecular_integral_sink.cu pass
# kDefaultSlaterFitNormalizationCode = 1 (negative_exponential_over_beta, 1/beta).
LEGACY_SLATER_FIT_NORMALIZATION = DEFAULT_SLATER_FIT_NORMALIZATION
LEGACY_SLATER_NORM_CODE = SLATER_FIT_NORMALIZATIONS[LEGACY_SLATER_FIT_NORMALIZATION]
_TRANSFORM_OFFSETS = (0, 1, 10, 40, 110, 245, 476)
_TRANSFORM_TABLE_SIZE = 840
_VALUE_TOLERANCE = 1.0e-8
_GRADIENT_TOLERANCE = 3.0e-8
_TRANSLATION_TOLERANCE = 3.0e-9


def slater_normalization_code(normalization: str) -> int:
    if normalization not in SLATER_FIT_NORMALIZATIONS:
        raise ValueError(
            f"unknown Slater fit normalization {normalization!r}; "
            f"expected one of {sorted(SLATER_FIT_NORMALIZATIONS)}"
        )
    return SLATER_FIT_NORMALIZATIONS[normalization]


def bind(
    library: Path,
    kernel: str,
    host_reference: bool,
    *,
    extended: bool = False,
) -> Callable:
    integers = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    doubles = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    suffix = "_host_reference" if host_reference else ""
    ext = "_ex" if extended else ""
    function = getattr(
        ct.CDLL(str(library.resolve())),
        f"gqc_f12_weighted_shells_{kernel}{ext}{suffix}",
    )
    args = [ct.c_int, ct.c_int, ct.c_double]
    if extended:
        args.append(ct.c_int)
    args.extend(
        [
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            integers,
            integers,
            doubles,
            doubles,
            doubles,
            integers,
            integers,
            doubles,
            doubles,
            ct.c_uint64,
            ct.POINTER(ct.c_uint64),
        ]
    )
    function.argtypes = args
    function.restype = ct.c_int
    return function


def bind_spherical_forward(
    library: Path, kind: str, host_reference: bool
) -> Callable | None:
    integers = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    doubles = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    suffix = "_host_reference" if host_reference else ""
    cdll = ct.CDLL(str(library.resolve()))
    func = getattr(cdll, f"gqc_f12_spherical_forward_{kind}{suffix}", None)
    if func is None:
        return None
    func.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_double,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        integers,
        integers,
        doubles,
        doubles,
        doubles,
        integers,
        doubles,
        doubles,
        ct.c_uint64,
        ct.POINTER(ct.c_uint64),
    ]
    func.restype = ct.c_int
    return func


def bind_spherical_weighted(
    library: Path, kernel: str, host_reference: bool
) -> Callable | None:
    integers = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    doubles = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    suffix = "_host_reference" if host_reference else ""
    cdll = ct.CDLL(str(library.resolve()))
    func = getattr(cdll, f"gqc_f12_spherical_weighted_{kernel}{suffix}", None)
    if func is None:
        return None
    func.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_double,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        integers,
        integers,
        integers,
        doubles,
        doubles,
        doubles,
        integers,
        doubles,
        doubles,
        doubles,
        doubles,
        ct.c_uint64,
        ct.POINTER(ct.c_uint64),
    ]
    func.restype = ct.c_int
    return func


def spherical_layout_descriptor(
    layout: Mapping[str, np.ndarray], group: str
) -> dict | None:
    """Return packed contiguous spherical shell metadata when layout uses canonical ordering."""
    if group not in ("metric", "ao"):
        return None
    angular = np.ascontiguousarray(layout["angular"], dtype=np.int32)
    if angular.ndim != 1 or len(angular) < 2 or int(angular.max()) > 6 or int(angular.min()) < 0:
        return None
    ncombined = int(layout["orbital_shell_count"])
    nshell = len(angular)
    if not 0 < ncombined < nshell:
        return None
    norb, naux = map(int, layout["ao_sizes"])
    sph_rows = layout["spherical_rows"]
    if sph_rows.shape[0] != nshell:
        return None

    sph_offsets = np.empty(nshell, dtype=np.int32)
    for start, end, expected_total in (
        (0, ncombined, norb),
        (ncombined, nshell, naux),
    ):
        offset = 0
        for s in range(start, end):
            width = 2 * int(angular[s]) + 1
            if width > sph_rows.shape[1]:
                return None
            if not np.array_equal(sph_rows[s, :width], np.arange(offset, offset + width)):
                return None
            sph_offsets[s] = offset
            offset += width
        if offset != expected_total:
            return None

    blocks = layout[group + "_blocks"]
    nfit = nshell - ncombined
    if group == "metric":
        n_pairs = nfit * (nfit + 1) // 2
        if blocks.shape != (n_pairs, 2):
            return None
        ref = np.stack(np.tril_indices(nfit), axis=1).astype(np.int32) + ncombined
        if not np.array_equal(blocks, ref):
            return None
    else:
        n_orb_pairs = ncombined * (ncombined + 1) // 2
        if blocks.shape != (nfit * n_orb_pairs, 3):
            return None
        # Check canonical layout quickly without materializing a full 3D copy.
        b3 = blocks.reshape(nfit, n_orb_pairs, 3)
        ref_pairs = np.stack(np.tril_indices(ncombined), axis=1).astype(np.int32)
        if not np.array_equal(b3[0, :, :2], ref_pairs):
            return None
        if not np.all(b3[:, :, :2] == b3[0:1, :, :2]):
            return None
        expected_aux = np.arange(ncombined, nshell, dtype=np.int32)[:, None]
        if not np.all(b3[:, :, 2] == expected_aux):
            return None

    table = np.zeros(_TRANSFORM_TABLE_SIZE, dtype=np.float64)
    for l in np.unique(angular):
        li = int(l)
        key = f"transform_{li}"
        if key not in layout:
            return None
        mat = np.ascontiguousarray(layout[key], dtype=np.float64)
        expected_shape = (2 * li + 1, (li + 1) * (li + 2) // 2)
        if mat.shape != expected_shape or not np.isfinite(mat).all():
            return None
        off = _TRANSFORM_OFFSETS[li]
        table[off : off + mat.size] = mat.ravel()

    return {
        "nshell": nshell,
        "orbital_shell_count": ncombined,
        "norb_sph": norb,
        "naux_sph": naux,
        "sph_offsets": sph_offsets,
        "transform_table": table,
        "nblocks": len(blocks),
    }


def channel_name(group: str, operator: str) -> str:
    if operator not in _OPCODES or group not in ("metric", "ao"):
        raise ValueError(f"unknown integral channel {group}/{operator}")
    return (
        "metric"
        if group == "metric" and operator == "coulomb"
        else group + "_" + operator
    )


def contract_geometry(
    fixture: Mapping[str, np.ndarray],
    manifest: dict,
    functions: Mapping[str, Callable],
    group: str,
    operator: str,
    budget: int,
    host_reference: bool,
    *,
    slater_normalization: str = DEFAULT_SLATER_FIT_NORMALIZATION,
    extended_abi: bool = False,
) -> tuple[str, dict]:
    """Execute one physical AO channel from shells, centers and Cartesian adjoints.

    Non-`_ex` geminal entrypoints (`extended_abi=False`) compile the default
    `negative_exponential_over_beta` (`norm_code=1`) normalization; non-default
    normalizations such as `negative_exponential` (`norm_code=0`) require `*_ex`.
    """
    label = channel_name(group, operator)
    norm_code = slater_normalization_code(slater_normalization)
    t0 = time.monotonic()
    blocks = fixture[group + "_blocks"]
    offsets = fixture[group + "_offsets"]
    weights = fixture["weights_" + label]
    angular = fixture["angular"]
    atom = fixture["atom"]
    exponents = fixture["exponents"]
    centers = fixture["centers"]
    natoms = len(manifest["elements"])
    nblocks, ncenter = blocks.shape
    if (
        offsets.shape != (nblocks + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(weights)
    ):
        raise ValueError(f"incomplete Cartesian shell offsets for {label}")
    if ncenter != (2 if group == "metric" else 3):
        raise ValueError(f"invalid number of shell centers for {label}")
    geometry = np.asarray(manifest["geometry_bohr"])
    if (
        geometry.shape != (natoms, 3)
        or not np.isfinite(geometry).all()
        or atom.shape != (len(angular),)
        or np.any(atom < 0)
        or np.any(atom >= natoms)
        or not np.array_equal(geometry[atom], centers)
    ):
        raise ValueError("inconsistent shell and atom geometry")
    output = np.full((nblocks, 10), np.nan)
    size = ct.c_uint64()
    kernel = "coulomb" if operator == "coulomb" else "geminal"
    before_kernel = time.monotonic()
    beta = 0.0 if operator == "coulomb" else manifest["options"]["f12_beta"]
    if extended_abi:
        status = functions[kernel](
            ncenter,
            _OPCODES[operator],
            beta,
            norm_code,
            len(angular),
            len(exponents),
            nblocks,
            len(weights),
            angular,
            fixture["prim_offsets"],
            exponents,
            fixture["radial"],
            centers,
            blocks,
            offsets,
            weights,
            output,
            budget,
            ct.byref(size),
        )
    else:
        if norm_code != LEGACY_SLATER_NORM_CODE and operator != "coulomb":
            raise RuntimeError(
                f"Configured SlaterFitNormalization={slater_normalization!r} (norm_code={norm_code}) "
                f"differs from the non-_ex default {LEGACY_SLATER_FIT_NORMALIZATION!r} "
                f"(norm_code={LEGACY_SLATER_NORM_CODE}) and requires an integral sink library "
                "exporting *_ex or *_spherical_* symbols"
            )
        status = functions[kernel](
            ncenter,
            _OPCODES[operator],
            beta,
            len(angular),
            len(exponents),
            nblocks,
            len(weights),
            angular,
            fixture["prim_offsets"],
            exponents,
            fixture["radial"],
            centers,
            blocks,
            offsets,
            weights,
            output,
            budget,
            ct.byref(size),
        )
    kernel_seconds = time.monotonic() - before_kernel
    if status:
        raise RuntimeError(f"{kernel} {label} returned status {status}")
    if not np.isfinite(output).all():
        raise ValueError(f"non-finite or missing shell output for {label}")
    # Metric stores centers A and C. Three-center stores A, B, C.
    if ncenter == 2 and np.any(output[:, 4:7] != 0):
        raise ValueError("two-center derivative has nonzero absent B center")
    xyz_offsets = (1, 7) if ncenter == 2 else (1, 4, 7)
    gradient = np.zeros((natoms, 3))
    for j, start in enumerate(xyz_offsets):
        np.add.at(gradient, atom[blocks[:, j]], output[:, start : start + 3])
    value = float(output[:, 0].sum())
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError(f"non-finite molecular value or gradient for {label}")
    translation = float(np.max(np.abs(gradient.sum(axis=0))))
    return label, {
        "contraction_checks_passed": bool(
            translation <= _TRANSLATION_TOLERANCE
            and size.value <= budget
            and (host_reference or size.value > 0)
        ),
        "weighted_value_Eh": value,
        "gradient_Eh_per_bohr": gradient.tolist(),
        "max_translation_residual_Eh_per_bohr": translation,
        "shell_blocks": nblocks,
        "cartesian_weights": len(weights),
        "seconds": time.monotonic() - t0,
        "kernel_and_transfer_seconds": kernel_seconds,
        "device_bytes": size.value,
    }


def compare_reference(label: str, row: dict, reference: dict) -> dict:
    """Validate one computed result against independent signed value and gradient."""
    if (
        row["shell_blocks"] != reference["shell_blocks"]
        or row["cartesian_weights"] != reference["cartesian_weights"]
    ):
        raise ValueError(f"incomplete shell blocks for {label}")
    energy = float(reference["weighted_value_Eh"])
    target = np.asarray(reference["gradient_Eh_per_bohr"], dtype=np.float64)
    gradient = np.asarray(row["gradient_Eh_per_bohr"])
    if (
        not np.isfinite(energy)
        or target.shape != gradient.shape
        or not np.isfinite(target).all()
    ):
        raise ValueError(f"invalid independent reference for {label}")
    de = abs(row["weighted_value_Eh"] - energy)
    dg = float(np.max(np.abs(gradient - target)))
    return {
        **row,
        "reference_status": "compared",
        "weighted_value_error_Eh": de,
        "max_gradient_error_Eh_per_bohr": dg,
        "passed": row["contraction_checks_passed"]
        and de <= _VALUE_TOLERANCE
        and dg <= _GRADIENT_TOLERANCE,
    }
