"""Generate two-/three-center geminal values and nuclear derivatives.

The shared DF angular DAG includes the Gaussian pair decay and physical center
derivatives. Its radial leaves are Hermite moments of the fitted geminal. For
each operator those moments obey dM_n/dT = -M_(n+1), as required by that DAG.
Basis normalization and contraction coefficients belong to the native caller.
"""

from __future__ import annotations

import re

from .df_derivatives import build_df_derivative_ir, build_df_derivative_kernel
from .scalar_c import ScalarCEmitter


def emit_f12_df_component(components: tuple[str, ...], name: str) -> str:
    """Emit one FP64 host/device primitive, with value then center/xyz outputs.

    ``exponents`` and ``centers`` use compact physical-center order. A metric
    has two auxiliary centers. A three-center integral has two orbital centers
    followed by one auxiliary center. The fitted Slater parameters are prepared
    once by the caller and may represent different left/right beta values.

    The consumer includes ``integrals/f12_geminal.hpp`` and ``<cmath>`` and
    defines ``GQC_F12_COMPONENT_HD`` as host/device qualifiers when appropriate.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError("expected a C++ function identifier")
    if len(components) not in (2, 3):
        raise ValueError("F12 DF primitives need two or three physical centers")
    family = "coulomb_metric" if len(components) == 2 else "three_center_eri"
    angular = tuple(len(component) for component in components)
    integral = build_df_derivative_ir(family, angular)
    kernel = build_df_derivative_kernel(integral, components)
    variables = {f"exponent_{i}": f"exponents[{i}]" for i in range(len(components))}
    variables.update(
        {
            f"center_{i}_{axis}": f"centers[{3 * i + j}]"
            for i in range(len(components))
            for j, axis in enumerate("xyz")
        }
    )
    variables.update(
        {f"boys_{n}": f"geometry.boys[{n}]" for n in range(kernel.boys_count)}
    )
    roots = (kernel.value, *(root for axes in kernel.gradients for root in axes))
    emitter = ScalarCEmitter(kernel.graph, variables)
    emitter.emit(roots)
    # Missing products are explicit unit Gaussian slots in the existing radial
    # provider. The angular DAG contains only the physical basis factors.
    slots = (0, None, 1, None) if len(components) == 2 else (0, 1, 2, None)
    expanded_exponents = [
        "0.0" if slot is None else f"exponents[{slot}]" for slot in slots
    ]
    expanded_centers = [
        "0.0" if slot is None else f"centers[{3 * slot + axis}]"
        for slot in slots
        for axis in range(3)
    ]
    unit_mask = sum(1 << i for i, slot in enumerate(slots) if slot is None)
    order = kernel.boys_count - 1
    lines = [
        f"GQC_F12_COMPONENT_HD inline bool {name}(",
        "    const double* exponents, const double* centers,",
        "    generativeqc::integrals::GeminalProduct op,",
        "    const generativeqc::integrals::FittedSlater& left,",
        "    const generativeqc::integrals::FittedSlater& right, double* output) {",
        "  struct Geometry {",
        "    double inverse_two_p, inverse_two_q, rho, difference[3], shifts[4][3];",
        "    double decay[4][3], product_scales[4], prefactor;",
        f"    double boys[{kernel.boys_count}];",
        "  } geometry;",
        "  const double expanded_exponents[4] = {"
        + ", ".join(expanded_exponents)
        + "};",
        "  const double expanded_centers[12] = {" + ", ".join(expanded_centers) + "};",
        "  if (!generativeqc::integrals::make_fitted_geminal_geometry<",
        f"          Geometry, {order}, {unit_mask}>(expanded_exponents, expanded_centers,",
        f"          {order}, op, left, right, geometry)) return false;",
        *emitter.lines,
        *(
            f"  output[{i}] = {emitter.names[root.identifier]};"
            for i, root in enumerate(roots)
        ),
        "  return true;",
        "}",
    ]
    return "\n".join(lines) + "\n"
