"""Bounded contraction of generated F12 Cartesian values and center derivatives."""

from __future__ import annotations

import re

from .f12_df import _emit_f12_df_angular, _emit_f12_df_geometry


def emit_f12_df_contracted(components: tuple[tuple[str, ...], ...], name: str) -> str:
    """Emit a scalar contracted Cartesian/spherical projection and its gradient.

    Each entry is one Cartesian product in the same shell signature. The caller
    supplies its linear projection coefficient, and normalized primitive radial
    coefficients for each center. Primitive arrays are concatenated by center.
    No normalization, basis truncation or pure-function convention is inferred.

    The explicit budget counts primitive Cartesian evaluations, including the
    projection terms. Only ten accumulated scalars are kept, never a molecular
    coordinate-major primitive derivative tensor. This is a bounded scalar
    source consumer. Each primitive combination prepares radial moments once
    for all Cartesian projection terms. The projection may cover a full shell;
    no coordinate-major derivative tensor or molecular integral is retained.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError("expected a C++ function identifier")
    if not components or len(set(components)) != len(components):
        raise ValueError("expected distinct Cartesian projection terms")
    centers = len(components[0])
    angular = tuple(map(len, components[0]))
    if centers not in (2, 3) or any(
        tuple(map(len, term)) != angular for term in components
    ):
        raise ValueError(
            "all projection terms must have the same two/three-center shell signature"
        )
    angular_sources = [
        _emit_f12_df_angular(term, f"{name}_cartesian_{i}")
        for i, term in enumerate(components)
    ]
    boys_count = angular_sources[0][1]
    if any(count != boys_count for _, count in angular_sources):
        raise ValueError("projection terms require the same radial moment order")
    fields = 1 + 3 * centers
    terms = len(components)
    calls = []
    for i in range(terms):
        calls.extend(
            [
                f"    if (projection[{i}] != 0.) {{",
                f"      if (!{name}_cartesian_{i}(alpha, centers, geometry.boys, value)) return false;",
                f"      const double weight = radial * projection[{i}];",
                "      if (!std::isfinite(weight)) return false;",
                f"      for (unsigned f = 0; f < {fields}; ++f) {{",
                "        const double addend = weight * value[f] - correction[f];",
                "        const double next = sum[f] + addend;",
                "        correction[f] = (next - sum[f]) - addend;",
                "        sum[f] = next;",
                "      }",
                "    }",
            ]
        )
    return "\n".join(
        [
            *(source for source, _ in angular_sources),
            f"GQC_F12_COMPONENT_HD inline bool {name}(",
            "    const int* counts, const double* exponents, const double* coefficients,",
            "    const double* centers, const double* projection, std::size_t max_components,",
            "    generativeqc::integrals::GeminalProduct op,",
            "    const generativeqc::integrals::FittedSlater& left,",
            "    const generativeqc::integrals::FittedSlater& right, double* output) {",
            "  if (!counts || !exponents || !coefficients || !centers || !projection || !output)",
            "    return false;",
            f"  std::size_t combinations = 1, begin[{centers}]{{}}, total = 0;",
            f"  if (max_components < {terms}) return false;",
            f"  const std::size_t combination_limit = max_components / {terms};",
            f"  for (unsigned c = 0; c < {centers}; ++c) {{",
            "    if (counts[c] <= 0 || std::size_t(counts[c]) > combination_limit / combinations)",
            "      return false;",
            "    combinations *= counts[c];",
            "    begin[c] = total;",
            "    if (std::size_t(counts[c]) > SIZE_MAX - total) return false;",
            "    total += counts[c];",
            "  }",
            "  for (std::size_t p = 0; p < total; ++p)",
            "    if (!std::isfinite(exponents[p]) || exponents[p] <= 0. ||",
            "        !std::isfinite(coefficients[p])) return false;",
            f"  for (unsigned c = 0; c < {3 * centers}; ++c)",
            "    if (!std::isfinite(centers[c])) return false;",
            "  bool active = false;",
            f"  for (unsigned t = 0; t < {terms}; ++t) {{",
            "    if (!std::isfinite(projection[t])) return false;",
            "    active = active || projection[t] != 0.;",
            "  }",
            f"  double sum[{fields}]{{}}, correction[{fields}]{{}}, value[{fields}], alpha[{centers}];",
            "  for (std::size_t p = 0; p < combinations; ++p) {",
            "    std::size_t remaining = p;",
            "    double radial = 1.;",
            f"    for (unsigned c = 0; c < {centers}; ++c) {{",
            "      const auto index = begin[c] + remaining % counts[c];",
            "      remaining /= counts[c];",
            "      alpha[c] = exponents[index];",
            "      radial *= coefficients[index];",
            "    }",
            "    if (!std::isfinite(radial)) return false;",
            "    if (!active) continue;",
            "    {",
            "    const double* exponents = alpha;",
            *_emit_f12_df_geometry(centers, boys_count),
            *calls,
            "    }",
            "  }",
            f"  for (unsigned f = 0; f < {fields}; ++f)",
            "    if (!std::isfinite(sum[f])) return false;",
            f"  for (unsigned f = 0; f < {fields}; ++f) output[f] = sum[f];",
            "  return true;",
            "}",
            "",
        ]
    )
