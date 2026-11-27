"""Slater fitting convention for focal reference requests."""

SLATER_FIT_NORMALIZATIONS = {
    "negative_exponential": 0,
    "negative_exponential_over_beta": 1,
}
DEFAULT_SLATER_FIT_NORMALIZATION = "negative_exponential_over_beta"


def slater_normalization_code(normalization: str) -> int:
    if normalization not in SLATER_FIT_NORMALIZATIONS:
        raise ValueError(
            f"unknown Slater fit normalization {normalization!r}; "
            f"expected one of {sorted(SLATER_FIT_NORMALIZATIONS)}"
        )
    return SLATER_FIT_NORMALIZATIONS[normalization]
