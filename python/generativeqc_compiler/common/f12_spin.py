"""Spin labels and frame validation shared by unrestricted F12 equations."""


def pair_frame_spins(
    left: tuple[int, int, int], right: tuple[int, int, int], channel: str
) -> tuple[str, str]:
    """Return the ordered spin labels for two occupied/virtual/CABS frames."""
    spins = {
        "alpha_alpha": ("alpha", "alpha"),
        "alpha_beta": ("alpha", "beta"),
        "beta_beta": ("beta", "beta"),
    }
    if channel not in spins or any(
        len(space) != 3 or any(type(n) is not int or n < 1 for n in space)
        for space in (left, right)
    ):
        raise ValueError("Invalid spin channel or occupied/virtual/CABS dimensions")
    if channel != "alpha_beta" and left != right:
        raise ValueError(
            "Both electrons of a same-spin pair need the same orbital frame"
        )
    return spins[channel]
