"""Auxiliary tiling for the extended spin Fock source."""

_TARGET_AUX_SLAB_BYTES = 256 << 20


def default_auxiliary_batch_size(
    auxiliaries: int, extent_a: int, extent_b: int | None = None
) -> int:
    """Choose an auxiliary chunk size keeping per-slab 3-center tensors <= 256 MiB."""
    if auxiliaries <= 1:
        return max(1, auxiliaries)
    eb = extent_a if extent_b is None else extent_b
    slice_bytes = max(1, extent_a * eb * 8)
    return max(1, min(auxiliaries, _TARGET_AUX_SLAB_BYTES // slice_bytes))


def _tile_groups(
    auxiliaries: int, width: int
) -> list[tuple[int, list[slice]]]:
    full = [slice(s, s + width) for s in range(0, (auxiliaries // width) * width, width)]
    groups: list[tuple[int, list[slice]]] = []
    if full:
        groups.append((width, full))
    rem_start = (auxiliaries // width) * width
    if rem_start < auxiliaries:
        groups.append((auxiliaries - rem_start, [slice(rem_start, auxiliaries)]))
    return groups
