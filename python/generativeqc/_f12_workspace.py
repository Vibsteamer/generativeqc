"""Integer workspace sizes at the F12 native ABI boundary."""

from operator import index


def workspace_bytes(value: int, *, signed: bool) -> int:
    size = index(value)
    maximum = (1 << (63 if signed else 64)) - 1
    if not 0 <= size <= maximum:
        raise ValueError(f"workspace budget must be between 0 and {maximum} bytes")
    return size
