"""Runtime-resolvable self annotations for the supported Python versions."""

import sys

__all__ = ["Self"]

if sys.version_info >= (3, 11):
    from typing import Self
else:
    try:
        from typing_extensions import Self
    except ModuleNotFoundError:
        from typing import TypeVar

        # Minimal 3.10 environments can still import and introspect batch types.
        Self = TypeVar("Self")
