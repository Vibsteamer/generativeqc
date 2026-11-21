"""Bounded unrestricted separable-Fock resolvent vectors and paired moments.

The canonical X/Y vectors share the actual nine-term triples equation builder.
Only bounded rank-six tiles are exposed. Native paired contractions leave
one same-spin axis open on each side; they never divide internal orbital gaps.
These mathematical frontends do not by themselves qualify a force endpoint.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import TYPE_CHECKING

from generativeqc_compiler.cc.uccsd_triples import (
    _build_tile_program,
    channel_shape,
)
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
)

if TYPE_CHECKING:
    from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
    from generativeqc_compiler.cc.uccsd_triples import TriplesTileProgram
    from generativeqc_compiler.tensor import Node


def build_tile_resolvent_program(
    layout: UCCSDLayout, channel: str, shape: tuple[int, ...], *, lanes: int | None = None
) -> TriplesTileProgram:
    """Return the scalar energy and bounded X=C/D, Y=(C+V)/D outputs.

    Operand bindings and denominator conventions are identical to the energy
    tile. X/Y contain no channel multiplicity; paired moment programs include
    that factor exactly once. The caller owns complete paired-grid admission,
    fresh parent qualification and at-most-two-page vector retention.
    """
    return _build_tile_program(layout, channel, shape, resolvents=True, lanes=lanes)


def _axis_moment(
    channel: str,
    axis: int,
    equation: str,
    x_left: Node,
    y_left: Node,
    x_right: Node,
    y_right: Node,
) -> Program:
    domain, spin = ("o" if axis < 3 else "v"), (channel * 2)[axis]
    multiplicity = Fraction(comb(3, channel.count("a")) ** 2, 36)
    factor = (-1 if axis < 3 else 1) * multiplicity / 2
    return Program(
        {
            f"f{domain * 2}_{spin * 2}": add(
                einsum(equation, x_left, y_right),
                einsum(equation, y_left, x_right),
                coefficients=(factor, factor),
            )
        },
        provenance={
            "method": "UCCSD(T)",
            "spin_channel": channel,
            "resolvent_axis": axis,
            "spin_multiplicity": (multiplicity.numerator, multiplicity.denominator),
            "contract": "symmetric complete same-space separable Fock resolvent moment",
        },
    )


def _vector(name: str, indices: tuple[Index, ...]) -> Node:
    return input_tensor(
        name,
        TensorSpec(
            indices,
            representation="spin_orbital",
            role="parameter",
            differentiable=True,
        ),
    )


def build_axis_moment_program(
    layout: UCCSDLayout,
    channel: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    axis: int,
) -> Program:
    """Contract one paired tile into a symmetric oo/vv Fock matrix block.

    Inputs are the actual canonical X=C/D and Y=(C+V)/D for one conserved
    spin channel. Five axes describe identical physical coordinates in both
    pages; the selected axis has independently paged left/right coordinates.
    The caller enforces that coordinate identity and scatters off-diagonal
    page pairs to both transposed matrix blocks exactly once. Channel factors
    and occupied-minus/virtual-plus signs are included here, not at scatter.
    The diagonal replaces, rather than supplements, the eps-to-Fock chain.
    """
    if type(axis) is not int or not 0 <= axis < 6:
        raise ValueError("triples Fock moment axis must be an integer in [0, 6)")
    full_shape = channel_shape(layout, channel)
    for shape in (left_shape, right_shape):
        if (
            not isinstance(shape, tuple)
            or len(shape) != 6
            or any(
                type(size) is not int or not 0 < size <= maximum
                for size, maximum in zip(shape, full_shape, strict=True)
            )
        ):
            raise ValueError(
                "triples Fock moment pages require bounded rank-six shapes"
            )
    if any(
        left_shape[position] != right_shape[position]
        for position in range(6)
        if position != axis
    ):
        raise ValueError("triples Fock moment pages differ on a contracted axis")
    left_indices = tuple(
        Index(
            label,
            IndexSpace(
                f"moment_{domain}{spin}_{label}",
                "occupied" if domain == "o" else "virtual",
                size,
                "alpha" if spin == "a" else "beta",
            ),
        )
        for label, domain, spin, size in zip(
            "ijkabc", "ooovvv", channel * 2, left_shape, strict=True
        )
    )
    domain, spin = ("o" if axis < 3 else "v"), (channel * 2)[axis]
    right_indices = tuple(
        Index(
            "p",
            IndexSpace(
                f"moment_{domain}{spin}_p",
                "occupied" if domain == "o" else "virtual",
                right_shape[axis],
                "alpha" if spin == "a" else "beta",
            ),
        )
        if position == axis
        else index
        for position, index in enumerate(left_indices)
    )

    x_left, y_left = (_vector(name, left_indices) for name in ("x_left", "y_left"))
    x_right, y_right = (_vector(name, right_indices) for name in ("x_right", "y_right"))
    left_labels = "ijkabc"
    right_labels = left_labels[:axis] + "p" + left_labels[axis + 1 :]
    equation = f"{left_labels},{right_labels}->{left_labels[axis]}p"
    return _axis_moment(channel, axis, equation, x_left, y_left, x_right, y_right)


def build_axis_moment_panel_program(
    layout: UCCSDLayout, channel: str, axis: int, lanes: int
) -> Program:
    """Contract a bounded panel of shared complements over a full open axis."""
    if type(axis) is not int or not 0 <= axis < 6:
        raise ValueError("triples Fock moment axis must be an integer in [0, 6)")
    if type(lanes) is not int or lanes < 1:
        raise ValueError("triples Fock moment panel requires positive lanes")
    extent = channel_shape(layout, channel)[axis]
    spin = "alpha" if (channel * 2)[axis] == "a" else "beta"
    domain = "occupied" if axis < 3 else "virtual"
    orbital = Index("i", IndexSpace("triples_panel_orbital", domain, extent, spin))
    complement = Index("q", IndexSpace("triples_panel_complement", "batch", lanes))
    x, y = (_vector(name, (orbital, complement)) for name in ("x", "y"))
    return _axis_moment(channel, axis, "iq,jq->ij", x, y, x, y)
