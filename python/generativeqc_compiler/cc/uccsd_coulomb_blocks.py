"""Spin and orbital ordering of antisymmetrized UCC Hamiltonian inputs."""

import typing
from dataclasses import dataclass


@dataclass(frozen=True)
class SpinMOBlock:
    """Chemists-ERI MO slots with an explicit alpha/beta owner per axis."""

    slots: tuple[tuple[int, ...], ...]
    spins: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(tuple(x) for x in self.slots))
        object.__setattr__(self, "spins", tuple(self.spins))
        if len(self.slots) != 4 or len(self.spins) != 4:
            raise ValueError("spin-resolved chemists' ERIs require four slots/spins")
        if any(spin not in ("alpha", "beta") for spin in self.spins):
            raise ValueError("MO slot spin must be alpha or beta")
        for slot in self.slots:
            if len(set(slot)) != len(slot) or any(
                type(i) is not int or i < 0 for i in slot
            ):
                raise ValueError("MO slots require unique nonnegative integer indices")

    @property
    def shape(self) -> typing.Any:
        return tuple(map(len, self.slots))

    def validate(self, snapshot: typing.Any) -> None:
        if any(i >= snapshot.nmo for slot in self.slots for i in slot):
            raise ValueError("MO index outside the unrestricted reference snapshot")
        for spin in set(self.spins):
            coefficients = getattr(snapshot, f"coefficients_{spin}", None)
            if coefficients is None or coefficients.shape != (
                snapshot.nmo,
                snapshot.nmo,
            ):
                raise ValueError(f"reference lacks canonical {spin} coefficients")


_COULOMB_PERMUTATIONS = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)


def _canonical_coulomb(block: SpinMOBlock) -> tuple[SpinMOBlock, tuple[int, ...]]:
    """Use only chemists' eightfold Coulomb symmetry, including spin owners."""
    permutation = min(
        _COULOMB_PERMUTATIONS,
        key=lambda axes: tuple((block.spins[axis], block.slots[axis]) for axis in axes),
    )
    canonical = SpinMOBlock(
        tuple(block.slots[axis] for axis in permutation),
        tuple(block.spins[axis] for axis in permutation),
    )
    return canonical, tuple(permutation.index(axis) for axis in range(4))


def antisymmetrized_integral_requests(
    slots: tuple[tuple[int, ...], ...], spins: tuple[str, ...]
) -> tuple:
    """Bind direct and exchange legs to their canonical chemists' blocks."""
    contributions = []
    for coefficient, axes, output_axes in (
        (1, (0, 2, 1, 3), (0, 2, 1, 3)),
        (-1, (0, 3, 1, 2), (0, 2, 3, 1)),
    ):
        if spins[axes[0]] != spins[axes[1]] or spins[axes[2]] != spins[axes[3]]:
            continue
        block = SpinMOBlock(
            tuple(slots[axis] for axis in axes), tuple(spins[axis] for axis in axes)
        )
        canonical, restore = _canonical_coulomb(block)
        contributions.append(
            (coefficient, canonical, tuple(restore[axis] for axis in output_axes))
        )
    return tuple(contributions)
