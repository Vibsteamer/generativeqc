"""Spin and exterior-product coordinates for connected triples and their duals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb, factorial, prod

from generativeqc_compiler.tensor.types import (
    Index,
    IndexSpace,
    Symmetry,
    TensorSpec,
    checked_size,
)

from .uccsd_layout import UCCSDLayout
from .uccsd_triples import CHANNELS

_SPINS = {"a": "alpha", "b": "beta"}
_PARTICLE_ORDER = {"v": 0, "c": 1}


def _axis_groups(domains: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(axis for axis, domain in enumerate(domains) if domain == group)
        for group in dict.fromkeys(domains)
    )


def _permutation_sign(order: tuple[int, ...]) -> int:
    inversions = sum(
        left > right for axis, left in enumerate(order) for right in order[axis + 1 :]
    )
    return (-1) ** inversions


@dataclass(frozen=True)
class TriplesSpinBlock:
    """One canonical row with `ijkabc` axes in a conserved spin channel.

    Particles in orbital virtuals (`v`) and the complementary space (`c`)
    have separate index identities, even when their dimensions are equal.
    The equation owner selects the particle-space sectors it needs.
    """

    layout: ExplicitTriplesSpinLayout
    channel: str
    particle_domains: str

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise ValueError("unknown spin-conserving triples channel")
        if (
            not isinstance(self.particle_domains, str)
            or len(self.particle_domains) != 3
            or any(domain not in _PARTICLE_ORDER for domain in self.particle_domains)
            or any(
                _PARTICLE_ORDER[self.particle_domains[i]]
                > _PARTICLE_ORDER[self.particle_domains[i + 1]]
                for i in range(2)
                if self.channel[i] == self.channel[i + 1]
            )
        ):
            raise ValueError("particle domains must be ordered within each spin")

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(
            domain + spin
            for domain, spin in zip(
                "ooo" + self.particle_domains, self.channel * 2, strict=True
            )
        )

    @property
    def spec(self) -> TensorSpec:
        """Return a differentiable spin-orbital feed with signed exchange axes."""
        domains = self.domains
        axes = tuple(
            Index(
                label,
                IndexSpace(
                    name,
                    "occupied" if name[0] == "o" else "virtual",
                    self.layout.population(name),
                    _SPINS[name[1]],
                ),
            )
            for label, name in zip("ijkabc", domains, strict=True)
        )
        symmetries = []
        for group in _axis_groups(domains):
            for left, right in zip(group, group[1:]):
                swap = list(range(6))
                swap[left], swap[right] = swap[right], swap[left]
                symmetries.append(Symmetry(tuple(swap), -1))
        return TensorSpec(
            axes,
            symmetries=tuple(symmetries),
            representation="spin_orbital",
            role="parameter",
            differentiable=True,
        )

    @property
    def independent_size(self) -> int:
        """Count Pauli-allowed representatives without materializing a tensor."""
        return prod(
            comb(self.layout.population(domain), count)
            for domain, count in Counter(self.domains).items()
        )

    @property
    def ordered_prefactor(self) -> Fraction:
        """Weight the ordered contraction of two antisymmetrized tensors."""
        return Fraction(
            1, prod(factorial(count) for count in Counter(self.domains).values())
        )


@dataclass(frozen=True)
class SignedTriplesCoordinate:
    """Canonical indices with a sign that also applies to reverse scatter."""

    block: TriplesSpinBlock
    occupied: tuple[int, ...]
    particles: tuple[int, ...]
    sign: int


@dataclass(frozen=True)
class ExplicitTriplesSpinLayout:
    """An unrestricted reference with separate alpha/beta complementary ranks."""

    reference: UCCSDLayout
    complement: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if not isinstance(self.reference, UCCSDLayout):
            raise TypeError("explicit triples require a UCCSD reference layout")
        complement = tuple(self.complement)
        if len(complement) != 2:
            raise ValueError("complement requires one rank per spin")
        for n in complement:
            checked_size(n, "complement rank")
        object.__setattr__(self, "complement", complement)

    def population(self, domain: str) -> int:
        if domain == "ca":
            return self.complement[0]
        if domain == "cb":
            return self.complement[1]
        if domain in ("oa", "ob", "va", "vb"):
            return getattr(self.reference, domain)
        raise ValueError("unknown triples spin domain")

    def blocks(self, complement_legs: int = 0) -> tuple[TriplesSpinBlock, ...]:
        """Enumerate unique typed blocks in one chosen complementary sector."""
        if type(complement_legs) is not int or not 0 <= complement_legs <= 3:
            raise ValueError("a triples sector requires zero to three complementary legs")
        return tuple(
            TriplesSpinBlock(self, channel, "".join(domains))
            for channel in CHANNELS
            for domains in product(_PARTICLE_ORDER, repeat=3)
            if domains.count("c") == complement_legs
            and all(
                _PARTICLE_ORDER[domains[i]] <= _PARTICLE_ORDER[domains[i + 1]]
                for i in range(2)
                if channel[i] == channel[i + 1]
            )
        )

    def coordinate(
        self,
        occupied_spins: str,
        occupied: tuple[int, int, int],
        particle_spins: str,
        particles: tuple[int, int, int],
        particle_domains: str = "vvv",
    ) -> SignedTriplesCoordinate | None:
        """Sort each determinant independently and return its exchange phase.

        A spin-changing excitation or a repeated spin orbital is a structural
        zero. Spatial coincidences across spins or virtual spaces are allowed.
        """
        if (
            any(
                not isinstance(labels, str) or len(labels) != 3
                for labels in (occupied_spins, particle_spins, particle_domains)
            )
            or any(spin not in _SPINS for spin in occupied_spins + particle_spins)
            or any(domain not in _PARTICLE_ORDER for domain in particle_domains)
            or len(occupied) != 3
            or len(particles) != 3
        ):
            raise ValueError(
                "triples coordinates require three typed holes and three typed particles"
            )

        holes = tuple(zip(occupied_spins, "ooo", occupied, strict=True))
        virtuals = tuple(zip(particle_spins, particle_domains, particles, strict=True))
        if any(
            type(i) is not int or not 0 <= i < self.population(domain + spin)
            for spin, domain, i in holes + virtuals
        ):
            raise ValueError("triples coordinate out of range for its spin space")
        if (
            sorted(occupied_spins) != sorted(particle_spins)
            or len(set(holes)) != 3
            or len(set(virtuals)) != 3
        ):
            return None

        hole_order = tuple(sorted(range(3), key=holes.__getitem__))
        particle_order = tuple(
            sorted(
                range(3),
                key=lambda i: (
                    virtuals[i][0],
                    _PARTICLE_ORDER[virtuals[i][1]],
                    virtuals[i][2],
                ),
            )
        )
        block = TriplesSpinBlock(
            self,
            "".join(occupied_spins[i] for i in hole_order),
            "".join(particle_domains[i] for i in particle_order),
        )
        return SignedTriplesCoordinate(
            block,
            tuple(occupied[i] for i in hole_order),
            tuple(particles[i] for i in particle_order),
            _permutation_sign(hole_order) * _permutation_sign(particle_order),
        )
