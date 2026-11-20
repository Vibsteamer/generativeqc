"""Project unrestricted CC Fock weights onto geometric core and DF Coulomb/exchange sources."""

from collections.abc import Callable, Mapping

import numpy as np
from generativeqc._f12_focal_one_electron import FocalOneElectron
from generativeqc._f12_spin_ao import _orbital_sources
from generativeqc._f12_spin_fock import spin_fock_pullback
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
)


def _spin_fock_weight(
    weights: Mapping[str, np.ndarray], spin: str, nmo: int, no: int, nf: int
) -> np.ndarray:
    if not 0 <= nf <= no <= nmo:
        raise ValueError("Invalid frozen or occupied spin partition")
    tag = spin[0] * 2
    bar = np.zeros((nmo, nmo))
    slices = {"o": slice(nf, no), "v": slice(no, nmo)}
    for block in ("oo", "ov", "vv"):
        name = "f" + block + "_" + tag
        dst = bar[slices[block[0]], slices[block[1]]]
        value = weights[name]
        if value.dtype != np.float64 or value.shape != dst.shape:
            raise ValueError(f"Inconsistent ordinary Fock cotangent {name}")
        dst[...] = value
    # The Fock and Hcore tangents are symmetric. The ov input has a single
    # copy, so averaging distributes half its weight to each transpose.
    return (bar + bar.T) / 2


def unrestricted_cc_fock_nuclear_forces(
    occupied: tuple[int, int],
    weights: Mapping[str, np.ndarray],
    coefficients: Mapping[str, np.ndarray],
    coefficient_derivatives: Mapping[str, np.ndarray],
    coulomb_exchange_derivatives: Mapping[str, np.ndarray],
    one_electron: FocalOneElectron,
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> dict[str, np.ndarray]:
    """Return the complete ordinary Fock contribution in Eh/bohr.

    Pass the result of one_electron_sources(request, derivatives=True).
    AO coefficients and the native Hcore/derivatives use the same real spherical
    AO order. The MO columns include frozen occupied orbitals and all ordinary
    virtuals. The last spin-dependent input is the full moving-MO derivative
    of the Coulomb and exchange part of F. The six live CC Fock cotangents
    then act on d(C.T H C) and d(J-K) once per coordinate.
    """
    hcore = one_electron.hcore
    hcore_derivatives = one_electron.d_hcore
    if hcore_derivatives is None or hcore_derivatives.ndim != 4 or hcore.ndim != 2:
        raise ValueError("Expected an AO core Hamiltonian and atom-axis derivatives")
    natoms, ndim, nao, ncol = hcore_derivatives.shape
    if natoms < 1 or ndim != 3 or nao < 1 or ncol != nao or hcore.shape != (nao, nao):
        raise ValueError("Inconsistent AO/core derivative dimensions")
    atom = Index("A", IndexSpace("atom", "atom", natoms))
    axis = Index("k", IndexSpace("cart", "cartesian", 3))
    aos = IndexSpace("ao", "ao", nao)
    u, v = (Index(n, aos) for n in ("u", "v"))

    def field(name: str, *indices: Index) -> Node:
        return input_tensor(name, TensorSpec(indices, role="input"))

    h = field("hcore", u, v)
    dh = field("d_hcore", atom, axis, u, v)
    fields = {"hcore": hcore, "d_hcore": hcore_derivatives}
    ao_weights, frame_actions, jk_actions = [], [], []
    for spin, no, nf in zip(("alpha", "beta"), occupied, frozen_occupied, strict=True):
        c = coefficients[spin]
        if (
            c.shape != (nao, one_electron.obs_ao_count)
            or not 0 <= nf <= no <= c.shape[1]
        ):
            raise ValueError("Inconsistent spin MO coefficients or occupied partition")
        nmo = c.shape[1]
        fields["bar_" + spin] = _spin_fock_weight(weights, spin, nmo, no, nf)
        fields["c_" + spin] = c
        fields["dc_" + spin] = coefficient_derivatives[spin]
        fields["djk_" + spin] = coulomb_exchange_derivatives[spin]

        mos = IndexSpace("mo_" + spin, "orbital", nmo, spin)
        i, j = (Index(n, mos) for n in ("i", "j"))
        fbar = field("bar_" + spin, i, j)
        coeff = field("c_" + spin, u, i)
        cb = einsum("ui,ij->uj", coeff, fbar)
        ao_weights.append(einsum("uj,vj->uv", cb, coeff))
        frame_weight = einsum("uv,vj->uj", h, cb, coefficient=2)
        dc = field("dc_" + spin, atom, axis, u, j)
        frame_actions.append(einsum("Akuj,uj->Ak", dc, frame_weight, coefficient=-1))
        djk = field("djk_" + spin, atom, axis, i, j)
        jk_actions.append(einsum("Akij,ij->Ak", djk, fbar, coefficient=-1))
    ao_bar = add(*ao_weights)
    explicit = einsum("Akuv,uv->Ak", dh, ao_bar, coefficient=-1)
    frame = add(*frame_actions)
    jk = add(*jk_actions)
    program = Program(
        {
            "explicit_core": explicit,
            "core_orbital_response": frame,
            "coulomb_exchange": jk,
            "ordinary_fock": add(explicit, frame, jk),
            "ao_core_weight": ao_bar,
        },
        provenance={
            "method": "Unrestricted CC Fock contribution from native AO Hcore and moving spin orbitals",
            "units": "Eh/bohr",
        },
    )
    return run(program, fields).outputs


def unrestricted_cc_fock_jk_pullback(
    occupied: tuple[int, int],
    weights: Mapping[str, np.ndarray],
    coefficients: Mapping[str, np.ndarray],
    three_center: np.ndarray,
    metric: np.ndarray,
    whitener: np.ndarray,
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> dict[str, np.ndarray]:
    """Transpose the six CC Fock blocks through the DF Coulomb and spin exchange.

    The shared three-center tensor is (aux, AO, AO). Both real spin frames
    include every ordinary MO column and use these AO rows. The full-rank
    whitener satisfies W.T @ W = inverse(metric). Its physical response is
    included implicitly through the metric equation. Frozen electrons enter
    the charge and exchange even when their CC Fock cotangents vanish.

    The returned AO integral and MO coefficient weights are energy derivatives.
    A negative contraction with nuclear tangents gives the J−K force.
    """
    nao, nmo = coefficients["alpha"].shape
    aux = len(metric)
    if (
        coefficients["beta"].shape != (nao, nmo)
        or metric.shape != (aux, aux)
        or whitener.shape != metric.shape
        or three_center.shape != (aux, nao, nao)
    ):
        raise ValueError("Inconsistent AO, MO or auxiliary dimensions for DF Fock")
    ao = {
        "coefficients_alpha": coefficients["alpha"],
        "coefficients_beta": coefficients["beta"],
        "ao_coulomb": three_center,
        "metric/coulomb": metric,
        "metric_whitener": whitener,
    }
    if any(arr.dtype != np.float64 for arr in ao.values()):
        raise ValueError("DF Fock inputs must be real FP64 arrays")
    if not np.allclose(whitener @ metric @ whitener.T, np.eye(aux), atol=2e-7, rtol=0):
        raise ValueError("DF Fock requires a full-rank whitener for the same metric")
    space = (*occupied, nmo, nmo, aux)
    projected, _ = _orbital_sources(
        space, ao, run, None, projected=None, role="ordinary_jk"
    )
    values = {
        k.removeprefix("source/"): v
        for k, v in projected.items()
        if k.startswith("source/")
    }
    # The DF Fock program also accepts an additive one-electron term.
    # Zeroing it isolates the two-electron part before differentiation.
    seeds = {}
    for spin, no, nf in zip(("alpha", "beta"), occupied, frozen_occupied, strict=True):
        values["hcore_" + spin] = np.zeros((nmo, nmo))
        seeds["fock_" + spin] = _spin_fock_weight(weights, spin, nmo, no, nf)
    seed = spin_fock_pullback((*occupied, nmo, aux), values, seeds, run)
    _, bars = _orbital_sources(
        space,
        ao,
        run,
        {"source/" + k: v for k, v in seed.items()},
        projected=projected,
        role="ordinary_jk",
    )
    assert bars is not None
    return bars
