"""Unrestricted DF-Fock and CABS singles composition in the alpha/beta frames."""

import math
from collections.abc import Callable, Mapping
from typing import Literal, NamedTuple

import numpy as np
from generativeqc_compiler.tensor import Program, transpose_program

from ._f12_cabs import cabs_singles
from ._f12_focal_tplus_sources import SpinOccupied, checked_occupied_weights
from ._f12_spin_ao import _orbital_sources
from ._f12_spin_fock import spin_fock, spin_fock_pullback
from ._f12_spin_pairs import (
    _shared_contiguous_operands,
    batch_execute,
    is_spin_symmetric_pair_data,
    occupied_pair_blocks,
    pair_operands,
)


class UnrestrictedCabsSingles(NamedTuple):
    energy: float
    spin_energies: tuple[float, float]
    source_cotangents: dict[str, np.ndarray] | None


class UnrestrictedF12Energy(NamedTuple):
    ordinary_mp2: float
    f12_increment: float
    cabs_singles: tuple[float, float]
    active_pairs: int


def unrestricted_cabs_energy(
    dimensions: tuple[int, int, int, int],
    sources: Mapping[str, np.ndarray],
    run: Callable,
    *,
    pullback: bool = True,
    singles: Callable = cabs_singles,
    auxiliary_batch_size: int | None = None,
) -> UnrestrictedCabsSingles:
    """Compute both singles contributions and their joint fixed-frame DF source VJP.

    Dimensions = alpha occupied, beta occupied, extended orbital count, fit count.
    The alpha/beta orbital frames can be distinct. The occupied plus extended-
    virtual Sylvester problem is separate for each spin. A shared Coulomb source
    couples their responses to both occupied densities through the Fock graph.
    Nuclear differentiation also needs frame, fitting and SCF response.
    """
    forward = spin_fock(dimensions, sources, run, auxiliary_batch_size)
    seeds = {}
    energies = []
    for spin, occupied in zip(("alpha", "beta"), dimensions[:2], strict=True):
        key = "fock_" + spin
        fock = forward[key]
        # Physical F is symmetric. Define the derivative on the symmetric
        # subspace even if raw and metric-solved factor inputs vary separately.
        energy, bar = singles(
            (fock + fock.T) / 2, occupied, spin_weight=1, pullback=pullback
        )
        energies.append(float(energy))
        if pullback:
            if bar is None:
                raise ValueError("missing singles Fock cotangent")
            seeds[key] = (bar + bar.T) / 2
    weights = None
    if pullback:
        weights = spin_fock_pullback(
            dimensions, sources, seeds, run, auxiliary_batch_size
        )
    return UnrestrictedCabsSingles(sum(energies), (energies[0], energies[1]), weights)


def unrestricted_mp2_f12_source_pullback(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    singles: Callable = cabs_singles,
    auxiliary_batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Pull the molecular UMP2, geminal and CABS energies back to their shared sources.

    One reverse per occupied pair provides weights reused for all nuclei. These
    are derivatives in the two moving OBS/CABS frames. Contract them with total
    physical source tangents including integral, metric, UHF and orbital motion,
    and add the reference gradient. The caller owns that geometry response.
    """
    weights, _, _ = _projected_response(
        space, data, run, frozen_occupied, singles, auxiliary_batch_size, energies=False
    )
    return weights


def unrestricted_mp2_f12_ao_response(
    space: tuple[int, int, int, int, int],
    ao: dict[str, np.ndarray],
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    singles: Callable = cabs_singles,
    auxiliary_batch_size: int | None = None,
    jkfit: Mapping[str, np.ndarray] | None = None,
    component: Literal["all", "ordinary", "f12", "singles"] = "all",
    occupied_f12_weights: SpinOccupied | None = None,
) -> tuple[UnrestrictedF12Energy, dict[str, np.ndarray]]:
    """Evaluate the correlation and singles energies and their common AO cotangents.

    Supply the two spin frames, raw operators, metrics and Coulomb whitener at
    one geometry. Each pair returns both its primal energies and its reverse
    contractions. The AO result includes both spin-coefficient cotangents and
    the physical Coulomb metric derivative. Nuclear and UHF response are
    applied by the force caller. With an independent JKFit panel, pass raw
    ao_coulomb, metric/coulomb and metric_whitener in that fit. Pair operators
    use the primary correlation fit. Selecting one component evaluates its
    energy and transposes its own seed through both fits and orbital frames.
    Both fits share the AO rows, spin frames
    and physical core. JKFit integral weights have the prefix "jkfit/".
    Selecting the F12 component can also seed its occupied energy partition.
    """
    if component not in ("all", "ordinary", "f12", "singles"):
        raise ValueError("unknown unrestricted F12 component")
    if occupied_f12_weights is not None:
        if component != "f12":
            raise ValueError("occupied F12 weights require the F12 component")
        checked_occupied_weights(occupied_f12_weights, space[:2])
    jk_data = None
    if jkfit is not None:
        if set(jkfit) != {"ao_coulomb", "metric/coulomb", "metric_whitener"}:
            raise ValueError(
                "JKFit requires Coulomb three-center integrals, metric and whitener"
            )
        n_ao = len(ao["overlap_ao"])
        qjk = len(jkfit["metric/coulomb"])
        if (
            qjk < 1
            or jkfit["ao_coulomb"].shape != (qjk, n_ao, n_ao)
            or jkfit["metric/coulomb"].shape != (qjk, qjk)
            or jkfit["metric_whitener"].shape != (qjk, qjk)
        ):
            raise ValueError("JKFit dimensions disagree with the common AO basis")
        if any(
            value.dtype != np.float64 or not np.isfinite(value).all()
            for value in jkfit.values()
        ):
            raise ValueError("JKFit sources must contain finite FP64 values")
        metric, whiten = jkfit["metric/coulomb"], jkfit["metric_whitener"]
        if not np.allclose(whiten @ metric @ whiten.T, np.eye(qjk), atol=2e-7, rtol=0):
            raise ValueError("JKFit whitener does not match its Coulomb metric")
        jk_space = (*space[:4], qjk)
        shared = ("overlap_ao", "hcore_ao", "coefficients_alpha", "coefficients_beta")
        jk_ao = {**{name: ao[name] for name in shared}, **jkfit}
        jk_data, _ = _orbital_sources(
            jk_space, jk_ao, run, None, projected=None, role="fock"
        )
    pair_role = "shared" if jk_data is None else "pairs"
    data, _ = _orbital_sources(space, ao, run, None, projected=None, role=pair_role)
    weights, terms, jk_weights = _projected_response(
        space,
        data,
        run,
        frozen_occupied,
        singles,
        auxiliary_batch_size,
        energies=True,
        fock_data=jk_data,
        component=component,
        occupied_f12_weights=occupied_f12_weights,
    )
    _, bars = _orbital_sources(space, ao, run, weights, projected=data, role=pair_role)
    assert terms is not None and bars is not None
    if jk_data is not None:
        _, jk_bars = _orbital_sources(
            jk_space, jk_ao, run, jk_weights, projected=jk_data, role="fock"
        )
        for name in shared:
            bars[name] += jk_bars[name]
        for name in ("ao_coulomb", "metric/coulomb"):
            bars["jkfit/" + name] = jk_bars[name]
    return terms, bars


def _projected_response(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    run: Callable,
    frozen_occupied: tuple[int, int],
    singles: Callable,
    auxiliary_batch_size: int | None,
    *,
    energies: bool,
    fock_data: dict[str, np.ndarray] | None = None,
    component: str = "all",
    occupied_f12_weights: SpinOccupied | None = None,
) -> tuple[
    dict[str, np.ndarray], UnrestrictedF12Energy | None, dict[str, np.ndarray] | None
]:
    oa, ob, _, ext, _ = space
    sources = {
        key.removeprefix("source/"): v
        for key, v in data.items()
        if key.startswith("source/")
    }
    fock_sources = (
        sources
        if fock_data is None
        else {
            key.removeprefix("source/"): v
            for key, v in fock_data.items()
            if key.startswith("source/")
        }
    )
    fock_space = (oa, ob, ext, fock_sources["raw_coulomb_alpha"].shape[0])
    fock = spin_fock(fock_space, fock_sources, run, auxiliary_batch_size)
    source_bars = {k: np.zeros_like(v) for k, v in sources.items()}
    raw_bars = {
        k: np.zeros_like(v)
        for k, v in data.items()
        if k == "cross_overlap" or k.startswith(("raw/", "metric/"))
    }
    fock_bars = {k: np.zeros_like(v) for k, v in fock.items()}
    singles_energies = [0.0, 0.0]
    mp2_terms, f12_terms = [], []
    for index, (spin, occupied) in enumerate((("alpha", oa), ("beta", ob))):
        if component not in ("all", "singles"):
            continue
        key = "fock_" + spin
        matrix = fock[key]
        energy, bar = singles(
            (matrix + matrix.T) / 2, occupied, spin_weight=1, pullback=True
        )
        if bar is None:
            raise ValueError("Missing UHF CABS singles Fock cotangent")
        if energies:
            singles_energies[index] = float(energy)
        fock_bars[key] += (bar + bar.T) / 2
    selected = tuple(
        name
        for part, name in (("f12", "pair_energy"), ("ordinary", "ordinary_pair_energy"))
        if component in ("all", part)
    )
    blocks = (
        occupied_pair_blocks(space, frozen_occupied=frozen_occupied) if selected else ()
    )
    spin_sym = is_spin_symmetric_pair_data(
        space, data, sources, fock, frozen_occupied, occupied_f12_weights
    )
    count = 0
    for sl, sr, nl, nr, graph, pairs in blocks:
        if spin_sym and sl == "beta" and sr == "beta":
            continue
        primal_graph = Program({name: graph.outputs[name] for name in selected})
        rev = transpose_program(primal_graph, selected).program
        kernel = Program({**rev.outputs, **primal_graph.outputs}) if energies else rev
        shared = _shared_contiguous_operands(data, fock, sl, sr)
        mirror_bb = spin_sym and sl == "alpha" and sr == "alpha"
        with batch_execute(run, kernel) as exec_one:
            for i, j in pairs:
                primal = pair_operands(data, sources, fock, sl, sr, nl, nr, ext, i, j)
                feeds = dict(shared)
                for k, v in primal.items():
                    if k not in shared:
                        feeds[k] = np.ascontiguousarray(v)
                weight = (
                    1.0
                    if occupied_f12_weights is None
                    else (occupied_f12_weights[sl == "beta"][i] + occupied_f12_weights[sr == "beta"][j]) * 0.5
                )
                feeds.update({"bar_" + name: np.array(weight) for name in selected})
                result = exec_one(feeds).outputs
                if energies:
                    count += 2 if mirror_bb else 1
                    if "ordinary_pair_energy" in selected:
                        val = float(result["ordinary_pair_energy"])
                        mp2_terms.append(val)
                        if mirror_bb:
                            mp2_terms.append(val)
                    if "pair_energy" in selected:
                        val = float(result["pair_energy"]) * weight
                        f12_terms.append(val)
                        if mirror_bb:
                            f12_terms.append(val)
                # Pair operands are views into the molecular arrays. Shared metric,
                # Fock, and repeated occupied-leg entries accumulate independently.
                buffers = pair_operands(
                    raw_bars, source_bars, fock_bars, sl, sr, nl, nr, ext, i, j
                )
                for key in rev.outputs:
                    buffers[key.removeprefix("bar_")] += result[key]
                if mirror_bb:
                    bb_buffers = pair_operands(
                        raw_bars, source_bars, fock_bars, "beta", "beta", nl, nr, ext, i, j
                    )
                    for key in rev.outputs:
                        bb_buffers[key.removeprefix("bar_")] += result[key]
    fock_weights = spin_fock_pullback(
        fock_space,
        fock_sources,
        fock_bars,
        run,
        auxiliary_batch_size,
        accumulate=source_bars if fock_data is None else None,
    )
    raw_bars.update({"source/" + k: v for k, v in source_bars.items()})
    terms = None
    if energies:
        mp2 = math.fsum(mp2_terms)
        f12 = math.fsum(f12_terms)
        alpha, beta = singles_energies
        if not all(math.isfinite(value) for value in (mp2, f12, alpha, beta)):
            raise ValueError("Nonfinite unrestricted MP2-F12 or CABS energy")
        terms = UnrestrictedF12Energy(mp2, f12, (alpha, beta), count)
    jk_weights = (
        None
        if fock_data is None
        else {"source/" + k: v for k, v in fock_weights.items()}
    )
    return raw_bars, terms, jk_weights
