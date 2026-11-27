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
    _F12_METRIC_OPS,
    _shared_contiguous_operands,
    batch_execute,
    is_spin_symmetric_pair_data,
    occupied_pair_blocks,
    pair_operands,
    prefactor_operator_residuals,
    pullback_operator_residuals,
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
    ao_bar_sink: Callable[[str, np.ndarray], None] | None = None,
    preprojected: tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None] | None = None,
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
        jk_streaming = "__ao_loader__" in jkfit
        req_keys = (
            {"__ao_loader__", "metric/coulomb", "metric_whitener"}
            if jk_streaming
            else {"ao_coulomb", "metric/coulomb", "metric_whitener"}
        )
        if not req_keys.issubset(set(jkfit)) or (set(jkfit) - req_keys - {"__obs_coulomb__"}):
            raise ValueError(
                "JKFit requires Coulomb three-center integrals, metric and whitener"
            )
        n_ao = len(ao["overlap_ao"])
        qjk = len(jkfit["metric/coulomb"])
        if (
            qjk < 1
            or (not jk_streaming and jkfit["ao_coulomb"].shape != (qjk, n_ao, n_ao))
            or jkfit["metric/coulomb"].shape != (qjk, qjk)
            or jkfit["metric_whitener"].shape != (qjk, qjk)
        ):
            raise ValueError("JKFit dimensions disagree with the common AO basis")
        if any(
            isinstance(value, np.ndarray)
            and (value.dtype != np.float64 or not np.isfinite(value).all())
            for value in jkfit.values()
        ):
            raise ValueError("JKFit sources must contain finite FP64 values")
        metric, whiten = jkfit["metric/coulomb"], jkfit["metric_whitener"]
        if not np.allclose(whiten @ metric @ whiten.T, np.eye(qjk), atol=2e-7, rtol=0):
            raise ValueError("JKFit whitener does not match its Coulomb metric")
        jk_space = (*space[:4], qjk)
        shared = ("overlap_ao", "hcore_ao", "coefficients_alpha", "coefficients_beta")
        jk_ao = {**{name: ao[name] for name in shared}, **jkfit}
        if preprojected is not None and preprojected[1] is not None:
            jk_data = dict(preprojected[1])
        else:
            jk_data, _ = _orbital_sources(
                jk_space, jk_ao, run, None, projected=None, role="fock"
            )
    pair_role = "shared" if jk_data is None else "pairs"
    use_occ_coul = jk_data is not None
    if preprojected is not None:
        data = dict(preprojected[0])
    else:
        data, _ = _orbital_sources(
            space, ao, run, None, projected=None, role=pair_role, occupied_coulomb=use_occ_coul
        )
    weights, terms, deferred_fock_bars = _projected_response(
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
        allow_ab_mirror=True,
        defer_split_fock_pullback=jk_data is not None,
    )
    _, bars = _orbital_sources(
        space,
        ao,
        run,
        weights,
        projected=data,
        role=pair_role,
        ao_bar_sink=ao_bar_sink,
        occupied_coulomb=use_occ_coul,
    )
    del data, weights
    assert terms is not None and bars is not None
    if jk_data is not None:
        assert deferred_fock_bars is not None
        oa_c, ob_c, _, ext_c, _ = space
        fock_sources = {
            key.removeprefix("source/"): v
            for key, v in jk_data.items()
            if key.startswith("source/")
        }
        fock_space = (oa_c, ob_c, ext_c, fock_sources["raw_coulomb_alpha"].shape[0])
        # Save compact [Q_jk, occ, ext] slices needed by reverse _orbital_sources before
        # in-place spin_fock_pullback overwrites the large [Q_jk, ext, ext] raw_coulomb buffer.
        jk_occ_slices = {}
        for s_name, n_occ in (("alpha", oa_c), ("beta", ob_c)):
            ck = "source/raw_coulomb_" + s_name
            if ck in jk_data:
                jk_occ_slices[ck] = np.array(
                    jk_data[ck][:, :n_occ, :], dtype=np.float64, order="C", copy=True
                )
        owns_jk_data = preprojected is None or preprojected[1] is None
        fock_weights = spin_fock_pullback(
            fock_space,
            fock_sources,
            deferred_fock_bars,
            run,
            auxiliary_batch_size,
            inplace_raw_coulomb=owns_jk_data,
        )
        del deferred_fock_bars, fock_sources
        jk_data.update(jk_occ_slices)
        del jk_occ_slices
        jk_weights = {"source/" + k: v for k, v in fock_weights.items()}
        del fock_weights
        _, jk_bars = _orbital_sources(
            jk_space, jk_ao, run, jk_weights, projected=jk_data, role="fock"
        )
        del jk_data, jk_weights
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
    allow_ab_mirror: bool = False,
    defer_split_fock_pullback: bool = False,
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
        occupied_pair_blocks(
            space, frozen_occupied=frozen_occupied, prefactored_metric=True
        )
        if selected
        else ()
    )
    spin_sym = is_spin_symmetric_pair_data(
        space, data, sources, fock, frozen_occupied, occupied_f12_weights
    )
    residuals = prefactor_operator_residuals(data, sources) if selected else {}
    res_bars = {k: np.zeros_like(v) for k, v in residuals.items()}
    count = 0
    for sl, sr, nl, nr, graph, pairs in blocks:
        if spin_sym and sl == "beta" and sr == "beta":
            continue
        primal_graph = Program({name: graph.outputs[name] for name in selected})
        rev = transpose_program(
            primal_graph, selected, max_elements=max(1_000_000, ext * ext)
        ).program
        kernel = Program({**rev.outputs, **primal_graph.outputs}) if energies else rev
        shared = _shared_contiguous_operands(
            data, fock, sl, sr, include_metrics=False
        )
        mirror_bb = spin_sym and sl == "alpha" and sr == "alpha"
        mirror_ab_block = spin_sym and allow_ab_mirror and sl == "alpha" and sr == "beta"
        with batch_execute(run, kernel) as exec_one:
            for i, j in pairs:
                if mirror_ab_block and i > j:
                    continue
                mirror_ab = mirror_ab_block and i < j
                primal = pair_operands(
                    data, sources, fock, sl, sr, nl, nr, ext, i, j, residuals=residuals
                )
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
                doubled = mirror_bb or mirror_ab
                if energies:
                    count += 2 if doubled else 1
                    if "ordinary_pair_energy" in selected:
                        val = float(result["ordinary_pair_energy"])
                        mp2_terms.append(val)
                        if doubled:
                            mp2_terms.append(val)
                    if "pair_energy" in selected:
                        val = float(result["pair_energy"]) * weight
                        f12_terms.append(val)
                        if doubled:
                            f12_terms.append(val)
                # Pair operands are views into the molecular arrays. Shared
                # Fock and repeated occupied-leg entries accumulate independently.
                buffers = pair_operands(
                    raw_bars, source_bars, fock_bars, sl, sr, nl, nr, ext, i, j,
                    residuals=res_bars,
                )
                for key in rev.outputs:
                    buffers[key.removeprefix("bar_")] += result[key]
                if mirror_bb:
                    bb_buffers = pair_operands(
                        raw_bars, source_bars, fock_bars, "beta", "beta", nl, nr, ext, i, j,
                        residuals=res_bars,
                    )
                    for key in rev.outputs:
                        bb_buffers[key.removeprefix("bar_")] += result[key]
                elif mirror_ab:
                    if "bar_fock_alpha" in result:
                        fock_bars["fock_beta"] += result["bar_fock_alpha"]
                    if "bar_fock_beta" in result:
                        fock_bars["fock_alpha"] += result["bar_fock_beta"]
                    if "bar_exchange_alpha" in result:
                        fock_bars["exchange_beta"] += result["bar_exchange_alpha"]
                    if "bar_exchange_beta" in result:
                        fock_bars["exchange_alpha"] += result["bar_exchange_beta"]
                    if "bar_cross_overlap" in result:
                        raw_bars["cross_overlap"] += result["bar_cross_overlap"].T
                    if "bar_fit_i_alpha" in result:
                        source_bars["fit_beta"][:, i] += result["bar_fit_i_alpha"]
                    if "bar_fit_j_beta" in result:
                        source_bars["fit_alpha"][:, j] += result["bar_fit_j_beta"]
                    if "bar_raw_coulomb_j_beta" in result:
                        source_bars["raw_coulomb_alpha"][:, j] += result["bar_raw_coulomb_j_beta"]
                    for op, _ in _F12_METRIC_OPS:
                        rk = "bar_raw_" + op + "_i_alpha"
                        if rk in result:
                            raw_bars["raw/" + op + "/beta"][:, i] += result[rk]
                        sk = "bar_res_" + op + "_j_beta"
                        if sk in result:
                            res_bars["res/" + op + "/alpha"][:, j] += result[sk]
    if selected:
        pullback_operator_residuals(data, sources, res_bars, raw_bars, source_bars)
    if spin_sym:
        # Closed-shell spin symmetry is exact; averaging eliminates 1-ULP GEMM-order
        # alpha/beta skew so spin_fock_pullback can share its large [Q, ext, ext] buffer.
        for base in ("fock", "exchange"):
            ka, kb = f"{base}_alpha", f"{base}_beta"
            if ka in fock_bars and kb in fock_bars:
                avg = 0.5 * (fock_bars[ka] + fock_bars[kb])
                fock_bars[ka] = avg
                fock_bars[kb] = avg.copy()
    if defer_split_fock_pullback and fock_data is not None:
        raw_bars.update({"source/" + k: v for k, v in source_bars.items()})
        terms = None
        if energies:
            mp2 = math.fsum(mp2_terms)
            f12 = math.fsum(f12_terms)
            alpha, beta = singles_energies
            if not all(math.isfinite(value) for value in (mp2, f12, alpha, beta)):
                raise ValueError("Nonfinite unrestricted MP2-F12 or CABS energy")
            terms = UnrestrictedF12Energy(mp2, f12, (alpha, beta), count)
        return raw_bars, terms, fock_bars
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
