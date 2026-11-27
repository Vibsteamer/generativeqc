"""Enumerate the occupied pairs of a canonical unrestricted 3C(FIX) reference."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from itertools import combinations, product
from typing import Any, NamedTuple

import numpy as np
from generativeqc_compiler.mp2.f12_opposite_spin import opposite_spin_projected_pair
from generativeqc_compiler.mp2.f12_unrestricted import projected_same_spin_f12_pair
from generativeqc_compiler.tensor import Program

from ._f12_mp2_pair_native import (
    aot_mp2f12_disabled,
    bind_mp2_f12_pair_energies,
    run_mp2_f12_pair_energies_native,
)


class SpinPairBlock(NamedTuple):
    left: str
    right: str
    left_occupied: int
    right_occupied: int
    graph: Program
    pairs: Iterator[tuple[int, int]]


def occupied_pair_blocks(
    space: tuple[int, int, int, int, int],
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    prefactored_metric: bool = False,
) -> Iterator[SpinPairBlock]:
    """Yield one graph and its occupied pairs per nonempty spin channel.

    Space gives the alpha/beta occupied, OBS, extended, and fit dimensions.
    Optional frozen counts omit lowest occupied pair indices independently per
    spin. Every occupied orbital still participates in the UHF Fock and in the
    strong-orthogonality projector. Nuclear gradients must use the same frozen
    orbital convention and differentiate the complete parent reference.
    """
    oa, ob, obs, ext, aux = space
    fa, fb = frozen_occupied
    if not (0 <= fa <= oa < obs and 0 <= fb <= ob < obs < ext and aux > 0):
        raise ValueError(
            "Invalid occupied/frozen spin spaces or orbital/fitting dimensions"
        )
    for sl, sr, nl, nr, il, ir in (
        ("alpha", "alpha", oa, oa, fa, fa),
        ("alpha", "beta", oa, ob, fa, fb),
        ("beta", "beta", ob, ob, fb, fb),
    ):
        if sl == sr:
            if nl - il < 2:
                continue
            pairs = combinations(range(il, nl), 2)
            graph = projected_same_spin_f12_pair(
                nl,
                obs - nl,
                ext - obs,
                auxiliaries=aux,
                prefactored_metric=prefactored_metric,
            )
        else:
            if nl == il or nr == ir:
                continue
            pairs = product(range(il, nl), range(ir, nr))
            graph = opposite_spin_projected_pair(
                nl,
                nr,
                obs,
                ext,
                auxiliaries=aux,
                prefactored_metric=prefactored_metric,
            )
        yield SpinPairBlock(sl, sr, nl, nr, graph, pairs)


_F12_METRIC_OPS = (
    ("geminal", "f12"),
    ("squared_geminal", "f12_squared"),
    ("geminal_coulomb", "f12_over_coulomb"),
    ("double_commutator", "double_commutator"),
)


def prefactor_operator_residuals(
    data: dict, sources: dict
) -> dict[str, np.ndarray]:
    """Precompute res/{op}/{spin} = raw/{op}/{spin} - metric/{op} @ fit_{spin} once per spin."""
    residuals: dict[str, np.ndarray] = {}
    for op, _ in _F12_METRIC_OPS:
        metric = data["metric/" + op]
        q = metric.shape[0]
        for spin in ("alpha", "beta"):
            raw_key = "raw/" + op + "/" + spin
            fit_key = "fit_" + spin
            if raw_key in data and fit_key in sources:
                raw = data[raw_key]
                fit = sources[fit_key]
                residuals["res/" + op + "/" + spin] = raw - (
                    metric @ fit.reshape(q, -1)
                ).reshape(fit.shape)
    return residuals


def pullback_operator_residuals(
    data: dict,
    sources: dict,
    res_bars: dict[str, np.ndarray],
    raw_bars: dict[str, np.ndarray],
    source_bars: dict[str, np.ndarray],
) -> None:
    """Propagate accumulated residual cotangents back to raw, fit, and metric once per spin."""
    for op, _ in _F12_METRIC_OPS:
        metric = data["metric/" + op]
        q = metric.shape[0]
        for spin in ("alpha", "beta"):
            res_key = "res/" + op + "/" + spin
            if res_key not in res_bars:
                continue
            bar_res = res_bars[res_key]
            raw_bars["raw/" + op + "/" + spin] += bar_res
            fit = sources["fit_" + spin]
            bar_res_2d = bar_res.reshape(q, -1)
            source_bars["fit_" + spin] -= (metric.T @ bar_res_2d).reshape(fit.shape)
            raw_bars["metric/" + op] -= bar_res_2d @ fit.reshape(q, -1).T


def pair_operands(
    data: dict,
    src: dict,
    fock: dict,
    sl: str,
    sr: str,
    nl: int,
    nr: int,
    n: int,
    i: int,
    j: int,
    *,
    residuals: dict | None = None,
) -> dict[str, np.ndarray]:
    """Select the orbital and fit tensors for one occupied spin pair."""
    dl, dr = src["fit_" + sl][:, i], src["fit_" + sr][:, j]
    raw_c = src["raw_coulomb_" + sr][:, j]
    select_i = np.zeros(nl if sl == sr else n)
    select_j = np.zeros(nr if sl == sr else n)
    select_i[i] = select_j[j] = 1.0
    if sl == sr:
        vals = {
            "select_i": select_i,
            "select_j": select_j,
            "fock": fock["fock_" + sl],
            "exchange": fock["exchange_" + sl],
            "fit_i": dl,
            "fit_j": dr,
            "raw_coulomb_j": raw_c,
        }
    else:
        vals = {
            "select_i": select_i,
            "select_j": select_j,
            "cross_overlap": data["cross_overlap"],
            "fit_i_alpha": dl,
            "fit_j_beta": dr,
            "raw_coulomb_j_beta": raw_c,
            **fock,
        }
    for op, rel in _F12_METRIC_OPS:
        la = data["raw/" + op + "/" + sl][:, i]
        if residuals is not None:
            rb = residuals["res/" + op + "/" + sr][:, j]
            if sl == sr:
                vals["raw_" + rel + "_i"] = la
                vals["res_" + rel + "_j"] = rb
            else:
                vals["raw_" + op + "_i_alpha"] = la
                vals["res_" + op + "_j_beta"] = rb
        else:
            rb = data["raw/" + op + "/" + sr][:, j]
            if sl == sr:
                vals.update(
                    {
                        "raw_" + rel + "_i": la,
                        "raw_" + rel + "_j": rb,
                        "metric_" + rel: data["metric/" + op],
                    }
                )
            else:
                vals.update(
                    {
                        "raw_" + op + "_i_alpha": la,
                        "raw_" + op + "_j_beta": rb,
                        "metric_" + op: data["metric/" + op],
                    }
                )
    return vals


class PairEnergy(NamedTuple):
    spins: tuple[str, str]
    occupied_indices: tuple[int, int]
    mp2: float | None
    f12: float


@contextmanager
def batch_execute(run: Callable, program: Program):
    """Reuse one prepared CUDA executor across repeated calls on the same program."""
    batch_fn = getattr(run, "batch", None)
    if callable(batch_fn):
        with batch_fn(program) as execute_one:
            yield execute_one
    else:
        yield lambda inputs: run(program, inputs)


def is_spin_symmetric_pair_data(
    space: tuple[int, int, int, int, int],
    data: dict,
    sources: dict,
    fock: dict,
    frozen_occupied: tuple[int, int],
    occupied_weights: tuple[np.ndarray, np.ndarray] | None = None,
) -> bool:
    """Verify from actual numerical arrays whether alpha and beta pair inputs are identical."""
    oa, ob = space[0], space[1]
    if oa != ob or frozen_occupied[0] != frozen_occupied[1]:
        return False
    if occupied_weights is not None and not np.array_equal(
        occupied_weights[0], occupied_weights[1]
    ):
        return False
    if not (
        np.array_equal(fock["fock_alpha"], fock["fock_beta"])
        and np.array_equal(fock["exchange_alpha"], fock["exchange_beta"])
        and np.array_equal(sources["fit_alpha"], sources["fit_beta"])
        and np.array_equal(sources["raw_coulomb_alpha"], sources["raw_coulomb_beta"])
    ):
        return False
    cross = data.get("cross_overlap")
    if cross is not None and not np.array_equal(cross, cross.T):
        return False
    for op in ("geminal", "squared_geminal", "geminal_coulomb", "double_commutator"):
        if not np.array_equal(
            data["raw/" + op + "/alpha"], data["raw/" + op + "/beta"]
        ):
            return False
    return True


def _is_rhf_spatial_identity_pair_data(
    space: tuple[int, int, int, int, int],
    data: dict,
    sources: dict,
    fock: dict,
    frozen_occupied: tuple[int, int],
    occupied_weights: tuple[np.ndarray, np.ndarray] | None = None,
) -> bool:
    """Return True when alpha/beta pair inputs match and cross_overlap == I with symmetric F/K/metrics."""
    oa, ob, _, ext, _ = space
    if oa != ob or frozen_occupied[0] != frozen_occupied[1]:
        return False
    if occupied_weights is not None and not np.array_equal(
        occupied_weights[0], occupied_weights[1]
    ):
        return False
    fa_mat = fock["fock_alpha"]
    ka_mat = fock["exchange_alpha"]
    if not (
        np.array_equal(fa_mat, fock["fock_beta"])
        and np.array_equal(ka_mat, fock["exchange_beta"])
        and np.array_equal(sources["fit_alpha"], sources["fit_beta"])
        and np.array_equal(
            sources["raw_coulomb_alpha"][:, :oa], sources["raw_coulomb_beta"][:, :ob]
        )
    ):
        return False
    for op in ("geminal", "squared_geminal", "geminal_coulomb", "double_commutator"):
        if not np.array_equal(
            data["raw/" + op + "/alpha"], data["raw/" + op + "/beta"]
        ):
            return False
        metric = data["metric/" + op]
        if not (float(np.max(np.abs(metric - metric.T))) <= 1e-12):
            return False
    if not (
        float(np.max(np.abs(fa_mat - fa_mat.T))) <= 1e-11
        and float(np.max(np.abs(ka_mat - ka_mat.T))) <= 1e-11
    ):
        return False
    cross = data.get("cross_overlap")
    if cross is not None:
        if cross.shape != (ext, ext):
            return False
        if not (float(np.max(np.abs(cross - np.eye(ext)))) <= 1e-12):
            return False
    return True


def _shared_contiguous_operands(
    data: dict, fock: dict, sl: str, sr: str, *, include_metrics: bool = True
) -> dict[str, np.ndarray]:
    """Pre-materialize contiguous shared matrices once per spin block instead of per pair."""
    if sl == sr:
        shared = {
            "fock": np.ascontiguousarray(fock["fock_" + sl]),
            "exchange": np.ascontiguousarray(fock["exchange_" + sl]),
        }
        if include_metrics:
            for op, rel in _F12_METRIC_OPS:
                shared["metric_" + rel] = np.ascontiguousarray(data["metric/" + op])
    else:
        shared = {
            "cross_overlap": np.ascontiguousarray(data["cross_overlap"]),
            **{k: np.ascontiguousarray(v) for k, v in fock.items()},
        }
        if include_metrics:
            for op, _ in _F12_METRIC_OPS:
                shared["metric_" + op] = np.ascontiguousarray(data["metric/" + op])
    return shared


def molecular_pair_energies(
    space: tuple[int, int, int, int, int],
    data: dict,
    sources: dict,
    fock: dict,
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    include_ordinary: bool = True,
    shell_library: Any = None,
    budget_mib: int | None = None,
    backend: str | None = None,
) -> Iterator[PairEnergy]:
    """Evaluate the orbital MP2 contribution and fixed-cusp increment per pair.

    The common auxiliary space is Coulomb fitted. Each spin retains its own
    extended Fock matrix and occupied/virtual MOs. By default routes to the
    dynamic-dimension AOT C++/cuBLAS kernel; set
    GENERATIVEQC_F12_DISABLE_AOT_MP2F12=1 to execute the TensorIR oracle.
    Set include_ordinary=False when a separate MP2Fit supplies ordinary MP2.
    """
    oa, ob, obs, ext, aux = space
    fa, fb = frozen_occupied
    if not (0 <= fa <= oa < obs and 0 <= fb <= ob < obs < ext and aux > 0):
        raise ValueError(
            "Invalid occupied/frozen spin spaces or orbital/fitting dimensions"
        )

    cuda = (
        backend == "cuda"
        if backend is not None
        else callable(getattr(run, "batch", None))
    )
    native_fn = bind_mp2_f12_pair_energies(
        host_reference=not cuda,
        shell_library=shell_library,
    )
    if shell_library is not None and native_fn is None and not aot_mp2f12_disabled():
        suffix = "_host_reference" if not cuda else "_cuda"
        raise RuntimeError(
            f"shell_library {shell_library} does not export gqc_f12_mp2_pair_energies{suffix}"
        )
    if native_fn is not None:
        rhf_spatial = _is_rhf_spatial_identity_pair_data(
            space, data, sources, fock, frozen_occupied
        )
        budget_bytes = (budget_mib << 20) if budget_mib is not None else (512 << 20)
        _, _, pair_f12, pair_ord = run_mp2_f12_pair_energies_native(
            native_fn,
            space=space,
            frozen_occupied=frozen_occupied,
            rhf_spatial=rhf_spatial,
            include_ordinary=include_ordinary,
            budget_bytes=budget_bytes,
            data=data,
            sources=sources,
            fock=fock,
        )
        idx = 0
        if oa - fa >= 2:
            for i, j in combinations(range(fa, oa), 2):
                ordinary = float(pair_ord[idx]) if include_ordinary and pair_ord is not None else None
                increment = float(pair_f12[idx])
                if (ordinary is not None and not np.isfinite(ordinary)) or not np.isfinite(increment):
                    raise ValueError(f"Nonfinite energy for alpha/alpha pair ({i}, {j})")
                yield PairEnergy(("alpha", "alpha"), (i, j), ordinary, increment)
                idx += 1
        if oa > fa and ob > fb:
            for i, j in product(range(fa, oa), range(fb, ob)):
                ordinary = float(pair_ord[idx]) if include_ordinary and pair_ord is not None else None
                increment = float(pair_f12[idx])
                if (ordinary is not None and not np.isfinite(ordinary)) or not np.isfinite(increment):
                    raise ValueError(f"Nonfinite energy for alpha/beta pair ({i}, {j})")
                yield PairEnergy(("alpha", "beta"), (i, j), ordinary, increment)
                idx += 1
        if ob - fb >= 2:
            for i, j in combinations(range(fb, ob), 2):
                ordinary = float(pair_ord[idx]) if include_ordinary and pair_ord is not None else None
                increment = float(pair_f12[idx])
                if (ordinary is not None and not np.isfinite(ordinary)) or not np.isfinite(increment):
                    raise ValueError(f"Nonfinite energy for beta/beta pair ({i}, {j})")
                yield PairEnergy(("beta", "beta"), (i, j), ordinary, increment)
                idx += 1
        return

    spin_sym = is_spin_symmetric_pair_data(space, data, sources, fock, frozen_occupied)
    residuals = prefactor_operator_residuals(data, sources)
    aa_cache: list[tuple[tuple[int, int], float | None, float]] = []
    for left, right, nl, nr, graph, pairs in occupied_pair_blocks(
        space, frozen_occupied=frozen_occupied, prefactored_metric=True
    ):
        if spin_sym and left == "beta" and right == "beta":
            for (i, j), ordinary, increment in aa_cache:
                yield PairEnergy(("beta", "beta"), (i, j), ordinary, increment)
            continue
        selected = (
            graph
            if include_ordinary
            else Program({"pair_energy": graph.outputs["pair_energy"]})
        )
        shared = _shared_contiguous_operands(
            data, fock, left, right, include_metrics=False
        )
        ab_cache: dict[tuple[int, int], tuple[float | None, float]] = {}
        with batch_execute(run, selected) as exec_one:
            for i, j in pairs:
                if spin_sym and left == "alpha" and right == "beta" and i > j:
                    ordinary, increment = ab_cache[(j, i)]
                    yield PairEnergy((left, right), (i, j), ordinary, increment)
                    continue
                values = pair_operands(
                    data,
                    sources,
                    fock,
                    left,
                    right,
                    nl,
                    nr,
                    space[3],
                    i,
                    j,
                    residuals=residuals,
                )
                feeds = dict(shared)
                for key, v in values.items():
                    if key not in shared:
                        feeds[key] = np.ascontiguousarray(v)
                result = exec_one(feeds).outputs
                ordinary = (
                    float(result["ordinary_pair_energy"]) if include_ordinary else None
                )
                increment = float(result["pair_energy"])
                if (ordinary is not None and not np.isfinite(ordinary)) or not np.isfinite(
                    increment
                ):
                    raise ValueError(f"Nonfinite energy for {left}/{right} pair ({i}, {j})")
                if spin_sym and left == "alpha" and right == "alpha":
                    aa_cache.append(((i, j), ordinary, increment))
                elif spin_sym and left == "alpha" and right == "beta" and i < j:
                    ab_cache[(i, j)] = (ordinary, increment)
                yield PairEnergy((left, right), (i, j), ordinary, increment)
