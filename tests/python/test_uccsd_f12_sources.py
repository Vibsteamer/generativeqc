"""Independent checks of the fixed-geminal external spin projection."""

from itertools import combinations, product

import numpy as np
import pytest
from generativeqc._f12_cc_spin_sources import (
    unrestricted_cc_f12_fixed_source_feeds,
    unrestricted_cc_f12_residual_sources,
)
from generativeqc_compiler.cc.f12_residual import standard_v_tile
from generativeqc_compiler.cc.uccsd_equations import build_f12_correction_program
from generativeqc_compiler.cc.uccsd_f12_quadratic import (
    unrestricted_fixed_geminal_quadratic_pair,
)
from generativeqc_compiler.cc.uccsd_f12_sources import (
    unrestricted_f12_pair_sources,
    unrestricted_standard_v_tile,
)
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.tensor import execute, transpose_program


def _dense_projection(fields: dict[str, np.ndarray]) -> np.ndarray:
    fit = fields["fit_left"]
    raw = fields["raw_coulomb_right"]
    geminal = fields["geminal_obs"]
    direct = fields["geminal_coulomb"]
    coulomb = np.tensordot(fit, raw, axes=(0, 0)).transpose(0, 2, 1, 3)
    return direct - (
        coulomb.reshape(direct.size, geminal.size) @ geminal.ravel()
    ).reshape(direct.shape)


@pytest.mark.parametrize("left_spin,right_spin", [("alpha", "beta"), ("beta", "alpha")])
def test_external_v_spin_projection_and_adjoint(
    left_spin: str, right_spin: str
) -> None:
    graph = unrestricted_standard_v_tile(
        5, 4, 7, 2, 3, left_spin=left_spin, right_spin=right_spin
    )
    nodes = {n.attrs["name"]: n for n in graph.live_nodes if n.op == "input"}
    assert tuple(axis.space.spin for axis in graph.outputs["v"].spec.indices) == (
        left_spin,
        right_spin,
    )
    fit, raw = (nodes[name].spec.indices for name in ("fit_left", "raw_coulomb_right"))
    assert fit[0].space == raw[0].space
    assert fit[2].space.spin == left_spin
    assert raw[2].space.spin == right_spin

    rng = np.random.default_rng(512)
    fields = {name: rng.normal(size=node.spec.shape) for name, node in nodes.items()}
    v = execute(graph, fields).outputs["v"]
    np.testing.assert_allclose(v, _dense_projection(fields), rtol=0, atol=3e-14)

    seed = rng.normal(size=v.shape)
    reverse = transpose_program(graph, ("v",)).program
    bars = execute(reverse, {**fields, "bar_v": seed}).outputs
    for name, field in fields.items():
        direction = rng.normal(size=field.shape)
        for step in (4e-4, 2e-4):
            plus = _dense_projection({**fields, name: field + step * direction})
            minus = _dense_projection({**fields, name: field - step * direction})
            numerical = np.vdot(seed, plus - minus) / (2 * step)
            np.testing.assert_allclose(
                np.vdot(bars["bar_" + name], direction), numerical, rtol=0, atol=2e-9
            )


@pytest.mark.parametrize("spin", ["alpha", "beta"])
def test_same_spin_external_v_reuses_ordered_spatial_projection(spin: str) -> None:
    rng = np.random.default_rng(411)
    obs, auxiliary = 5, 6
    left, right = [0, 1, 4], [1, 3]
    raw = rng.normal(size=(auxiliary, obs, obs))
    raw = 0.5 * (raw + raw.swapaxes(1, 2))
    metric_factor = rng.normal(size=(auxiliary, auxiliary))
    metric = metric_factor @ metric_factor.T + np.eye(auxiliary)
    fit = np.linalg.solve(metric, raw.reshape(auxiliary, -1)).reshape(raw.shape)
    direct, geminal = rng.normal(size=(2, obs, obs))
    block = np.ix_(left, right)
    feeds = {
        "geminal_coulomb": direct[block],
        "geminal_obs": geminal,
        "fit_left": fit[:, left, :],
        "raw_coulomb_right": raw[:, right, :],
    }
    graph = unrestricted_standard_v_tile(
        obs, obs, auxiliary, len(left), len(right), left_spin=spin, right_spin=spin
    )
    spatial = standard_v_tile(obs, auxiliary, len(left), len(right))
    np.testing.assert_array_equal(
        execute(graph, feeds).outputs["v"], execute(spatial, feeds).outputs["v"]
    )

    # With a common physical Coulomb metric, electron exchange commutes with
    # the projector. Supplying the dressed triplet source must weight it once.
    dressed = {
        **feeds,
        "geminal_coulomb": 0.25 * (direct - direct.T)[block],
        "geminal_obs": 0.25 * (geminal - geminal.T),
    }
    result = execute(graph, dressed).outputs["v"]
    full = _dense_projection(
        {**feeds, "geminal_coulomb": direct, "fit_left": fit, "raw_coulomb_right": raw}
    )
    np.testing.assert_allclose(
        result, 0.25 * (full - full.T)[block], rtol=0, atol=2e-14
    )


def test_same_spin_external_v_requires_a_common_obs() -> None:
    with pytest.raises(ValueError, match="common OBS"):
        unrestricted_standard_v_tile(
            4, 5, 7, 2, 3, left_spin="alpha", right_spin="alpha"
        )


def _pair_sources_reference(
    feeds: dict[str, np.ndarray], lo: int, ro: int, obs: int
) -> dict[str, np.ndarray]:
    dl = feeds["fit_left"]
    dr = feeds["fit_right"]

    def dressed(operator: str) -> np.ndarray:
        left = feeds[f"raw_{operator}_left"]
        right = feeds[f"raw_{operator}_right"]
        metric = feeds[f"metric_{operator}"]
        matrix = dl.T @ right + left.T @ dr - dl.T @ metric @ dr
        if "cross_overlap" not in feeds:
            return (matrix - matrix.T) / 4
        s = feeds["cross_overlap"]
        return (3 * matrix + s @ matrix.T @ s) / 8

    geminal = dressed("geminal")
    gf = dressed("geminal_coulomb")
    g = dl.T @ feeds["raw_coulomb_right"]
    n = len(geminal)
    obs_projector = np.diag((np.arange(n) < obs).astype(float))
    left_free = np.diag((np.arange(n) >= lo).astype(float))
    right_free = np.diag((np.arange(n) >= ro).astype(float))
    strong = np.kron(left_free, right_free) - np.kron(
        left_free @ obs_projector, right_free @ obs_projector
    )
    f_l = feeds["fock_left"]
    f_r = feeds.get("fock_right", f_l)
    full_fock = np.kron(f_l, np.eye(n)) + np.kron(np.eye(n), f_r)
    qf = strong @ geminal.ravel()
    return {
        "w_obs": _dense_projection({
            "fit_left": feeds["fit_obs_left"],
            "raw_coulomb_right": feeds["raw_coulomb_obs_right"],
            "geminal_obs": geminal[:obs, :obs],
            "geminal_coulomb": gf[:obs, :obs],
        }),
        "c_vv": (full_fock @ qf).reshape(n, n)[lo:obs, ro:obs],
        "linear_v_energy": 2 * (
            feeds["select_i"] @ gf @ feeds["select_j"]
            - g.ravel() @ (geminal.ravel() - qf)
        ),
    }


@pytest.mark.parametrize("channel,lo,ro", [("alpha_alpha", 2, 2), ("alpha_beta", 2, 1), ("beta_beta", 2, 2)])
def test_cusp_dressed_df_pair_sources_and_molecular_input_adjoint(
    channel: str, lo: int, ro: int
) -> None:
    obs, nc, aux = 4, 3, 5
    graph = unrestricted_f12_pair_sources((lo, obs - lo, nc), (ro, obs - ro, nc), aux, channel=channel)
    rng = np.random.default_rng(1106)
    feeds = {
        n.attrs["name"]: rng.normal(scale=0.2, size=n.spec.shape)
        for n in graph.live_nodes if n.op == "input"
    }
    feeds["select_i"] = np.eye(obs + nc)[1]
    feeds["select_j"] = np.eye(obs + nc)[0]
    if channel == "alpha_beta":
        obs_rotation = np.linalg.qr(rng.normal(size=(obs, obs)))[0]
        cabs_rotation = np.linalg.qr(rng.normal(size=(nc, nc)))[0]
        feeds["cross_overlap"] = np.zeros((obs + nc, obs + nc))
        feeds["cross_overlap"][:obs, :obs] = obs_rotation
        feeds["cross_overlap"][obs:, obs:] = cabs_rotation
    actual = execute(graph, feeds).outputs
    expected = _pair_sources_reference(feeds, lo, ro, obs)
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], atol=2e-14, rtol=0)

    seeds = {k: rng.normal(size=v.shape) for k, v in actual.items()}
    reverse = transpose_program(graph, tuple(seeds)).program
    bars = execute(reverse, {**feeds, **{f"bar_{k}": v for k, v in seeds.items()}}).outputs
    directions = {k: rng.normal(scale=0.03, size=v.shape) for k, v in feeds.items() if not k.startswith("select")}
    adjoint = sum(np.vdot(bars["bar_" + k], v) for k, v in directions.items())

    def objective(step: float) -> float:
        outputs = _pair_sources_reference(
            {**feeds, **{k: feeds[k] + step * v for k, v in directions.items()}}, lo, ro, obs
        )
        return sum(np.vdot(seeds[k], v) for k, v in outputs.items())

    for step in (4e-4, 2e-4):
        np.testing.assert_allclose(adjoint, (objective(step) - objective(-step)) / (2 * step), atol=2e-9, rtol=0)


@pytest.mark.parametrize(
    "occupied,frozen",
    [
        ((3, 2), (0, 0)),
        ((3, 2), (1, 1)),
        ((2, 0), (0, 0)),
        ((1, 0), (0, 0)),
    ],
)
def test_molecular_adapter_canonical_spin_permutations(
    occupied: tuple[int, int], frozen: tuple[int, int]
) -> None:
    rng = np.random.default_rng(196)
    obs, ext, aux = 5, 7, 4
    data = {"cross_overlap": np.eye(ext)}
    fock = {}
    for spin, n in zip(("alpha", "beta"), occupied, strict=True):
        raw = rng.normal(scale=0.2, size=(aux, ext, ext))
        raw = (raw + raw.swapaxes(-1, -2)) / 2
        data["source/raw_coulomb_" + spin] = raw
        data["source/fit_" + spin] = raw[:, :n]
        fock["fock_" + spin] = rng.normal(scale=0.2, size=(ext, ext))
        for op in ("geminal", "geminal_coulomb"):
            data[f"raw/{op}/{spin}"] = rng.normal(scale=0.2, size=(aux, n, ext))
            data[f"metric/{op}"] = np.eye(aux)
    blocks, v = unrestricted_cc_f12_residual_sources(
        (*occupied, obs, ext, aux), data, fock, np.eye(aux), execute, frozen_occupied=frozen
    )
    assert np.isfinite(v)
    for name in ("w_vo", "w_vv", "c_f12"):
        np.testing.assert_array_equal(blocks[f"{name}_abab"], -blocks[f"{name}_baab"].swapaxes(0, 1))
        np.testing.assert_array_equal(blocks[f"{name}_abba"], -blocks[f"{name}_baba"].swapaxes(0, 1))
    for name in ("w_vv", "c_f12"):
        np.testing.assert_array_equal(blocks[f"{name}_abab"], -blocks[f"{name}_abba"].swapaxes(2, 3))
        np.testing.assert_allclose(blocks[f"{name}_aaaa"], -blocks[f"{name}_aaaa"].swapaxes(2, 3), rtol=0, atol=2e-15)

    oa, ob = occupied
    layout = UCCSDLayout(oa - frozen[0], ob - frozen[1], obs - oa, obs - ob)
    program = build_f12_correction_program(layout)
    amplitudes = layout.unpack(np.zeros(layout.packed_size))
    accepted = execute(program, {**amplitudes, **blocks, "fixed_geminal_energy": np.array(v)}).outputs
    np.testing.assert_equal(accepted["correlation_energy"], v)
    if sum(occupied) < 2:
        assert v == 0 and all(not x.any() for x in blocks.values())

    quadratic_operators = ("geminal", "squared_geminal", "double_commutator")
    for spin, n in zip(("alpha", "beta"), occupied, strict=True):
        fock["exchange_" + spin] = rng.normal(scale=0.2, size=(ext, ext))
        for op in quadratic_operators[1:]:
            data[f"raw/{op}/{spin}"] = rng.normal(scale=0.2, size=(aux, n, ext))
            data[f"metric/{op}"] = np.eye(aux)
    feeds = unrestricted_cc_f12_fixed_source_feeds(
        (*occupied, obs, ext, aux),
        data,
        fock,
        np.eye(aux),
        execute,
        frozen_occupied=frozen,
    )
    assert set(feeds) == {*blocks, "fixed_geminal_energy"}
    for name, block in blocks.items():
        np.testing.assert_array_equal(feeds[name], block)

    quadratic = np.zeros(2)
    for left, right, nl, nr, fl, fr in (
        ("alpha", "alpha", oa, oa, frozen[0], frozen[0]),
        ("alpha", "beta", oa, ob, *frozen),
        ("beta", "beta", ob, ob, frozen[1], frozen[1]),
    ):
        il, ir = range(nl - fl), range(nr - fr)
        same = left == right
        pairs = combinations(il, 2) if same else product(il, ir)
        for i, j in pairs:
            reference = {f"metric_{op}": data[f"metric/{op}"] for op in quadratic_operators}
            reference.update(select_i=np.eye(len(il))[i], select_j=np.eye(len(ir))[j])
            if not same:
                reference["cross_overlap"] = data["cross_overlap"]
            for side, spin, first, stop in (
                ("left", left, fl, nl),
                ("right", right, fr, nr),
            ):
                if side == "right" and same:
                    continue
                active = slice(first, stop)
                reference[f"fock_{side}"] = fock[f"fock_{spin}"]
                reference[f"exchange_{side}"] = fock[f"exchange_{spin}"]
                reference[f"fit_{side}"] = data[f"source/fit_{spin}"][:, active]
                for op in quadratic_operators:
                    reference[f"raw_{op}_{side}"] = data[f"raw/{op}/{spin}"][:, active]
            quadratic += _dense_quadratic_pair(reference, nl, nr, obs, (fl, fr))
    scalar = feeds["fixed_geminal_energy"]
    assert scalar.shape == () and scalar.dtype == np.float64
    np.testing.assert_allclose(scalar, v + quadratic[0] - quadratic[1], rtol=0, atol=2e-14)
    complete = execute(program, {**amplitudes, **feeds}).outputs
    np.testing.assert_equal(complete["correlation_energy"], scalar)


def _dense_quadratic_pair(
    values: dict[str, np.ndarray], no: int, nr: int, obs: int, frozen: tuple[int, int]
) -> np.ndarray:
    dl = values["fit_left"]
    dr = values.get("fit_right", dl)
    lf = values["fock_left"]
    rf = values.get("fock_right", lf)
    lk = values["exchange_left"]
    rk = values.get("exchange_right", lk)
    n = lf.shape[0]
    same = "cross_overlap" not in values
    i = values["select_i"]
    j = values["select_j"]
    fl, fr = frozen
    electron_pair = np.outer(i, j)
    pair_chi = np.zeros((n, n))
    pair_chi[fl:no, fr:nr] = electron_pair

    def cusp(matrix: np.ndarray) -> np.ndarray:
        if same:
            return (matrix - matrix.T) / 4
        s = values["cross_overlap"]
        return (3 * matrix + s @ matrix.T @ s) / 8

    def apply_operator(operator: str, occupied_pair: np.ndarray) -> np.ndarray:
        left = values[f"raw_{operator}_left"]
        right = values.get(f"raw_{operator}_right", left)
        full = (np.einsum("aip,ajq->pqij", dl, right)
                + np.einsum("aip,ajq->pqij", left, dr)
                - np.einsum("aip,ab,bjq->pqij", dl, values[f"metric_{operator}"], dr))
        return cusp(np.einsum("pqij,ij->pq", full, occupied_pair)).ravel()

    def projector(start: int, stop: int) -> np.ndarray:
        return np.eye(n)[:, start:stop] @ np.eye(n)[start:stop]

    q = np.kron(projector(no, n), projector(nr, n)) - np.kron(projector(no, obs), projector(nr, obs))
    fock = np.kron(lf, np.eye(n)) + np.kron(np.eye(n), rf)
    exchange = np.kron(lk, np.eye(n)) + np.kron(np.eye(n), rk)
    chi = cusp(pair_chi).ravel()
    f = apply_operator("geminal", electron_pair)
    ff = apply_operator("squared_geminal", electron_pair)
    u = apply_operator("double_commutator", electron_pair)
    # Form the complete QFQ and 1-Q matrices. The reference acts with the
    # dense occupied two-particle Fock before evaluating either f or f².
    b = chi @ u + ff @ (fock + exchange) @ chi + f @ (q @ fock @ q - fock - exchange) @ f
    ofock = np.kron(lf[fl:no, fl:no], np.eye(nr-fr)) + np.kron(np.eye(no-fl), rf[fr:nr, fr:nr])
    acted = (ofock @ electron_pair.ravel()).reshape(electron_pair.shape)
    x = chi @ apply_operator("squared_geminal", acted) - f @ (np.eye(n*n)-q) @ apply_operator("geminal", acted)
    return np.array([b, x]) / (2 if same else 1)


@pytest.mark.parametrize("channel,frozen", [("alpha_alpha", (1, 1)), ("alpha_beta", (1, 0)), ("beta_beta", (1, 1))])
def test_quadratic_pair_uses_full_active_occupied_fock_and_transpose(
    channel: str, frozen: tuple[int, int]
) -> None:
    no, nr, obs, ext, aux = 3, 2 if channel == "alpha_beta" else 3, 4, 6, 5
    graph = unrestricted_fixed_geminal_quadratic_pair((no, obs-no, ext-obs), (nr, obs-nr, ext-obs), aux,
                                                      channel=channel, frozen_occupied=frozen)
    rng = np.random.default_rng(3781)
    feeds = {n.attrs['name']: rng.normal(scale=0.3,size=n.spec.shape) for n in graph.live_nodes if n.op=='input'}
    for key in ('fock_left','fock_right','exchange_left','exchange_right'):
        if key in feeds:
            feeds[key] = (feeds[key] + feeds[key].T) / 2
    feeds['select_i'] = np.eye(no-frozen[0])[0]
    feeds['select_j'] = np.eye(nr-frozen[1])[1]
    if channel == 'alpha_beta':
        feeds['cross_overlap'] = np.linalg.qr(rng.normal(size=(ext,ext)))[0]
    order = ('b_energy','x_fock_energy')
    def oracle(f: dict[str, np.ndarray]) -> np.ndarray:
        return _dense_quadratic_pair(f,no,nr,obs,frozen)
    values = execute(graph,feeds).outputs
    ref = oracle(feeds)
    np.testing.assert_allclose([values[k] for k in order],ref,rtol=0,atol=1e-14)
    # Actual off-diagonal occupied Fock matrix elements contribute to X;
    # replacing it with the orbital energy sum would erase the response.
    diagonal = {**feeds, **{k:np.diag(np.diag(feeds[k])) for k in ('fock_left', 'fock_right') if k in feeds}}
    assert abs(ref[1]-oracle(diagonal)[1])>1e-7
    seeds = rng.normal(size=2)
    bars = execute(transpose_program(graph,order).program,
                   {**feeds,**{'bar_'+k:np.array(x) for k,x in zip(order,seeds,strict=True)}}).outputs
    dirs = {k:rng.normal(scale=0.03,size=x.shape) for k,x in feeds.items() if not k.startswith('select_')}
    tangent = sum(np.vdot(bars['bar_'+k],x) for k,x in dirs.items())
    for step in (4e-4,2e-4):
        plus,minus = (oracle({**feeds,**{k:feeds[k]+sign*step*x for k,x in dirs.items()}}) for sign in (1,-1))
        np.testing.assert_allclose(tangent,np.dot(seeds,plus-minus)/(2*step),rtol=0,atol=1e-8)
