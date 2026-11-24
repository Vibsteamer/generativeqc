"""Coupled unrestricted orbital response for real, canonical DF-UHF F12 forces."""

import ctypes
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
from generativeqc_compiler.method.canonical_frame_response import (
    orbital_metric_cotangent,
    orbital_rotation_cotangent,
)
from generativeqc_compiler.mp2.f12_unrestricted_reference import (
    unrestricted_df_canonical_driver,
    unrestricted_df_nuclear_direction,
    unrestricted_df_nuclear_weights,
)

from . import _native
from ._canonical_frame import canonical_full_frame_seeds
from .elements import checked_integer


class UnrestrictedOrbitalForces(NamedTuple):
    rotations: np.ndarray
    orbital_gradient: np.ndarray
    forces: np.ndarray


def _block(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != shape
        or array.dtype != np.float64
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must contain finite FP64 values of shape {shape}")
    return array.ravel(order="C")


class UnrestrictedNuclearSourceTerms(NamedTuple):
    nuclear_rhs: np.ndarray
    fixed_gradients: dict[str, np.ndarray]


def _nuclear_center(
    central: Mapping[str, np.ndarray], oa: int, ob: int
) -> dict[str, np.ndarray]:
    fields = {"whitener": central["whitener"]}
    for spin, occ in (("alpha", oa), ("beta", ob)):
        for key in ("coefficients", "df", "fock_seed", "overlap_seed"):
            name = f"{key}_{spin}"
            fields[name] = central[name]
        if occ:
            name = f"occupied_energies_{spin}"
            fields[name] = central[name]
    return fields


def unrestricted_frame_overlap_weights(
    central: Mapping[str, np.ndarray],
    complement: np.ndarray,
    direct: Mapping[str, np.ndarray],
    run: Callable,
) -> dict[str, np.ndarray]:
    """Derive the two AO overlap cotangents from the matching spin-MO/CABS frames."""
    cabs = np.asarray(complement)
    if cabs.ndim != 2 or not cabs.shape[1] or cabs.dtype != np.float64 or not np.isfinite(cabs).all():
        raise ValueError("Expected finite FP64 CABS columns")
    atomic, rank = cabs.shape
    weights = {}
    for spin in ("alpha", "beta"):
        basis = np.asarray(central["coefficients_" + spin])
        if basis.ndim != 2 or basis.shape[0] != atomic or not basis.shape[1] or basis.dtype != np.float64 or not np.isfinite(basis).all():
            raise ValueError("OBS and CABS coefficients must share the same AO rows")
        obs = basis.shape[1]
        retained = obs + rank
        bar = np.asarray(direct["coefficients_" + spin])
        _block(bar, (atomic, retained), "extended coefficient cotangent")
        # Relative gauge of bar and CABS columns is fixed by the forward phase convention.
        coefficients = np.column_stack((basis, cabs))
        mask = np.full((retained, retained), -0.5)
        mask[:obs, obs:] = -1.0
        mask[obs:, :obs] = 0.0
        weight = run(orbital_metric_cotangent(atomic, retained), {
            "coefficients": coefficients, "frame_bar": bar, "metric_blocks": mask,
        }).outputs["overlap_bar"]
        _block(weight, (atomic, atomic), "frame overlap cotangent")
        weights[spin] = weight
    return weights


def unrestricted_canonical_source_seeds(
    occupations: tuple[int, int],
    coefficients: Mapping[str, np.ndarray],
    energies: Mapping[str, np.ndarray],
    frame_cotangents: Mapping[str, np.ndarray],
    run: Callable,
    *,
    gap_threshold: float = 1e-8,
    stationarity_threshold: float = 1e-9,
    maximum_bytes: int = 128 << 20,
    device_id: int = 0,
    library: ctypes.CDLL | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Build central Fock and overlap weights from the two spin frame cotangents.

    Supply matching OBS columns, canonical energies and AO coefficient cotangents
    from one reference. A cotangent with complete OBS+CABS columns is accepted;
    the separate CABS frame response consumes the complement columns. The full
    OBS seeds include occupied/virtual couplings with the relaxed DF-UHF Fock.
    The caller applies orbital response only once through its coupled equation.
    """
    result, torques = {}, {}
    for spin, occ in zip(("alpha", "beta"), occupations, strict=True):
        c = np.asarray(coefficients["coefficients_" + spin])
        e = np.asarray(energies[spin])
        bar = np.asarray(frame_cotangents["coefficients_" + spin])
        n = e.size
        if (
            c.ndim != 2
            or c.shape[1] != n
            or e.shape != (n,)
            or bar.ndim != 2
            or bar.shape[0] != c.shape[0]
            or bar.shape[1] < n
            or any(v.dtype != np.float64 for v in (c, e, bar))
            or not np.isfinite(e).all()
        ):
            raise ValueError(
                "Inconsistent canonical orbitals, energies or AO frame cotangents"
            )
        graph = orbital_rotation_cotangent(c.shape[0], n)
        m = run(
            graph, {"coefficients": c, "frame_bar": np.ascontiguousarray(bar[:, :n])}
        ).outputs["rotation_gradient"]
        f, s, t, _ = canonical_full_frame_seeds(
            m,
            e,
            occ,
            gap_threshold=gap_threshold,
            stationarity_threshold=stationarity_threshold,
            maximum_bytes=maximum_bytes,
            device_id=device_id,
            library=library,
        )
        result["fock_seed_" + spin] = f
        result["overlap_seed_" + spin] = s
        torques[spin] = t
    return result, torques


def unrestricted_nuclear_source_terms(
    space: tuple[int, int, int, int],
    central: Mapping[str, np.ndarray],
    directions: Iterable[Mapping[str, np.ndarray]],
    atoms: int,
    run: Callable,
) -> UnrestrictedNuclearSourceTerms:
    """Build DF-UHF orbital equations and six fixed terms on the supplied backend.

    Iterate over the raw AO tangents in atom-major x,y,z order. Each direction
    supplies doverlap, dhcore, dthree_center and dmetric. Both spin frames use
    the same central metric whitener and AO basis. The dense central df blocks
    must use that whitener. Fock and overlap seeds are in the corresponding
    central MO frame. No orbital derivative or preassembled nuclear RHS enters.
    """
    oa, ob, n, q = space
    n_atoms = checked_integer(atoms, "atoms", low=1)
    coeff = np.asarray(central["coefficients_alpha"])
    if coeff.ndim != 2:
        raise ValueError("Expected an AO-by-OBS alpha coefficient matrix")
    program = unrestricted_df_nuclear_direction(oa, ob, n, q, coeff.shape[0])
    keys = ("dmetric", "dthree_center", "doverlap", "dhcore")
    spin_pairs = (("alpha", oa), ("beta", ob))
    fields = _nuclear_center(central, oa, ob)
    rhs = np.zeros((n_atoms, 3, oa * (n - oa) + ob * (n - ob)))
    fixed = {
        f"{key}_{spin}": np.empty((n_atoms, 3))
        for spin, _ in spin_pairs
        for key in ("fixed_fock", "metric_density", "fixed_overlap")
    }
    for (atom, axis), tangent in zip(np.ndindex(n_atoms, 3), directions, strict=True):
        inputs = {**fields, **{key: tangent[key] for key in keys}}
        result = run(program, inputs).outputs
        rhs[atom, axis] = np.concatenate(
            [result["rhs_" + spin].ravel() for spin, occ in spin_pairs if occ]
        )
        for key, channel in fixed.items():
            channel[atom, axis] = result[key].item()
    if not np.isfinite(rhs).all() or any(
        not np.isfinite(a).all() for a in fixed.values()
    ):
        raise ValueError("Nonfinite DF-UHF nuclear equation or fixed gradient")
    return UnrestrictedNuclearSourceTerms(rhs, fixed)


def _run_unrestricted_nuclear_weights_tiled(
    space: tuple[int, int, int, int],
    central: Mapping[str, np.ndarray],
    z: np.ndarray,
    run: Callable,
    *,
    direct_weights: Mapping[str, np.ndarray] | None = None,
    frame_overlap: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Bounded-memory auxiliary-tiled execution of DF-UHF nuclear source weights on `run`."""
    from generativeqc_compiler.tensor import Index, IndexSpace, Program, TensorSpec, einsum, input_tensor

    from ._f12_spin_fock import _tile_groups, default_auxiliary_batch_size
    from ._f12_spin_pairs import batch_execute

    oa, ob, n, q = space
    w = np.ascontiguousarray(central["whitener"])
    nao = central["coefficients_alpha"].shape[0]
    spins = (("alpha", oa), ("beta", ob))

    # Build effective MO seeds P_s (for explicit and gmetric) and bar_sm_s (for overlap)
    p_seeds: dict[str, np.ndarray] = {}
    bar_sm: dict[str, np.ndarray] = {}
    offset = 0
    for spin, occ in spins:
        p = np.array(central["fock_seed_" + spin], dtype=np.float64, copy=True)
        sm = np.array(central["overlap_seed_" + spin], dtype=np.float64, copy=True)
        if occ:
            size = (n - occ) * occ
            z_s = z[offset : offset + size].reshape(n - occ, occ)
            offset += size
            p[occ:, :occ] -= z_s
            sm[occ:, :occ] += z_s * central["occupied_energies_" + spin][None, :]
        p_seeds[spin] = p
        bar_sm[spin] = sm

    # Check if C_s is supported purely on the leading n OBS rows (standard OBS+CABS layout)
    obs_only_ao = nao > n and all(
        np.count_nonzero(central["coefficients_" + spin][n:, :]) == 0 for spin, _ in spins
    )
    n_eff = n if obs_only_ao else nao
    c_eff = {
        spin: np.ascontiguousarray(central["coefficients_" + spin][:n_eff, :])
        for spin, _ in spins
    }
    same_spin = (
        oa == ob
        and np.array_equal(c_eff["alpha"], c_eff["beta"])
        and (
            central["df_alpha"] is central["df_beta"]
            or np.array_equal(central["df_alpha"], central["df_beta"])
        )
        and np.array_equal(p_seeds["alpha"], p_seeds["beta"])
    )

    width = default_auxiliary_batch_size(q, max(n, n_eff))
    groups = _tile_groups(q, width)

    # 1) charge[a] and bar_dcharge[a] via tiled contractions on `run`
    charge = np.zeros(q, dtype=np.float64)
    bar_dcharge = np.zeros(q, dtype=np.float64)
    mo_sp = IndexSpace("obs", "orbital", n)
    p_i, q_i = Index("p", mo_sp), Index("q", mo_sp)
    for tile_len, slices in groups:
        aux_sp = IndexSpace("aux", "auxiliary", tile_len)
        a_i = Index("a", aux_sp)
        b_in = input_tensor("b", TensorSpec((a_i, p_i, q_i), role="input"))
        p_in = input_tensor("p", TensorSpec((p_i, q_i), role="input"))
        dot_prog = Program({"d": einsum("pq,apq->a", p_in, b_in)})
        with batch_execute(run, dot_prog) as exec_dot:
            for spin, occ in spins:
                if same_spin and spin == "beta":
                    continue
                scale = 2.0 if same_spin else 1.0
                b_s = central["df_" + spin]
                p_c = np.ascontiguousarray(p_seeds[spin])
                for rows in slices:
                    b_tile = np.ascontiguousarray(b_s[rows])
                    bar_dcharge[rows] += scale * exec_dot({"p": p_c, "b": b_tile}).outputs["d"]
                    if occ:
                        charge[rows] += scale * np.einsum("aii->a", b_tile[:, :occ, :occ])
    bar_metric_charge = -bar_dcharge

    # 2) Complete bar_sm_s (metric_charge and metric_exchange) and assemble 2D hcore/overlap weights
    bar_dh = np.zeros((nao, nao), dtype=np.float64)
    bar_ds = np.zeros((nao, nao), dtype=np.float64)
    ao_sp = IndexSpace("ao_eff", "orbital", n_eff)
    m_i, v_i = Index("m", ao_sp), Index("n", ao_sp)
    c_in = input_tensor("c", TensorSpec((m_i, p_i), role="input"))
    mat_in = input_tensor("m", TensorSpec((p_i, q_i), role="input"))
    cong_prog = Program({"out": einsum("mp,np->mn", c_in, einsum("nq,pq->np", c_in, mat_in))})
    for spin, occ in spins:
        if occ:
            b_s = central["df_" + spin]
            p_s = p_seeds[spin]
            occ_sp = IndexSpace("occ_" + spin, "occupied", occ)
            i_i, j_i = Index("i", occ_sp), Index("j", occ_sp)
            for tile_len, slices in groups:
                aux_sp = IndexSpace("aux", "auxiliary", tile_len)
                a_i = Index("a", aux_sp)
                bo_in = input_tensor("bo", TensorSpec((a_i, p_i, i_i), role="input"))
                oo_in = input_tensor("oo", TensorSpec((a_i, i_i, j_i), role="input"))
                mc_in = input_tensor("mc", TensorSpec((a_i,), role="input"))
                pm_in = input_tensor("pm", TensorSpec((p_i, q_i), role="input"))
                pb = einsum("pq,api->aqi", pm_in, bo_in)
                sm_prog = Program(
                    {
                        "s_mc": einsum("a,aij->ij", mc_in, oo_in),
                        "s_ex": einsum("aqi,aqj->ij", pb, bo_in),
                    }
                )
                p_c = np.ascontiguousarray(p_s)
                with batch_execute(run, sm_prog) as exec_sm:
                    for rows in slices:
                        bo_tile = np.ascontiguousarray(b_s[rows, :, :occ])
                        oo_tile = np.ascontiguousarray(b_s[rows, :occ, :occ])
                        out = exec_sm(
                            {
                                "bo": bo_tile,
                                "oo": oo_tile,
                                "mc": np.ascontiguousarray(bar_metric_charge[rows]),
                                "pm": p_c,
                            }
                        ).outputs
                        bar_sm[spin][:occ, :occ] += out["s_mc"] + out["s_ex"]
        c_s = c_eff[spin]
        bar_dh[:n_eff, :n_eff] += run(
            cong_prog, {"c": c_s, "m": np.ascontiguousarray(p_seeds[spin])}
        ).outputs["out"]
        bar_ds[:n_eff, :n_eff] += run(
            cong_prog, {"c": c_s, "m": np.ascontiguousarray(bar_sm[spin])}
        ).outputs["out"]

    # 3) Tiled bar_dB -> bar_M (q x q) and bar_dr (q x nao x nao) on `run`
    bar_m = np.zeros((q, q), dtype=np.float64)
    if direct_weights is not None:
        base_ao = direct_weights["ao_coulomb"]
        bar_dr = np.array(base_ao, dtype=np.float64, copy=True)
    else:
        bar_dr = np.zeros((q, nao, nao), dtype=np.float64)

    for tile_len, slices in groups:
        aux_sp = IndexSpace("aux", "auxiliary", tile_len)
        a_i = Index("a", aux_sp)
        # AO back-projection per chunk: ao_db[a, m, n] = sum_p,q C[m, p] bar_db[a, p, q] C[n, q]
        db_in = input_tensor("db", TensorSpec((a_i, p_i, q_i), role="input"))
        proj_prog = Program(
            {"ao": einsum("mp,apn->amn", c_in, einsum("apq,nq->apn", db_in, c_in))}
        )
        # Metric cross-contraction chunk (a_tile vs b_tile)
        t_groups = _tile_groups(q, width)
        # Prebuild per-spin exchange contraction programs for this tile_len
        ex_progs = {}
        for spin, occ in spins:
            if occ:
                occ_sp = IndexSpace("occ_" + spin, "occupied", occ)
                i_i = Index("i", occ_sp)
                bo_in = input_tensor("bo", TensorSpec((a_i, q_i, i_i), role="input"))
                ps_in = input_tensor("ps", TensorSpec((p_i, q_i), role="input"))
                ex_progs[spin] = Program({"ex": einsum("pq,aqi->api", ps_in, bo_in)})

        # Open batch executors for projection and W-transform
        with batch_execute(run, proj_prog) as exec_proj:
            for rows in slices:
                ao_db_chunk = np.zeros((tile_len, n_eff, n_eff), dtype=np.float64)
                for spin, occ in spins:
                    if same_spin and spin == "beta":
                        continue
                    scale = 2.0 if same_spin else 1.0
                    b_s = central["df_" + spin]
                    p_s = p_seeds[spin]
                    bar_db = charge[rows, None, None] * p_s[None, :, :]
                    if occ:
                        idx = np.arange(occ)
                        bar_db[:, idx, idx] += bar_dcharge[rows, None]
                        psym = np.ascontiguousarray(p_s + p_s.T)
                        ex_term = run(
                            ex_progs[spin],
                            {"ps": psym, "bo": np.ascontiguousarray(b_s[rows, :, :occ])},
                        ).outputs["ex"]
                        bar_db[:, :, :occ] -= ex_term
                    # Accumulate bar_M[rows, b_rows] via tiled GPU contractions
                    bar_db_c = np.ascontiguousarray(bar_db)
                    for b_len, b_slices in t_groups:
                        b_sp = IndexSpace("aux_b", "auxiliary", b_len)
                        b_i = Index("b", b_sp)
                        b_in = input_tensor("b", TensorSpec((b_i, p_i, q_i), role="input"))
                        m_prog = Program({"m": einsum("apq,bpq->ab", db_in, b_in)})
                        with batch_execute(run, m_prog) as exec_m:
                            for b_rows in b_slices:
                                bar_m[rows, b_rows] += (-0.5 * scale) * exec_m(
                                    {"db": bar_db_c, "b": np.ascontiguousarray(b_s[b_rows])}
                                ).outputs["m"]
                    ao_part = exec_proj({"c": c_eff[spin], "db": bar_db_c}).outputs["ao"]
                    ao_db_chunk += scale * ao_part
                # Symmetrize ao_db_chunk before W^T accumulation (linear, identical to symmetrizing bar_dr)
                ao_db_sym = np.ascontiguousarray(0.5 * (ao_db_chunk + ao_db_chunk.swapaxes(1, 2)))
                ao_in = input_tensor("ao", TensorSpec((a_i, m_i, v_i), role="input"))
                for t_len, t_slices in t_groups:
                    t_sp = IndexSpace("aux_t", "auxiliary", t_len)
                    t_i = Index("t", t_sp)
                    w_in = input_tensor("w", TensorSpec((a_i, t_i), role="input"))
                    wt_prog = Program({"dr": einsum("at,amn->tmn", w_in, ao_in)})
                    with batch_execute(run, wt_prog) as exec_wt:
                        for t_rows in t_slices:
                            bar_dr[t_rows, :n_eff, :n_eff] += exec_wt(
                                {"w": np.ascontiguousarray(w[rows, t_rows]), "ao": ao_db_sym}
                            ).outputs["dr"]

    # 4) bar_dj = W^T @ bar_M @ W and symmetrize 2D outputs
    q_sp = IndexSpace("aux_full", "auxiliary", q)
    a_f, b_f, t_f, u_f = Index("a", q_sp), Index("b", q_sp), Index("t", q_sp), Index("u", q_sp)
    wf_in = input_tensor("w", TensorSpec((a_f, t_f), role="input"))
    mf_in = input_tensor("m", TensorSpec((a_f, b_f), role="input"))
    dj_prog = Program({"dj": einsum("at,au->tu", wf_in, einsum("ab,bu->au", mf_in, wf_in))})
    bar_dj = run(dj_prog, {"w": w, "m": np.ascontiguousarray(bar_m)}).outputs["dj"]

    bar_dj = 0.5 * (bar_dj + bar_dj.T)
    bar_dh = 0.5 * (bar_dh + bar_dh.T)
    bar_ds = 0.5 * (bar_ds + bar_ds.T)
    if direct_weights is not None and frame_overlap is not None:
        bar_ds = bar_ds + direct_weights["overlap_ao"] + frame_overlap["alpha"] + frame_overlap["beta"]
        bar_dh = bar_dh + direct_weights["hcore_ao"]
        bar_dj = bar_dj + direct_weights["metric/coulomb"]
    weights = {
        "overlap_ao": bar_ds,
        "hcore_ao": bar_dh,
        "ao_coulomb": bar_dr,
        "metric/coulomb": bar_dj,
    }
    if any(not np.isfinite(val).all() for val in weights.values()):
        raise ValueError("Nonfinite DF-UHF nuclear source weights")
    return weights


def unrestricted_nuclear_source_weights(
    space: tuple[int, int, int, int],
    central: Mapping[str, np.ndarray],
    orbital_adjoint: np.ndarray,
    run: Callable,
    *,
    direct_weights: Mapping[str, np.ndarray] | None = None,
    frame_overlap: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Prepare orbital-response gradient weights for the Gaussian nuclear sinks.

    Each returned AO/metric array contracts with a derivative of the same
    operator. Add these four arrays to the direct energy weights; separately
    include the OBS/CABS frame overlap response. The adjoint and the central
    seeds must describe the same electronic observable and SCF root. This
    operation uses only central quantities and produces every atomic direction's
    weights in one reverse evaluation. To return effective weights for these four
    operators, supply their direct energy cotangents and both spin frame-overlap
    cotangents. The other Gaussian operator weights pass unchanged to their sinks.
    """
    oa, ob, n, q = space
    coeff = np.asarray(central["coefficients_alpha"])
    if coeff.ndim != 2:
        raise ValueError("Expected an AO-by-OBS alpha coefficient matrix")
    include_direct = direct_weights is not None
    if include_direct != (frame_overlap is not None):
        raise ValueError(
            "Direct Gaussian and frame-overlap weights must be supplied together"
        )
    z = _block(orbital_adjoint, (oa * (n - oa) + ob * (n - ob),), "orbital adjoint")
    nao = coeff.shape[0]
    if q * max(n, nao) * max(n, nao) * 8 > (64 << 20):
        return _run_unrestricted_nuclear_weights_tiled(
            space,
            central,
            z,
            run,
            direct_weights=direct_weights,
            frame_overlap=frame_overlap,
        )
    graph = unrestricted_df_nuclear_weights(
        oa, ob, n, q, nao, include_direct=include_direct
    )
    fields = _nuclear_center(central, oa, ob)
    if direct_weights is not None and frame_overlap is not None:
        for name in ("overlap_ao", "hcore_ao", "ao_coulomb", "coulomb_metric"):
            fields["direct_" + name] = direct_weights[
                "metric/coulomb" if name == "coulomb_metric" else name
            ]
        for spin in ("alpha", "beta"):
            fields["frame_" + spin] = frame_overlap[spin]
    offset = 0
    for spin, occ in (("alpha", oa), ("beta", ob)):
        for key in ("fixed_fock", "metric_density", "fixed_overlap"):
            fields[f"bar_{key}_{spin}"] = np.array(1.0)
        if occ:
            size = (n - occ) * occ
            fields["bar_rhs_" + spin] = z[offset : offset + size].reshape(n - occ, occ)
            offset += size
    weights = dict(run(graph, fields).outputs)
    weights["metric/coulomb"] = weights.pop("coulomb_metric")
    if any(not np.isfinite(w).all() for w in weights.values()):
        raise ValueError("Nonfinite DF-UHF nuclear source weights")
    return weights


def unrestricted_orbital_forces(
    space: tuple[int, int, int, int],
    factors: Mapping[str, np.ndarray],
    electronic_driver: np.ndarray,
    nuclear_rhs: np.ndarray,
    reference_forces: np.ndarray,
    fixed_gradients: Sequence[np.ndarray] = (),
    *,
    device_id: int = 0,
    maximum_bytes: int = 1536 << 20,
    library: ctypes.CDLL | None = None,
) -> UnrestrictedOrbitalForces:
    """Solve Hᵀz=g and return F_ref − Σ fixed gradients − b_Rᵀz on CUDA.

    Space contains alpha/beta occupied counts, OBS orbitals and the shared DF
    rank. Supply vo, oo, vv and gap for each occupied spin under names such as
    vo_alpha. Use the same Coulomb whitening for both spins. Flatten the
    electronic driver in alpha-then-beta virtual-major, occupied-minor order.
    Nuclear right-hand sides have shape (atoms, 3, rotations). They solve
    H x_R=b_R. The fixed-frame energy gradients must exclude this rotation
    response. Reference forces and each fixed gradient have shape (atoms, 3).
    The returned orbital gradient uses the positive energy derivative sign.

    maximum_bytes bounds explicit CUDA arrays including pivot and LU workspaces.
    """
    if np.asarray(reference_forces).size == 0:
        raise ValueError("Expected nonempty reference forces")
    return _solve_orbital_response(space, factors, electronic_driver, nuclear_rhs, reference_forces,
                                   fixed_gradients, device_id, maximum_bytes, library)


def unrestricted_source_orbital_adjoint(
    space: tuple[int, int, int, int],
    central: Mapping[str, np.ndarray],
    canonical_energies: Mapping[str, np.ndarray],
    run: Callable,
    *,
    device_id: int = 0,
    maximum_bytes: int = 1536 << 20,
    library: ctypes.CDLL | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the canonical electronic driver and solve its transposed orbital system.

    The spin DF tensors and canonical Fock cotangents must use the same MO frame
    and shared auxiliary metric. Returns alpha-then-beta VO driver and rotations.
    This solve does not consume nuclear RHS, forces or an external orbital adjoint.
    """
    oa, ob, n, q = space
    from ._f12_spin_fock import _tile_groups, default_auxiliary_batch_size
    from ._f12_spin_pairs import batch_execute

    values, factors = {}, {}
    for spin, occ in (("alpha", oa), ("beta", ob)):
        if not occ:
            continue
        df = np.asarray(central["df_" + spin])
        seed = np.asarray(central["fock_seed_" + spin])
        eps = _block(canonical_energies[spin], (n,), "canonical energies")
        if df.shape != (q, n, n) or seed.shape != (n, n):
            raise ValueError("Canonical driver and fitted MO space disagree")
        values["df_" + spin] = df
        values["fock_seed_" + spin] = seed
        factors.update({key + "_" + spin: block for key, block in (
            ("vo", df[:, occ:, :occ]), ("oo", df[:, :occ, :occ]), ("vv", df[:, occ:, occ:]),
            ("gap", eps[occ:, None] - eps[None, :occ]))})
    width = default_auxiliary_batch_size(q, n)
    if width >= q:
        channels = run(unrestricted_df_canonical_driver(*space), values).outputs
    else:
        channels = {}
        for tile_len, slices in _tile_groups(q, width):
            graph = unrestricted_df_canonical_driver(oa, ob, n, tile_len)
            with batch_execute(run, graph) as exec_one:
                for rows in slices:
                    tile_vals = {}
                    for spin, occ in (("alpha", oa), ("beta", ob)):
                        if not occ:
                            continue
                        tile_vals["df_" + spin] = np.ascontiguousarray(values["df_" + spin][rows])
                        tile_vals["fock_seed_" + spin] = values["fock_seed_" + spin]
                    part = exec_one(tile_vals).outputs
                    for k, arr in part.items():
                        channels[k] = arr.copy() if k not in channels else channels[k] + arr
    driver = np.concatenate([_block(channels[spin], (n - occ, occ), "orbital driver")
                             for spin, occ in (("alpha", oa), ("beta", ob)) if occ])
    # Zero coordinates select the rotations-only ABI mode. No nuclear arrays are uploaded.
    result = _solve_orbital_response(space, factors, driver, np.empty((0, 3, driver.size)),
                                     np.empty((0, 3)), (), device_id, maximum_bytes, library)
    return driver, result.rotations


def _solve_orbital_response(space, factors, electronic_driver, nuclear_rhs, reference_forces,
                            fixed_gradients, device_id, maximum_bytes, library):
    oa, ob, n, q = (
        checked_integer(value, name)
        for value, name in zip(
            space,
            ("alpha occupation", "beta occupation", "OBS orbitals", "DF rank"),
            strict=True,
        )
    )
    if not oa + ob or n <= max(oa, ob) or q < 1:
        raise ValueError(
            "Expected occupied electrons, OBS virtuals and a nonempty fitting basis"
        )
    device = checked_integer(device_id, "CUDA device")
    budget = checked_integer(
        maximum_bytes, "workspace bytes", low=1, high=np.iinfo(np.uintp).max
    )
    rank = checked_integer(oa * (n - oa) + ob * (n - ob), "orbital rotations", low=1)
    ref = np.asarray(reference_forces)
    if ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError("Reference forces must have three Cartesian axes")
    directions = checked_integer(ref.size, "Cartesian directions")
    terms = checked_integer(len(fixed_gradients), "fixed gradient terms")
    if 8 * rank * rank > budget:
        raise ValueError(
            "The dense orbital Hessian exceeds the device workspace budget"
        )
    packed = []
    for spin, o in (("alpha", oa), ("beta", ob)):
        if not o:
            continue
        v = n - o
        for key, shape in (
            ("vo", (q, v, o)),
            ("oo", (q, o, o)),
            ("vv", (q, v, v)),
            ("gap", (v, o)),
        ):
            name = f"{key}_{spin}"
            packed.append(_block(factors[name], shape, name))
    packed.append(_block(electronic_driver, (rank,), "electronic driver"))
    packed.append(_block(nuclear_rhs, (*ref.shape, rank), "nuclear RHS"))
    packed.append(_block(ref, ref.shape, "reference forces"))
    packed.extend(_block(g, ref.shape, "fixed gradient") for g in fixed_gradients)
    if (sum(a.size for a in packed) + rank * rank + directions) * 8 + (
        rank + 1
    ) * 4 > budget:
        raise ValueError(
            "Molecular adjoint inputs and Hessian exceed the device workspace budget"
        )
    data = np.concatenate(packed)
    if library is None:
        library = _native.load_library(device="cuda", device_id=device)
    routine = library.generativeqc_uhf_orbital_adjoint_cuda_v1
    pointer = ctypes.POINTER(ctypes.c_double)
    routine.argtypes = [ctypes.c_int] * 7 + [
        pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        pointer,
        pointer,
        pointer,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    routine.restype = ctypes.c_int
    z, gradient, forces = np.empty(rank), np.empty(ref.shape), np.empty(ref.shape)
    error = ctypes.create_string_buffer(512)
    status = routine(
        device,
        oa,
        ob,
        n,
        q,
        directions,
        terms,
        data.ctypes.data_as(pointer),
        data.size,
        budget,
        z.ctypes.data_as(pointer),
        gradient.ctypes.data_as(pointer),
        forces.ctypes.data_as(pointer),
        error,
        len(error),
    )
    if status:
        raise RuntimeError(error.value.decode(errors="replace"))
    return UnrestrictedOrbitalForces(z, gradient, forces)
