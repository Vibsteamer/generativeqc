"""Density-fitted coupled UHF orbital Hessian for unrestricted F12 references."""

from ..tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    multiply,
    slice_tensor,
    transpose,
    transpose_program,
)


def _field(spin: str, key: str, *axes: Index) -> Node:
    return input_tensor(
        key + "_" + spin, TensorSpec(axes, role="input", differentiable=True)
    )


def unrestricted_df_orbital_action(
    nalpha: int, nbeta: int, orbitals: int, auxiliaries: int
) -> Program:
    """Apply the real, occupied/virtual spin rotation Hessian without forming it.

    B matrices use the same Coulomb-metric whitener for both sets of spatial
    orbitals. Both spin variations contribute to J. K has the genuine same-spin
    direct and exchanged occupied/virtual contractions. Vector order is a,i.
    """
    if (
        any(type(x) is not int for x in (nalpha, nbeta, orbitals, auxiliaries))
        or min(nalpha, nbeta) < 0
        or nalpha + nbeta == 0
        or max(nalpha, nbeta) >= orbitals
        or auxiliaries < 1
    ):
        raise ValueError(
            "Need occupied electrons and nonempty virtual and fitting spaces"
        )
    aux = Index("A", IndexSpace("f12_fitting", "auxiliary", auxiliaries))
    exchange = {}
    vo = {}
    charge = []
    for spin, occ in (("alpha", nalpha), ("beta", nbeta)):
        if not occ:
            continue
        occ_space = IndexSpace("occupied_" + spin, "occupied", occ, spin=spin)
        virt_space = IndexSpace("virtual_" + spin, "virtual", orbitals - occ, spin=spin)
        i, j = Index("i", occ_space), Index("j", occ_space)
        a, b = Index("a", virt_space), Index("b", virt_space)

        vv, oo = _field(spin, "vv", aux, a, b), _field(spin, "oo", aux, i, j)
        kernel = _field(spin, "vo", aux, a, i)
        trial = _field(spin, "trial", a, i)
        one = _field(spin, "gap", a, i)
        charge.append(einsum("Aai,ai->A", kernel, trial))
        kv = einsum("Aab,bj->Aaj", vv, trial)
        ki = einsum("Abi,bj->Aij", kernel, trial)
        exchange[spin] = add(
            multiply(one, trial),
            einsum("Aaj,Aij->ai", kv, oo),
            einsum("Aaj,Aij->ai", kernel, ki),
            coefficients=(1, -1, -1),
        )
        vo[spin] = kernel
    total = add(*charge)
    return Program(
        {
            spin: add(part, einsum("Aai,A->ai", vo[spin], total), coefficients=(1, 2))
            for spin, part in exchange.items()
        },
        provenance={
            "method": "coupled canonical DF-UHF occupied-virtual response",
            "layout": "virtual-major",
        },
    )


def unrestricted_df_canonical_driver(
    nalpha: int, nbeta: int, orbitals: int, auxiliaries: int
) -> Program:
    """Differentiate the UHF Fock matrices against spin-resolved canonical seeds."""
    if (any(type(x) is not int for x in (nalpha, nbeta, orbitals, auxiliaries))
            or min(nalpha, nbeta) < 0 or not nalpha + nbeta
            or max(nalpha, nbeta) >= orbitals or auxiliaries < 1):
        raise ValueError("Need occupied electrons and nonempty virtual and fitting spaces")
    aux = Index("A", IndexSpace("f12_fitting", "auxiliary", auxiliaries))
    charges, exchanges, blocks = [], {}, {}
    for spin, occ in (("alpha", nalpha), ("beta", nbeta)):
        if not occ:
            continue
        orbital = IndexSpace("OBS_" + spin, "orbital", orbitals, spin=spin)
        p, q = Index("p", orbital), Index("q", orbital)
        factors = _field(spin, "df", aux, p, q)
        seed = _field(spin, "fock_seed", p, q)
        charges.append(einsum("Apq,pq->A", factors, seed))
        weighted = einsum("Api,ij->Apj", factors, seed)
        exchanges[spin] = einsum("Apj,Aqj->pq", weighted, factors)
        blocks[spin] = (occ, factors)
    charge = add(*charges)
    channels = {}
    for spin, (occ, factors) in blocks.items():
        field = add(einsum("Apq,A->pq", factors, charge), exchanges[spin], coefficients=(1, -1))
        channels[spin] = slice_tensor(add(field, transpose(field, (1, 0))),
                                      ((occ, orbitals), (0, occ)))
    return Program(channels, provenance={"method": "Canonical Fock cotangent to coupled DF-UHF orbital driver"})


def unrestricted_df_nuclear_direction(
    nalpha: int, nbeta: int, orbitals: int, auxiliaries: int, ao_functions: int
) -> Program:
    """Canonical UHF nuclear equation and fixed gradient terms from AO tangents.

    Supply raw three-center Coulomb derivatives and their two-center metric
    derivative in the common fitting basis. W is a fixed central whitener with
    W J Wᵀ = I. The symmetric gauge dB = W dR − (W dJ Wᵀ) B / 2 yields the
    physical derivative of the DF product Bᵀ B for any such W. Coefficients and
    orbital energies are the central SCF root. Fock and overlap seeds represent
    the observable's fixed canonical-frame contractions in each spin basis.
    """
    dims = (nalpha, nbeta, orbitals, auxiliaries, ao_functions)
    if (
        any(type(x) is not int for x in dims)
        or min(nalpha, nbeta) < 0
        or nalpha + nbeta == 0
        or max(nalpha, nbeta) >= orbitals
        or auxiliaries < 1
        or ao_functions < orbitals
    ):
        raise ValueError("Invalid spin occupations, OBS, AO or fitting dimensions")
    aux = IndexSpace("f12_fitting", "auxiliary", auxiliaries)
    ao = IndexSpace("AO", "orbital", ao_functions)
    a, t = Index("a", aux), Index("t", aux)
    m, n = Index("m", ao), Index("n", ao)

    def source(name: str, *axes: Index) -> Node:
        return input_tensor(name, TensorSpec(axes, role="input", differentiable=True))

    w = source("whitener", a, t)
    dj = source("dmetric", a, t)
    dr = source("dthree_center", t, m, n)
    ds, dh = source("doverlap", m, n), source("dhcore", m, n)
    metric = einsum("av,bv->ab", einsum("at,tv->av", w, dj), w)
    traces, dtraces, metric_traces, fields = [], [], [], {}
    for spin, o in (("alpha", nalpha), ("beta", nbeta)):
        frame = IndexSpace("obs_" + spin, "orbital", orbitals, spin=spin)
        p, q = Index("p", frame), Index("q", frame)
        c = _field(spin, "coefficients", m, p)
        b = _field(spin, "df", a, p, q)
        seed, sseed = (
            _field(spin, "fock_seed", p, q),
            _field(spin, "overlap_seed", p, q),
        )
        raw_mo = einsum("tmp,mq->tqp", einsum("tmn,np->tmp", dr, c), c)
        db = add(
            einsum("at,tpq->apq", w, raw_mo),
            einsum("at,tpq->apq", metric, b),
            coefficients=(1, "-1/2"),
        )
        sm = einsum("mp,mq->pq", c, einsum("mn,nq->mq", ds, c))
        hm = einsum("mp,mq->pq", c, einsum("mn,nq->mq", dh, c))
        if o:
            occ = (0, o)
            full = (0, orbitals)
            auxiliary = (0, auxiliaries)
            oo = slice_tensor(b, (auxiliary, occ, occ))
            doo = slice_tensor(db, (auxiliary, occ, occ))
            bo, dbo = (slice_tensor(x, (auxiliary, full, occ)) for x in (b, db))
            soo = slice_tensor(sm, (occ, occ))
            traces.append(einsum("aii->a", oo))
            dtraces.append(einsum("aii->a", doo))
            metric_traces.append(einsum("aij,ij->a", oo, soo))
            ex = einsum("api,aqi->pq", dbo, bo)
            dcore = add(hm, ex, transpose(ex, (1, 0)), coefficients=(1, -1, -1))
            metric_exchange = einsum("apj,aqj->pq", einsum("api,ij->apj", bo, soo), bo)
            eps = _field(spin, "occupied_energies", Index("i", frame, 0, o))
        else:
            dcore, metric_exchange, eps = hm, None, None
        fields[spin] = o, b, db, sm, dcore, metric_exchange, eps, seed, sseed

    charge, dcharge, metric_charge = add(*traces), add(*dtraces), add(*metric_traces)
    outputs = {}
    for spin, (
        o,
        b,
        db,
        sm,
        dcore,
        metric_exchange,
        eps,
        seed,
        sseed,
    ) in fields.items():
        explicit = add(
            dcore, einsum("apq,a->pq", b, dcharge), einsum("apq,a->pq", db, charge)
        )
        gmetric = einsum("apq,a->pq", b, metric_charge, coefficient=-1)
        if metric_exchange is not None:
            gmetric = add(gmetric, metric_exchange)
        outputs["fixed_fock_" + spin] = einsum("pq,pq->", seed, explicit)
        outputs["metric_density_" + spin] = einsum("pq,pq->", seed, gmetric)
        outputs["fixed_overlap_" + spin] = einsum("pq,pq->", sseed, sm)
        if o:
            vo = ((o, orbitals), (0, o))
            outputs["rhs_" + spin] = add(
                einsum("ai,i->ai", slice_tensor(sm, vo), eps),
                slice_tensor(explicit, vo),
                slice_tensor(gmetric, vo),
                coefficients=(1, -1, -1),
            )
    return Program(
        outputs,
        provenance={
            "method": "canonical DF-UHF AO nuclear equation and fixed gradients",
            "df_metric_gauge": "symmetric",
        },
    )


def unrestricted_df_nuclear_weights(
    nalpha: int,
    nbeta: int,
    orbitals: int,
    auxiliaries: int,
    ao_functions: int,
    *,
    include_direct: bool = False,
) -> Program:
    """Transpose the nuclear equations into four symmetric Gaussian weights.

    Feed the solved occupied/virtual adjoint as the RHS cotangent and one as
    each fixed energy-gradient cotangent. Full Frobenius weights include both
    off-diagonal entries and contract directly with physical integral sinks.
    """
    primal = unrestricted_df_nuclear_direction(
        nalpha, nbeta, orbitals, auxiliaries, ao_functions
    )
    fields = {
        "doverlap": "overlap_ao",
        "dhcore": "hcore_ao",
        "dmetric": "coulomb_metric",
        "dthree_center": "ao_coulomb",
    }
    reverse = transpose_program(
        primal, tuple(primal.outputs), inputs=tuple(fields)
    ).program
    outputs = {}
    for key, name in fields.items():
        weight = reverse.outputs["bar_" + key]
        axes = (0, 2, 1) if key == "dthree_center" else (1, 0)
        pair = add(weight, transpose(weight, axes), coefficients=("1/2", "1/2"))
        if include_direct:
            spec = TensorSpec(pair.spec.indices, role="input", differentiable=True)
            pair = add(pair, input_tensor("direct_" + name, spec))
            if key == "doverlap":
                pair = add(
                    pair,
                    input_tensor("frame_alpha", spec),
                    input_tensor("frame_beta", spec),
                )
        outputs[name] = pair
    return Program(
        outputs, provenance={"method": "symmetric DF-UHF nuclear source cotangents"}
    )
