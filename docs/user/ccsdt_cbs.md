# MP2-assisted CCSD(T)/CBS energy and forces

`CCSDTCBSCalculator` composes native HF, MP2 and CCSD(T) calculations. It uses
two larger bases for MP2 extrapolation and one smaller basis for the
CCSD(T)−MP2 correction. It does **not** run CCSD(T) in both larger bases, or
use a reference package or finite differences as a production force backend.

## Run the workflow

Supply checked local [canonical basis records](external_basis.md), with explicit
cardinal numbers from one consistent family. For example, TZ/QZ MP2 plus a DZ
correction:

```python
from generativeqc import CBSBasis, CCSDTCBSCalculator, load_basis

workflow = CCSDTCBSCalculator(
    (CBSBasis(load_basis("cc-pvtz.json"), 3),
     CBSBasis(load_basis("cc-pvqz.json"), 4)),
    correction_basis=load_basis("cc-pvdz.json"),
    device="cuda",
    calculator_options={"correlation_memory_budget_bytes": 2 << 30},
)
result = workflow.singlepoint(
    [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
    properties=("energy", "forces"),
)
print(result.energy, result.forces)
print(result.hf_energy, result.mp2_correlation_energy, result.delta_ccsdt_energy)
for component in result.components:
    print(component.roles, component.elapsed_seconds, component.result.correlation)
```

The example presupposes that those local records exist and their native
operator/derivative domains are admitted; names do not download basis data or
promise an executable GPU domain. Use `import_bse` for an explicit local BSE
conversion. The bundled named-basis subset is not a complete CBS basis library.
If `correction_basis` is omitted, the lower MP2 basis is also the correction
basis and its MP2 result is reused. The same reuse applies when the explicitly
supplied correction record has either MP2 basis's full identity.

Set `properties=("energy",)` to omit all analytic forces. The default requests
both energy and forces. Ionic charge is passed identically to every component;
singlets use RHF/MP2/CCSD(T), while higher multiplicities select
UHF/UMP2/UCCSD(T). See the open-shell limits below before requesting radicals
or high-spin ions.

## Open-shell energy and forces

Set `multiplicity` to the spin multiplicity (`2S + 1`). The default includes
analytic forces; energy-only execution remains available. A hydrogen doublet can use:

```python
workflow = CCSDTCBSCalculator(
    (CBSBasis("sto-3g", 2), CBSBasis("def2-svp", 3)),
    device="cpu",
)
result = workflow.singlepoint(
    [("H", (0.0, 0.0, 0.0))], multiplicity=2, properties=("energy",)
)
```

This small named-basis pair demonstrates the API, **not** a systematic CBS
sequence. Use a consistent basis family and suitable cardinals for scientific
extrapolation. Charge and multiplicity are validated and passed unchanged to
every basis group; `basis_metadata` records them and the selected
`spin_reference`. A singlet still selects the restricted model; multiplicity
alone does not request a broken-symmetry open-shell singlet.

Open-shell CBS supports conventional or explicitly fitted,
real-FP64 energy and analytic-force endpoints on CPU/CUDA. Explicit CBS
auxiliary bases require CUDA. The correction basis inherits native UCCSD(T)'s s/p/d,
numeric memory and scalar-work admission, without a fixed AO-count cutoff.
The finite work policy and explicit override are documented under
[correlated resource controls](correlated_methods.md#control-cost-memory-and-convergence).
The larger UMP2 bases retain their own native limits. Scalar ECPs follow the
Hamiltonian contract below; frozen core is not substituted or silently
downgraded. Unsupported native initial-guess
or screening controls also remain errors.

## Scalar ECP references

Orbital `BasisSet` snapshots may contain supported scalar semilocal Gaussian
ECPs on CPU/CUDA, for restricted singlets and unrestricted higher multiplicities.
The removed core sets valence occupations and ionic charges; nuclear repulsion
uses those same charges. The local/nonlocal residual enters the reference
Hamiltonian and every correlated one-electron response exactly once. First
forces include basis-center and ECP-center motion, orbital response and Pulay
terms; they are analytic, not production numerical differences.

Every atom must have identical core counts and scalar potential parameters in
the low, high and correction orbital bases. Mixing all-electron and ECP records,
different removed cores or different potentials raises `ValueError` before any
native endpoint starts. JSON formatting and unused element records do not change
this comparison. `basis_metadata["ecp_core_electrons"]` records the per-atom
removed core. All **remaining valence electrons** are correlated: the ECP core
is not an additional frozen-core approximation.

Explicit CUDA auxiliary bases retain the same correlation-only/full-DF policy;
they approximate the electron interaction, never the ECP operator. The checked
two-grid ECP convergence gate, angular limits and conservative numeric-workspace
reservation remain mandatory. An unsupported potential, unconverged reference,
response failure or resource refusal returns no partial composite. Standalone
HF/DF and other method-specific ECP admission is not enlarged by this workflow.
See [scalar ECP conventions and qualification](ecp.md).

The independent correlation qualification uses the physical LANL2DZ Na potential
with compact test-only Na/H valence bases, including nonzero unrestricted triples,
charged singlets and empty-beta triplets. These records qualify implementation,
not a systematic CBS family or arbitrary heavy-element accuracy. Supply a
consistent ECP/basis family and scientifically meaningful cardinals yourself.

Native unrestricted owners retain strict stopping ceilings even under
`policy="standard"` or `"fast"`; they do not admit the restricted relaxed
convergence transport. Explicit tighter tolerances are preserved.
`convergence` records `selected="strict_fp64"` and
`reason="unrestricted_strict_convergence"` for this path rather than claiming
relaxed native execution.

The model is `UHF(limit) + UMP2-correlation(limit) + [UCCSD(T)-UMP2](small)`,
with the same geometry-independent weights as the restricted workflow.
Each basis group shares one converged physical UHF reference and selected
interaction owner. Fitted correlation retains a conventional reference by
default; `hf_reference="df"` selects fitted UHF J/K as well. Integral, metric,
orbital and CC responses match that choice. See the explicit CUDA hybrid and
dense fitted-resource contract in [unrestricted CC](unrestricted_cc.md).
The correction-basis UMP2 phase finishes before the CC
workspace is allocated; it currently performs its own bounded transformation
rather than borrowing restricted CC MO blocks. All requested components must
succeed before any group or complete `CBSResult` is published.

<a id="fast-convergence"></a>

## Convergence profiles

`CCSDTCBSCalculator` defaults to `policy="standard"`: budget-scaled SCF,
CCSD, Lambda and orbital-response convergence. `policy="tight"` retains the
original strict CBS stopping limits. `policy="fp64"` uses those same strict
limits with all-FP64 preconvergence. Tight is a CBS-only preset; the HF/DFT
`PrecisionCalculator` profiles are unchanged.

`policy="fast"` remains a compatibility name for Standard's default allowance.
It is not an additional faster tier. Existing
`PrecisionPolicy("fast", budget=TargetErrorBudget(...))` calls retain their
custom allowance. Explicit target audits and other precision presets are not
implemented for CBS. To recover the old Standard convergence, select `tight`;
to recover its original FP64 schedule as well, select `fp64`.

The independent `preconvergence` option selects bounded auxiliary FP32 warmup
for CUDA DF-CCSD and Lambda. Final CCSD/Lambda solves and independent checks,
triples and analytic force response use FP64. The selected Hamiltonian, basis
groups and complete analytic-force work are preserved. Budget-scaled stopping
applies to conventional CPU/CUDA CBS and both CUDA DF reference models;
CPU and conventional CCSD keep FP64 arithmetic.
Unrestricted CBS also retains its strict FP64 owner, including explicit mixed
preconvergence requests: result metadata records the request separately from
`preconvergence_selection="fp64"`; it never invokes the restricted mixed owner.

| Request | Final convergence | Auxiliary preconvergence |
| --- | --- | --- |
| `policy="standard", preconvergence="mixed"` | Budget-scaled | Bounded mixed request |
| `policy="standard", preconvergence="fp64"` | Budget-scaled | FP64 |
| `policy="tight", preconvergence="mixed"` | Strict | Bounded mixed request |
| `policy="tight", preconvergence="fp64"` | Strict | FP64 |

`preconvergence="auto"` is the default. Admitted Standard/Fast CUDA DF calls
request mixed warmup. Tight fitted CUDA energy+force owners request it with at
least 9 occupied and 48 virtual orbitals; energy-only Tight calls use the
existing CCSD domain of at least 9 occupied and 128 virtual orbitals. Smaller
Tight endpoints retain FP64 because conversion and a fresh strict iteration
can offset the savings. An explicit mixed request bypasses these shape gates,
but never the resource/provider checks or final numerical requirements.

Older libraries without the independent execution capability retain their
legacy schedule for `auto` and report
`preconvergence_fallback="native_execution_transport_unavailable"`. Explicit
`mixed` or `fp64` selections on fitted Standard/Tight/Fast require that
capability. Mixed selection requires auxiliary bases. `policy="fp64"` rejects
an explicit mixed request. Libraries without correlated convergence transport
reject budget-scaled Standard/Fast before execution; they cannot silently
substitute a different stopping policy. DF budget-scaled calls also require
native precision telemetry.

```python
workflow = CCSDTCBSCalculator(
    (CBSBasis(load_basis("cc-pvdz.json"), 2),
     CBSBasis(load_basis("cc-pvtz.json"), 3)),
    auxiliary_bases=(load_basis("cc-pvdz-jkfit.json"),
                     load_basis("cc-pvtz-jkfit.json")),
    hf_reference="df",
    device="cuda",
    calculator_options={"correlation_memory_budget_bytes": 32 << 30},
)
result = workflow.singlepoint(atoms)
print(result.convergence)
```

For strict convergence, add `policy="tight"`. For a custom smaller allowance,
import `PrecisionPolicy` and `TargetErrorBudget` and pass
`policy=PrecisionPolicy("fast", budget=TargetErrorBudget(5e-6, 5e-5, 1.5e-5))`.

The default composite allowance is `1e-5 Eh` energy, `1e-4 Eh/Bohr` maximum
force error and `3e-5 Eh/Bohr` RMS force error. The workflow combines
coefficients of repeated method/basis components, then divides each allowance
by the sum of absolute combined coefficients. This accounts for extrapolation
amplification without assuming cancellation. For default DZ/TZ extrapolation
with a DZ correction that sum is approximately `4.196866`.

Python CBS planning queries `generativeqc_correlation_convergence_query_v1`,
which uses the same stopping-policy owner as the native correlated solvers.
Standalone `Calculator` validates CC ceilings using its selected library before
creating a context. Planning also requires an installed native runtime: it uses
the normal native loader and never computes stopping limits in Python. A library
without this query must be rebuilt or updated; a refused query or malformed
receipt is an error rather than permission to rediscover another library. This does not replace the
separate native convergence-transport capability required for actual execution.

The component allowance selects a scale `s`: the minimum of one and its
energy/max-force/RMS ratios to `1e-5`, `1e-4`, `3e-5`, respectively (omit the RMS
ratio when unspecified). The following ceilings apply:

| Control | Standard ceiling | Tight default |
| --- | ---: | ---: |
| Reference SCF energy change | `max(1e-11, s * 1e-8)` | `1e-12` |
| Reference SCF density change | `1e-11` | `1e-11` |
| CCSD energy change | `1e-8` | `1e-12` |
| CCSD residual maximum | `min(1e-7, max(1e-9, s * 1e-6))` | `1e-10` |
| Lambda residual | `max(1e-9, s * 1e-7)` | `1e-9` |
| Response absolute solve tolerance | `max(1e-12, s * 1e-8)` | `1e-12` |
| Independent CC orbital residual | `max(1e-10, s * 1e-7)` | `1e-10` |
| Complete CC orbital stationarity | `max(1e-8, s * 1e-6)` | `1e-8` |

Explicit tighter SCF/CC values in `calculator_options` take precedence. Below
`s=1e-3`, budget-scaled policies retain strict convergence and report
`component_budget_requires_strict`. Tight CBS energy/residual defaults
(`1e-12`/`1e-10`) are tighter than standalone
`Calculator(method="ccsd(t)")` defaults (`1e-11`/`1e-9`), which are unchanged.
All CBS profiles cap the SCF density tolerance at `1e-11` to preserve the
physical reference export contract, including at stretched geometries.
Use `result.convergence["limits"]` for effective controls. With default
DZ/TZ weights, Standard's `s` is about `0.238273`; the independent `1e-7`
CCSD residual cap is tighter than its budget-scaled value of about `2.38e-7`.
The same cap applies to Fast, including custom allowances. Stabilized energy
alone does not ensure a sufficiently small physical residual.

Stopping thresholds do not estimate the accuracy of CCSD(T) or of the CBS
approximation. The energy-change threshold controls CCSD iteration; the
conventional `(T)` correction is noniterative. Tightening it cannot remove
basis, fitting or correlation-model errors. Force calculations additionally
require converged Lambda and orbital response. The allowance-to-tolerance
mapping is empirical, not a per-call observable certificate or a mathematical
error bound. No duplicate strict endpoint is run to estimate the error.
Validate representative molecules against Tight or an independent reference
when choosing an allowance, especially for sensitive relative energies or
numerical derivatives. Static energy/force qualification does not establish
frequency accuracy, force continuity or long-time energy conservation.

RHF convergence does not guarantee the lowest or a stable electronic state.
Different initial guesses can reach different stationary branches, especially
at stretched bonds. Compare numerical tolerances on the same branch and check
reference stability separately; tighter CCSD convergence cannot resolve a
different HF state. The [retained qualification](../../benchmarks/results/cbs-routine-standard-20261025/README.md)
includes an independently matched stretched-N2 branch and its original
default-reference mismatch.

Conventional CUDA correlated references use scale-normalized DIIS, including
scalar-ECP references. This preserves useful small-residual history without
relaxing energy, density or physical-stationarity gates. Cold restricted
references also bound initial charge sloshing: during at most 32 iterations, an
undamped density proposal with RMS change above `0.1` is half-mixed with the
current density and its DIIS history is retired. Acceptance and reported density
changes always use the undamped proposal; warm references keep their existing
schedule. A rejected reference reports its SCF status, stopping reason, iteration
count, energy change, density change and last evaluated physical commutator RMS;
no correlated result is published. The final exported canonical reference is
checked independently.

Conventional CCSD(T) analytic forces right-precondition the physical orbital
response with the diagonal of the independently checked positive-definite
physical Hessian. Acceptance still requires the original unpreconditioned
residual and a separate dense-action residual check. A failed solve reports
its stopping reason, iterations, restarts, true residual and requested
tolerance; it does not publish partial forces or a partial CBS composite.

To isolate scheduling from stopping policy, compare the four combinations
above at the same geometry, basis, resource budget and endpoint. FP64 final
arithmetic checks the selected limits; it does not restore Tight convergence.
An allowance is a ceiling rather than a target error: an easy case can agree
much more closely. Correlated reference controls should not be replaced by
HF/DFT thresholds without validating the complete E+F endpoint.

Fresh physical residual replays gate publication using the selected limits.
Reference canonical-frame checks, denominator/gap safeguards, nonfinite
rejection, iteration limits and resource checks remain active. The SCF density
ceiling preserves the post-HF canonical-reference contract. Integral screening
is unchanged. DF-CCSD warmup switches to FP64 at a residual of at least 100 times
the selected CC residual ceiling, or after at most 24 warmup evaluations, and
clears mixed DIIS history. Resource/provider refusal retains the bounded FP64
path; numerical failures remain visible. Lambda uses at most 16 FP32 warmup
iterations with a `1e-6` handoff tolerance, then starts a fresh FP64 GMRES solve
with empty Krylov history. Its independent transpose audit and parameter/factor
derivatives remain FP64. The auxiliary batch retains the ordinary Q32 ceiling;
memory admission may reduce it.

`result.convergence` records the requested policy, selected route, composite
and component allowances, coefficient sum, effective limits and preconvergence.
Budget-scaled Standard/Fast reports `selected="relaxed"`; strict convergence
with mixed warmup reports `"strict"`, and strict FP64 execution reports
`"strict_fp64"`. These values describe execution, not an accuracy audit.
`preconvergence_selection` records the native request after shape admission,
before possible resource refusal. `arithmetic_request` records that request;
`arithmetic` reports `"mixed"` only when CCSD or Lambda FP32 GEMMs actually
executed, otherwise `"fp64"`. `lambda_arithmetic_request` and
`lambda_arithmetic` record Lambda separately; energy-only calls report
`"not_requested"`.

Component `cc_performance` records admission/refusal, evaluations or iterations,
GEMMs and casts. DF force endpoints also expose Lambda, source-response and
orbital phase times, actual auxiliary batch and numeric/device capacities.
Lambda action counts exclude its separate independent audit. Capacities are
not measured peak VRAM. Unavailable telemetry is `None`, distinct from measured
zero. Native diagnostics retain measured residuals and iteration counts.
Private correlated-budget transport rejects prepared batches and legacy RI-MP2.

## Correlation-only DF-CBS on CUDA

Supply `auxiliary_bases=(low_fit, high_fit)` to select **DF-MP2 correlation**
extrapolation and a **DF-CCSD(T)−DF-MP2** correction. The reference is still
conventional exact RHF in every orbital basis; this is not a DF-HF composite.
When the CUDA reference uses nonresident direct integrals, preliminary FP64
iterations use Schwarz and density-weighted J/K screening. The same plan then
clears its operator-dependent DIIS history and reconverges with unscreened FP64
integrals before publishing the reference. Final convergence and physical-frame
checks are unchanged; this does not introduce a screened reference into the
correlation energy or analytic forces. Resident-integral routes retain their
existing execution policy.

For a distinct correction orbital basis, also supply
`correction_auxiliary_basis`. For example, with checked local records:

```python
workflow = CCSDTCBSCalculator(
    (CBSBasis(load_basis("cc-pvdz.json"), 2),
     CBSBasis(load_basis("cc-pvtz.json"), 3)),
    auxiliary_bases=(load_basis("cc-pvdz-ri.json"),
                     load_basis("cc-pvtz-ri.json")),
    device="cuda",
    calculator_options={"correlation_memory_budget_bytes": 2 << 30},
)
result = workflow.singlepoint(atoms, properties=("energy", "forces"))
```

The lower basis is the default correction basis, so this example executes two
native groups, not six standalone calculations. Each group owns one converged
RHF reference. The correction group retains one original whitened DF source,
metric and MO frame through both correlation consumers. Its MP2 companion
does not repeat SCF, three-center integrals, metric factorization or MO source
construction, and does not solve CC or Lambda. HF forces use the same stationary
reference. Method-specific orbital response solves remain separate.

Equal orbital basis identities must share the same auxiliary identity. These
identities are snapshots, not mutable file names. Different orbital bases do
not share orbitals or integral factors. Native DF source, metric-rank,
derivative, convergence and budget checks still apply. Analytic forces include
auxiliary-center and metric derivatives as well as the exact RHF response.
This correlation source/nuclear sink admits orbital shells through f and
auxiliary shells through g, including spherical fitting bases used with TZ.
This scoped admission does not widen generic CUDA DF exports or orbital g.
There is no inherited conventional 12/56-AO force cutoff on this DF owner, but
this is not an unlimited-size admission promise: the source retains
`O(nAO² naux)` values and MP2's generated energy/response scratch is
`O(nocc² nvir²)`. A representative QM7 molecule needs actual basis/auxiliary
counts and endpoint qualification; atom count alone does not establish fit.

Standalone same-reference DF-MP2 is selected with
`Calculator(method="mp2", density_fitting="correlation_cuda", device="cuda",
auxiliary_basis=..., correlation_memory_budget_bytes=2 << 30)` and an explicit
`singlepoint(..., properties=("energy", "forces"))` request. This DF owner
requires a positive explicit correlation budget; standalone DF-MP2 defaults
to energy-only. The older `density_fitting="cuda"` RI-MP2 selector
keeps its DF-HF reference and energy-only CUDA contract; it is not silently
reinterpreted. Correlation-only DF-MP2 currently supports single-point/groups,
not prepared batches or CPU fallback.

## Full DF-HF/RI-JK CBS on CUDA

Add `hf_reference="df"` to the workflow with explicit `auxiliary_bases` to
use a stationary RI-JK reference throughout, not merely a DF initial guess.
The same RI operator supplies the orbital-response actions and final residual;
analytic forces include the three-center, auxiliary-metric, nuclear-center,
orbital and Pulay derivatives. No four-center RHF audit is substituted into
this different Hamiltonian. Allocation failures do not silently change it.

Each supplied auxiliary record currently serves **both JK and correlation**.
Choose and validate a fitting basis suitable for both purposes; using an
MP2-RI basis for HF is a separate fitting approximation, not automatically
equivalent to a JK-fit basis. Separate JK-fit and correlation-fit records
within one orbital-basis group are not yet supported. The old default
`hf_reference="exact"` preserves the correlation-only model and its results.
The returned `basis_metadata["hf_reference"]` distinguishes the models.

The native SCF RI-JK plan, its metric and factors stay alive through all
same-basis HF/MP2/CCSD(T) companion responses. Correlation retains its own
MO-source owner shared by the correlation consumers. Its construction does
not yet borrow SCF's AO factors; this is not a claim of a single factor build
across SCF and correlation. Different orbital-basis groups remain sequential.

Standalone full-DF analytic E+F uses
`Calculator(method="mp2", density_fitting="full_cuda", device="cuda",
auxiliary_basis=..., correlation_memory_budget_bytes=2 << 30)`;
`method="df-ccsd(t)"` accepts the same selector. This is distinct from legacy
`density_fitting="cuda"` RI-MP2. Full DF currently supports restricted and
unrestricted single points and same-basis groups, not prepared batches or CPU fallback.
The unrestricted UMP2/UCCSD owners also admit an explicitly requested native
CPU-reference context with these fitting selectors; results report
`cpu_reference`. This separate reference domain never substitutes CPU work for
a requested CUDA context. The restricted full-DF owners remain CUDA-only.
The restricted full-DF closure admits orbital shells through **g (`l=4`)** and
complete fitting bases through **h (`l=5`)**, for both energies and first
analytic nuclear derivatives. This includes unmodified heavy-atom
aug-cc-pVQZ/aug-cc-pVQZ-JKFIT records; shells are never truncated. Select
`hf_reference="df"` for CBS, or `density_fitting="full_cuda"` for standalone
MP2/DF-CCSD(T), with FP64 and an explicit complete correlation budget.
The bounded generated polynomial consumer owns the higher classes; optional
f/f/g Hermite storage and shell-specialized response schedules are not assumed
to cover g orbitals or h auxiliaries. Their resource/domain refusal retains
the complete generic source and weighted nuclear sink.

Native unrestricted CUDA DF Fock plans and resident single/bucket UHF execution
reject orbital g before one-electron or fitting-source preparation. This consumer
gate does not narrow the shared integral source used by qualified restricted full DF.

This does **not** admit conventional four-center orbital-g CUDA HF/correlation,
DFT orbital g, or higher derivatives. Heavy-atom aug-cc-pV5Z additionally needs
orbital h and auxiliary i (`l=6`), which remain explicit errors. QZ support is
not Q/5 qualification. Inspect `basis_capability_report` for the actual route
before planning a larger basis. The independent allocated regression gates in
`tests/python/test_full_df_cbs_native.py` cover complete augmented-QZ component
energies/forces and two-step reconverged finite differences, plus T/Q CBS
composition with separate DZ and TZ corrections. Retain the component values,
actual extrapolation weights, basis identities and root/resource diagnostics;
a CBS result is not a certified estimate of basis incompleteness.

Optional heavy-atom Q/5 anchors require a separate staged extension:

1. Extend the shared compiler algebra and native packing to orbital h/auxiliary
   i, including all S/T/V, three-center and metric values, spherical transforms
   and their first nuclear derivatives. Independently check the new raw blocks
   and raised-moment bounds before admitting a molecular route.
2. Qualify complete full-DF RHF/MP2 E/F with reconverged same-root references,
   two finite-difference steps, metric-rank diagnostics and bounded resource
   refusal, using the unmodified 5Z basis and fitting records.
3. Separately qualify CCSD(T), Lambda/orbital response and weighted force sinks,
   then Q/5 CBS composition with retained absolute components and actual weights.
   Record complete endpoint time, sampled device memory and semantic work;
   retain explicit h/i rejection until these gates pass.

Unrestricted fitted components admit orbital shells through f and auxiliary
shells through g for both values and analytic response, including unmodified
Na def2-universal-JKFIT and B cc-pVTZ-JKFIT auxiliaries. See the
[unrestricted fitting contract](unrestricted_cc.md) for diagnostic value-policy
restrictions and the retained bounded coordinate-derivative staging.

The positive `correlation_memory_budget_bytes` bounds complete numeric
capacity, including simultaneous host/device owners and a conservative retained
RI-JK allowance. It is not a measured GPU-memory peak. A budget refusal does
not authorize switching to exact RHF, omitting Lambda/response, or publishing
energy-only work as complete E+F. Choose an explicit allowance appropriate to
both the endpoint and available device memory.

(python-cbs-values)=
## Scientific model and values

For cardinals $X<Y$ and small basis $S$, the model is

```{math}
E = E_{\mathrm{HF}}^{\mathrm{limit}}
  + E_{\mathrm{corr,MP2}}^{\mathrm{limit}}
  + E_{\mathrm{CCSD(T)}}^S-E_{\mathrm{MP2}}^S.
```

MP2 **correlation** energy, not total MP2 energy, uses
$E_{\mathrm{corr}}(X)=E_{\mathrm{corr}}^{\mathrm{limit}}+A X^{-p}$.
The default is $p=3$. Its low/high weights are

```{math}
(w_X,w_Y)=\left(-\frac{X^p}{Y^p-X^p},\frac{Y^p}{Y^p-X^p}\right).
```

HF is handled separately. The default `hf_extrapolation="exponential"` uses
$E_{\mathrm{HF}}(X)=E_{\mathrm{HF}}^{\mathrm{limit}}+B e^{-\alpha X}$,
with fixed `hf_exponent=1.63`. Writing $r=e^{-\alpha(Y-X)}$, its weights are
$(-r/(1-r),1/(1-r))$. `hf_extrapolation="highest"` instead uses HF in basis
$Y$, with weights $(0,1)$; it is explicitly a highest-basis approximation,
not an independently extrapolated HF limit. `correlation_exponent` and
`hf_exponent` are caller-selected model inputs, not geometry-fitted parameters.
These are conventional extrapolation forms, not universal error guarantees;
see Helgaker et al., *J. Chem. Phys.* **106**, 9639 (1997), and the
[Psi4 CBS scheme definitions](https://psicode.org/psi4manual/master/cbs.html).

Every component uses the same geometry, charge, spin and all-electron or matching
scalar-ECP Hamiltonian, with the full remaining occupied space correlated. The default uses
exact correlation integrals; explicit auxiliaries select correlation DF, while
`hf_reference` selects exact or RI-JK reference response. Final physical checks
and response use FP64; the [preconvergence policy](#convergence-profiles) can
select bounded FP32 DF-CCSD/Lambda warmup independently of stopping limits.
Additional frozen-core variants are not admitted. Open-shell energies use the
unrestricted model and limits described above. Cardinals must be increasing integers, the larger
bases must have distinct identities, and all bases must have the same AO
representation. Cardinal numbers are explicit declarations; an arbitrary
record is not certified as a systematic CBS family merely by labeling it TZ.

Because the weights are geometry independent, the analytic force is

```{math}
\mathbf F=\mathbf F_{\mathrm{HF}}^{\mathrm{limit}}
  +\sum_{K=X,Y}w_K(\mathbf F_{\mathrm{MP2}}^K-\mathbf F_{\mathrm{HF}}^K)
  +\mathbf F_{\mathrm{CCSD(T)}}^S-\mathbf F_{\mathrm{MP2}}^S.
```

Do not use extrapolated energy with only the larger MP2 force. Do not refit
exponents at each geometry without their derivatives. Same-basis components
execute as one native group: MP2 supplies an HF companion, and CCSD(T) supplies
MP2 and, when needed, HF companions. Each group shares its converged canonical
RHF or UHF reference and orbitals. In the conventional model they also share the exact
AO interaction owner. In the restricted model, the correction-basis
MP2 energy and initial force adjoint also borrow the already transformed CC
OVOV block. Its remaining orbital-RHS, relaxed-weight and response-operator
reads borrow a complete six-block CC ERI symmetry cover when admitted, rather
than repeating AO-to-MO transformations. Equal method/basis identities still
contribute only once per call.

The other conventional CUDA MP2 E+F groups can build the same six-block symmetry cover once
from their own exact device-readable RHF source. This optional cache is admitted
only when it uses at most one quarter of the remaining correlation phase budget,
its construction fits, and the full force consumer fits alongside it. Otherwise
the existing bounded force path reconstructs tiles. Only memory failures trigger
that fallback; numerical errors propagate. Cache production runs on CUDA, but
its returned MO blocks and some response orchestration retain host storage.

Coordinates use Bohr, energy Hartree, and force Hartree/Bohr with shape
`(natoms, 3)` and sign `force = -gradient`. `CBSResult` exposes the three energy
and force contributions, fixed weights, actual completed native components,
basis identities/cardinals, native `execution_groups` and endpoint timing. Each component records its
combined coefficient and native diagnostics; the weighted total components
equal the composite. All force fields are `None` for energy-only calls.
Component timings identify their containing group and are **not additive**;
use `execution_groups` for non-overlapping group times. Companion convergence
fields describe the completed group, not an additional standalone method solve.
Convergence of every component is necessary but does not certify CBS model
accuracy. No basis-error certificate is manufactured from SCF convergence.

(python-cbs-errors)=
## Errors and publication

Malformed cardinals, representations, extrapolation parameters, occupations,
properties and unsupported calculator options fail with validation errors.
Unsupported Hamiltonians/spin domains fail explicitly. Local basis reads retain
their filesystem/checksum errors. Native method, basis, derivative, memory,
stationarity, response and convergence failures propagate without a partial
`CBSResult`; a successful subset is not a successful composite. Nonfinite
energies/forces or a component executing on the wrong backend also reject.
Previously returned results remain independent of a later failed call.

`calculator_options` supplies common native SCF/CC convergence, screening,
initial-guess and workspace/resource controls. Defaults tighten SCF energy and
density tolerances to `1e-12`/`1e-10`, CC energy/residual tolerances to
`1e-12`/`1e-10`, with 200 SCF and 150 CC iterations. It cannot override component
methods/bases, precision, frozen core or density fitting. Native admissibility
checks still apply to these controls. `Calculator.singlepoint_components` also
exposes same-reference groups directly for conventional, full and correlation-only
DF MP2/CCSD(T), and native CPU/CUDA UMP2/UCCSD(T) energy/force endpoints.
An unrestricted owner defaults to a `uhf` companion; UCCSD(T) also accepts
`companions=("uhf", "ump2")`. Restricted companion names are not aliases for
unrestricted methods.

(python-cbs-ownership)=
## Ownership and concurrency

The installed native runtime owns extrapolation weights and the six focal-point
coefficients through
{ref}`generativeqc_cbs_extrapolation_query_v1 <native-c-generativeqc-cbs-extrapolation-query-v1>`.
Construction queries this geometry-independent plan and the convergence policy
from the same normally selected host library, including for CUDA workflows.
Missing support, a native refusal or a malformed receipt rejects construction
before electronic execution; Python does not reconstruct a fallback plan.
This host metadata query neither executes nor admits a molecular backend.

Python still owns basis-family/identity validation, method/basis grouping,
coefficient-L1 budget allocation, sequential component execution and observable
composition/publication. The native coefficient plan is not a complete native
CBS endpoint or CLI method, a basis-error estimate, or a performance certificate.

`CBSBasis` snapshots local records at construction. Modifying a file or the
caller-owned options dictionary does not change an existing workflow. Keep the
workflow's configuration fixed during use. One workflow serializes its calls;
use separate instances for concurrent execution.

Native basis groups execute sequentially and release their retained single-point
storage in `finally`, including failed calls. Only one native endpoint owner is
retained at a time. The Python workflow retains owned basis data and completed
result/force diagnostics, not the several bases' CUDA executables. Published
arrays outlive cleanup and are separate from the composite force storage.
Frozen dataclasses do not recursively freeze their NumPy arrays or mappings.

Within a group, CC-only blocks/amplitudes are retired after its complete E+F
replay. Only an admitted six-block MO cache, reference/source and published primary outputs stay
live beside the companions. Every companion admission charges these live
reservations. A memory-only fallback can release optional MO/source/plan
residency and reconstruct bounded tiles; numerical failures are never retried
through a different scientific model. Different-basis orbitals/integrals are
not interchangeable. MP2 and CC retain separate method-specific adjoint/Lambda
and orbital-response solves; this is not a shared multi-RHS response solver.

Native budgets apply independently to each **group**, not to total process RSS,
Python storage or a scheduler-wide device reservation. Planned/measured peaks
are available in each native result and must not be added or confused with a
measured whole-workflow allocation peak. This policy trades retained prepared-
plan warm reuse for bounded sequential ownership; it does not claim a fully
resident multi-basis GPU workflow.

(python-cbs-backends)=
## Backends and qualification

Select `device="cpu"` or `device="cuda"` explicitly; there is no CPU fallback
when CUDA is requested. `executed_backend` retains the native backend spelling
(`cpu_reference` or `cuda`). Chemistry, force-domain and resource support are
inherited from each actual native component, not broadened by composition.
In particular, legacy DF-HF CUDA RI-MP2 forces remain unsupported, and the small
CCSD(T) force calculation must fit its actual native domain. The explicit
correlation-only DF owner is separate from conventional force admission. This orchestration
does not remove remaining host staging or improve the asymptotic MP2 response
algorithm. Optional MO caching reduces repeated transformations, not the
method-specific response work or its domain limits.

`tests/python/test_cbs_open_shell_native.py` independently reconstructs the
input basis shells in PySCF and qualifies every unrestricted component and
the full composite for neutral doublets and charged, empty-beta triplets.
Its compact synthetic basis pair qualifies composition, not CBS predictive
accuracy. Energy acceptance is `1e-8` Hartree for Tight and `1e-7` Hartree for
Standard. Separate tests retain atomic all-or-nothing publication on CC or
UMP2-companion failure and reject spin-incompatible companion names.

Independent reference data live under `tests/reference_data/cbs/`. H2 uses the
physical spherical cc-pVDZ/cc-pVTZ sequence. A separate water **synthetic pair**
tests the composition with nonzero triples; it is not evidence of predictive
CBS accuracy for water. Both reference composites are checked against
independently reconverged displaced energies. The public native acceptance
gates are `1e-9` Hartree for energy, `2e-6` Hartree/Bohr for force and composite
directional derivatives, and every constituent endpoint is checked separately.

The restricted conventional iodine-ECP regression in
`tests/python/test_conventional_ecp_cbs_regression.py` uses checksum-pinned
CH3IN2, H2IO(-) and CH2ClI geometries, spherical cc-pV(D,T)Z-PP on iodine,
cc-pV(D,T)Z on the other atoms, and the matching 28-electron scalar ECP.
It retains the original 43-GiB numeric budget and native stopping limits.
Physical HF/orbital-response regressions and complete CBS qualification are
separate tests: passing an originally failing step does not qualify the whole
energy-and-force endpoint. Complete acceptance checks all five component
energies and forces, the composite, zero net force, and independent PySCF
directional energy differences at both `2e-4` and `6e-5` Bohr. Its absolute gates
are `3e-8` Hartree per component, `9e-8` Hartree for CBS energy, `3e-7`
Hartree/Bohr for directional force and `3e-8` Hartree/Bohr for net force.
These cases qualify that implementation/input domain, not arbitrary elements,
bases, geometries or predictive CBS accuracy.

Run each long case in its own finite GPU allocation; the opt-in test also
requires `SLURM_JOB_ID`. For example, on a cluster exposing `gpu:pro6000`:

```bash
srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --time=04:00:00 env GENERATIVEQC_ECP_CBS_3034_CUDA_TEST=1 \
  python -m pytest -q \
  'tests/python/test_conventional_ecp_cbs_regression.py::test_reported_complete_ecp_cbs_force[hf_tz]'
```

Repeat with `hf_dz` and `response_dz`; preserve the Slurm-assigned CUDA device
visibility and use the same source-matched CUDA library for all three cases.

Run CPU contracts with `pytest tests/python/test_ccsdt_cbs*.py`. CUDA qualification
requires `GENERATIVEQC_CBS_CUDA_TEST=1` and a real Slurm allocation. The
physical H2 DZ/TZ **CPU force** replay is opt-in through
`GENERATIVEQC_CBS_LARGE_TEST=1`; routine CPU CI checks its energy and the water
algebra E+F case, while allocated CUDA qualification checks both full E+F cases.
The endpoint
benchmark `benchmarks/ccsdt_cbs_endpoint.py` times the **whole** workflow,
including component construction, SCF, correlation, analytic forces and owner
cleanup, and optionally records an untimed semantic-work replay. Timed replay
samples are not advertised as retained prepared-batch warm calls.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 python benchmarks/ccsdt_cbs_endpoint.py \
  --device cuda --case h2-dz-tz --oracle --work --repeats 1 \
  --output .artifacts/ccsdt-cbs-h2.json
```

### Explicitly correlated component assembly

`generativeqc.experimental.f12_focal.combine` accepts independently evaluated
energy and Cartesian force contributions from a single geometry. Pass a
`FocalSeries` for HF plus CABS singles, a second series for ordinary MP2
correlation plus the explicit F12 pair correction, and a high-level correction.
For an MP2-F12 energy and force, explicitly pass `post_mp2=None` to `combine`,
`plan_singlepoint`, or `singlepoint`. This evaluates just the first two series.
To add the explicitly correlated doubles increment, pass `post_mp2=None` and
`f12_doubles=FocalSeries(upper, lower, power)` with the desired bases or terms.
The `ccsd_f12b_minus_mp2_f12` step takes the difference of CCSD-F12b and MP2-F12
correlation energies and forces at each cardinal, using the same determinant,
orbital basis, three fitting roles, CABS and geminal exponent. The molecular
provider solves the spin-orbital equations on an all-active UHF or singlet RHF
determinant. Both spin lines use the same spatial orbitals on RHF, and their
derivatives add to the single RHF orbital response. A separate `triples` term
can add the conventional perturbative triples contribution.
Use the role `post_mp2` for a combined correction. To treat conventional
(T) separately, use `ccsd_minus_mp2` for the matched CCSD minus MP2 increment
and pass the `triples` term or series by keyword. Each correction accepts
a single term or its own cardinal pair. Difference CCSD and MP2 using the same
orbital basis, fitted auxiliary policy and correlated space at each cardinal.

A triples mapping selects independent recipes in one call. The keys
`conventional`, `tplus` and `f12` refer to the conventional correction, the
Kállay occupied-orbital scaling and the explicitly correlated triples equation.
For example, `triples={"tplus": basis, "f12": basis}` returns
`FocalTriplesAlternatives` with one result per method under `.variants`.
Each energy and force includes its own triples correction exactly once. A
molecular model owner receives all requested variants on a shared determinant
and matching basis together. Pass the same alternatives to `singlepoint_energy`
with their analytic center for signed energy checks.

When an alternatives request includes explicit triples, the built-in molecular
provider checks the Köhn implementation's electronic-root and derivative scope
before starting the molecular calculation. The RHF implementation supports
energy-only evaluation. Requesting full analytic forces for that alternative raises
`NotImplementedError`. Automatic `tplus` selection for an unpinned triples
series requires provider qualification for both energy and full analytic force
at every requested cardinal. An existing `protocol_id` and explicit method
mappings keep the selected recipe fixed.

A single upper term selects a finite endpoint. Giving a lower term and a
positive exponent selects an inverse-power limit. For an exponential HF plus
CABS limit use `FocalSeries(upper, lower, alpha, extrapolation="exponential")`.
Each series can have its own family and cardinal pair, including parent,
heavy-augmented and fully augmented basis sets. Use the same
surface identity for all terms, including ordered coordinates, electronic root,
ECP and frozen-core convention, and distinct basis identities for different
cardinals. Returned weights apply to both energy (Eh) and force (Eh/bohr).

Use `generativeqc.experimental.f12_focal.singlepoint` to dispatch components
from one ordered geometry in Bohr. Supply `FocalBasis` records (or series of
records) and an explicit component provider. Each basis snapshots its orbital,
JKFit, and applicable MP2Fit, F12 fitting and CABS data from canonical local
basis records. Give the F12 exponent in inverse Bohr for pair correlation.
The caller also supplies charge, multiplicity, root and correlated-space
identities. All orbital records must describe the same scalar Hamiltonian.

In frozen-core definition version 2, both `"frozen-core"` and the production
`"frozen-core-spd"` space freeze only Na 1s. Na 2s and 2p participate in
MP2-F12, CCSD-F12b, (T), (T+), ordinary CCSD(T) and ΔSR. The `"cv-active"`
space also freezes Na 1s, so ΔCV releases core shells only on other atoms.
Both valence spaces subtract the pairs represented by a scalar ECP.
`"frozen-core-spd"` freezes the complete (n−1)d core shell on post-d p-block
atoms such as Br and I.

The provider selects molecular occupied orbitals by projecting them onto
spherical atomic HF core orbitals in the requested orbital basis. Each frozen
orbital must have majority core character and each active occupied orbital
must have less than half. Every frozen–active occupied pair in each spin
channel must have an absolute energy separation of at least 0.05 Eh.
For example, Na 2s can lie below frozen I 4d in NaI and remain active.
An ambiguous assignment, incomplete core span, or smaller gap raises
`ValueError`. The explicit `"frozen:<n>"` and `"frozen:<na>,<nb>"` options
select the lowest-energy occupied orbitals and are intended for diagnostic
comparisons.

Production assembly reads the correlation space and frozen-core definition
version from the composed valence result and verifies matching ΔCV and ΔSR
versions. Archive the version with every label. For explicit ΔCV or ΔSR
energy or force values, also provide the corresponding
`delta_cv_frozen_core_definition_version` or
`delta_sr_frozen_core_definition_version`.

Conventional triples calls on ordinary basis sets use the selected core on the
retained CCSD reference. Pass a `telemetry` mapping to archive the core space,
version and selected original occupied indices. For custom surface IDs, add
the public `FROZEN_CORE_DEFINITION_VERSION` to the hash inputs.
The grouped `conventional_df_post_mp2` endpoint requires `"all-active"` and a
common JKFit and MP2Fit.

Use Na `cc-pCVnZ-F12` with its OptRI and RIFIT at the same cardinal for the
F12 steps, with the recipe's existing diffuse additions and JKFit. Use
`aug-cc-pCV{D,T}Z` and an appropriate declared MP2Fit for the ordinary
heavy-augmented steps. Na uses `aug-cc-pCVTZ` for ΔCV. Supply the records
through the element mappings described below.

The table pins the Na entries from Basis Set Exchange 0.12, basis data
version 0. To check each digest, request the element 11 JSON record and
hash its `elements` object with sorted keys and compact JSON separators.
[Hill and Peterson orbital and OptRI data](https://pubs.rsc.org/en/content/articlelanding/2010/cp/c0cp00020e)
and [Kritikou and Hill density-fitting data](https://doi.org/10.1021/acs.jctc.5b00816)
are distributed through [Basis Set Exchange](https://www.basissetexchange.org/).

| Basis Set Exchange name | Na element SHA-256 |
| --- | --- |
| `cc-pCVDZ-F12` | `913a951a36e6f530950b60e81db035569c1e45a9bfa7a2a44ca4582b7f907652` |
| `cc-pCVDZ-F12-OPTRI` | `953d823f0a9b1ce1d8e1fa29ca36690b00b4396823607443dfb117ae29ec4110` |
| `cc-pCVDZ-F12-RIFIT` | `31e39aa3b8f09980514acebfeaceb63e6a7e7d9d9ed1f110b3cbfc3ffb141aa8` |
| `cc-pCVTZ-F12` | `a42d5506fcd75c34f845060234902c42850ac50f2c8b94dc338dc35530ab9bea` |
| `cc-pCVTZ-F12-OPTRI` | `1be54fb7044e555c26b7da933ed420b467d5dcd00452b2ee7b1e78e5cbc90ee7` |
| `cc-pCVTZ-F12-RIFIT` | `0b5983717dbc9bf0dcf819e4252c17f968a7df13b1e986248df3831154b46fae` |
| `cc-pCVQZ-F12` | `1eafa8abc4ddc15eb9fcb5e8c4fc5b90155092922d4822712f52b2a135761e95` |
| `cc-pCVQZ-F12-OPTRI` | `e8a0d628a862c338fbfd93cc34fa7bbf370e8eb5e8aec878bd0f0934e19126a8` |
| `cc-pCVQZ-F12-RIFIT` | `3dbb70e7bee59a2bc957e5d1f5979ace0fe10fdb29d25e9998c1568e5443ba6d` |
| `aug-cc-pCVDZ` | `022733068f4d127b66b2c9a23a3cc972a5e124eba0ffb58772737e0e375a2aec` |
| `aug-cc-pCVTZ` | `c390e87191c8a5c390bb9917d5dc016ed7d7b697da2a272cb80264c1daeeae9f` |

The CUDA molecular provider factors a full-rank F12Fit Coulomb metric
`M = L Lᵀ` and uses `L⁻¹` to whiten correlation-fit panels. The JKFit
role uses the symmetric inverse square root. When those roles have the same
basis the provider can reuse the raw integrals while retaining the factor
appropriate to each contraction. The CUDA F12Fit factor requires a resolved
gap to the requested rank cutoff. Ordinary MP2 energies, occupied-orbital
contributions and the corresponding analytic forces use the canonical
reference orbital energies for their denominators. The explicit F12
increment uses the extended Fock matrices.

Each of the five basis roles also accepts an element mapping. For example,
`orbital={"H": "sto-3g", "He": "def2-svp"}` selects only the H record from
STO-3G and the He record from def2-SVP. Use an element symbol or its atomic
number as the key. Each value can be a bundled name, a `BasisSet`, or a local
canonical basis file. Provide every element in the geometry for each required
role. A role has one AO representation across all its elements. Selections
preserve the source shells, scalar ECPs and provenance, and use the same
resolved basis identity for equivalent symbol and atomic-number mappings.
A cardinal pair must change the orbital basis on at least one element in the
geometry.

For each requested role and cardinal, the provider receives the resolved basis
and geometry and returns a converged `FocalEvaluation`. The result must echo
the request's surface and basis identity and report its executed backend.
The requested backend is CUDA by default. Set `backend="cpu_reference"` to
request that backend explicitly. A backend mismatch raises before numerical
assembly. The completed components are returned alongside the combined energy,
forces and weights.

For a CUDA molecular calculation, `executed_backend` identifies the native
backend. Individual scientific stages can also use host arithmetic. For
example, occupied-orbital (T+) scaling evaluates its energy and cotangents in
FP64 on the host. Wrap the calculation in
`generativeqc._f12_ram_hygiene.capture_hybrid_phases()` to collect the
disclosed stages. Set `GENERATIVEQC_F12_STRICT_GPU_ONLY=1` to reject a
disclosed host contraction before it executes. Report CUDA JIT compilation
counts separately from where scientific arithmetic ran.

Spatial RHF CCSD-F12b reports `f12_cuda_preparations` and
`f12_cuda_rhs_evaluations` for the active fixed-cusp stages. A forward-only
solve has zero Lambda RHS evaluations. Historical libraries lack these
counters and use host arithmetic for constant preparation and the active
Lambda RHS. Strict GPU-only mode requires a capable native library before
starting an active spatial solve and checks the executed pass counts.

The built-in provider uses conventional triples for a bare triples basis or series.
Select occupied (T+) explicitly with `triples={"tplus": card}`. A provider may
supply a `qualified_tplus` predicate for an independently qualified recipe.
An unpinned recipe with F12 doubles can then select (T+) for that recipe.
A protocol ID keeps a bare triples selection conventional.

On the spatial RHF CUDA path, `MolecularFocalProvider.cuda` accepts two
optional precision settings for conventional and occupied-scaled triples.
`triples_fp32_w=True` selects single-pass FP32 W contractions.
`triples_w_schedule="fp32_split3"` selects three limbs per operand and
recombines nine FP32 products in FP64. Select one setting at a time. Compare
the returned `forward_w_precision`, `reverse_w_precision` and
`w_precision_schedule` with the requested schedule to identify the work that
ran.

Without either setting, triples use FP64 W and the provider's
`preconvergence="auto"`. The library does not select a W precision from the
GPU model.

For finite-difference checks of `fp32_split3`, use identical settings at the
analytic center and all displaced energies. Check single-pass W against the
FP64 reference instead: analytic forces within 1e-8 Eh/bohr of FP64 central
differences and energies within 1e-9 Eh of FP64. Central differences of its
own energies amplify FP32 rounding noise by 1/(2h), so they serve only as a
diagnostic.

Analytic triples forces accept `triples_fp32_w=True` only on an NVIDIA L20
(`sm_89`). On any other GPU or the host backend, the request raises
`generativeqc.experimental.f12_focal.TriplesWPrecisionError`, a `ValueError`,
before native triples work. Use the default FP64
schedule or `fp32_split3` there. The CUDA device is ordinal 0 after
`CUDA_VISIBLE_DEVICES` filtering, identified by driver product name and
compute capability. The spatial RHF single-pass schedule uses FP32 W in
both directions. The spin-blocked CUDA UHF schedule uses FP32 forward W
and an FP64 pullback. If the native triples report a different executed
schedule or a resource fallback for production single-pass W, the call raises
`TriplesWPrecisionError` and returns no forces.

Force telemetry records `w_precision_schedule`, `preconvergence`,
`device_name` and `device_architecture`. The host backend records `cpu` and
`n/a`. The conventional `triples_energy_and_forces` entry requires a
`telemetry` mapping for single-pass W.

For a non-production comparison on any device, pass
`diagnostic_fp32_single_w_forces=True` to `triples_energy_and_forces` or to a
(T+) `..._energy_and_forces_on_reference` call. Its telemetry then reports
`production_eligible=False`. Production focal validation and assembly reject
such components. They require a recorded NVIDIA L20 (`sm_89`) device and
reject resource fallbacks for single-pass W force components. The energy-only
interfaces honour the caller's explicit W setting regardless of CUDA device.

`MolecularFocalProvider.assemble_production_energy_and_forces` takes the
composed valence result as `valence`. Pass a `FocalRunResult` or
`FocalMolecularResult` for energy and forces, or a `FocalEnergyResult` for the
energy only. For independent component jobs, pass the result of `plan.collect`
described below. Assembly recomposes the valence energy and forces from the
result's components and weights. It raises `ValueError` when the result has no
components or its stored totals differ from that sum, and `TypeError` for
untyped results or components. An energy-only result cannot be combined with
Delta-CV or Delta-SR forces.

For closed-shell RHF singlets, `singlepoint_energy(..., triples={"f12": card},
f12_doubles=card)` evaluates the full Köhn extended-SP energy with the
FfgCompleted Z2 intermediate and a stationary CCSD-F12b parent. The CPU
reference and CUDA kernels share the 33-term equation. CUDA uses FP64 host
AO-to-MO/metric projections. The molecular component carries the model ID
`kohn_extended_sp_rhf_ffg_v1`, native backend, CABS rank in the result,
and precision telemetry. This endpoint excludes unrestricted roots, Br/I
semicore correlation and Cartesian forces. It is experimental. Its independent
controls include a GeCCo BH term breakdown and the PySCF zero-coupling limit.

For the built-in molecular recipe, use `MolecularFocalProvider.cuda(...)` and
pass `post_mp2=None`, `f12_doubles` and `triples` to `singlepoint`. For example,
choose separate fully augmented D/T and heavy-augmented D/T bases with their
matching fits and CABS.

```python
from generativeqc.experimental.f12_focal import FocalSeries, singlepoint, singlepoint_energy

recipe = dict(
    hf_cabs=FocalSeries(a_t, a_d, hf_alpha, extrapolation="exponential"),
    correlation=FocalSeries(a_t, a_d, mp2_power),
    post_mp2=None,
    f12_doubles=FocalSeries(ha_t, ha_d, cc_power),
    triples=FocalSeries(ha_t, ha_d, triples_power),
    provider=provider,
    root_id="rhf",
    correlation_space_id="all-active",
    protocol_id=recipe_id,
)
center = singlepoint(atoms_bohr, **recipe)
displacement = singlepoint_energy(displaced_atoms_bohr, center=center, **recipe)
```

The center shares one determinant per distinct complete basis identity and
returns a Cartesian force array with shape `(N, 3)`. The displacement evaluates energy
only and checks the weighted recipe, ordered nuclei, occupied reference and
CABS rank against the center. Use `root_id="uhf"` with the corresponding
`multiplicity` for an unrestricted determinant. The same series interface
accepts explicit T/Q or Q/5 basis records. Each electronic solve still has
its native basis and memory limits. Supply the orbital, JKFit, MP2Fit,
F12Fit and CABS for the elements and cardinals used in your recipe.

For independent component jobs, call `plan_singlepoint` with the same geometry
and basis arguments, execute the requests in `plan.steps`, then pass the
outputs in any completion order to `plan.collect`. It matches each role and
cardinal against the planned surface and basis. When archiving components for
later collection, supply a fixed `protocol_id` that identifies the method
and implementation version across all jobs.

The CUDA wheel installs the molecular shell and HF+CABS response libraries
alongside the native runtime under `generativeqc/lib`. For an SDK build,
choose the CUDA architectures explicitly. For example, this builds both
libraries and installs their F12 component into a local prefix.

```sh
cmake -S . -B build/f12 -DGENERATIVEQC_ENABLE_CUDA=ON \
  -DGENERATIVEQC_CUDA_ARCHITECTURES='80-real;89-real' -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_LIBDIR=lib -DGENERATIVEQC_BUILD_CLI=OFF
cmake --build build/f12 --target focal_shell_i focal_hf_cabs_response
cmake --install build/f12 --component f12 --prefix "$PWD/build/f12-sdk"
```

The F12 component contains `libfocal_shell_i.so` and
`libfocal_hf_cabs_response.so` and their public symbol lists under
`share/generativeqc/f12`. A CPU-reference configuration
(`-DGENERATIVEQC_ENABLE_CUDA=OFF`) builds the shell target as
`libfocal_shell_i_host.so`. The HF+CABS response target requires CUDA.

For a source-checkout qualification build, run
`python src/posthf/f12/build_f12_molecular_integral_sink.py --backend host --cabs-i --output-dir build/shells`
(or use `--backend cuda --arch sm_89`). Read the `library` path and `sha256`
from the JSON receipt on stdout. The published path identifies the backend,
CUDA architecture where applicable, shell domain and compiled bytes.

In a source checkout the molecular integral benchmark accepts a planned
`FocalRequest` directly in `forward_integrals` and `contract_live_weights`.
The HF+CABS role evaluates the two Coulomb channels in JKFit. The MP2-F12
role evaluates the Coulomb and four geminal operators over two and three
centers in F12 fit. The first combined AO rows belong to the orbital basis,
followed by the atom-centered OptRI rows, all in native real-spherical order.
The reverse call returns the Cartesian gradient of the supplied integral
weights. The SCF, CABS projector and correlated-state response supply the
remaining contributions to a molecular force.

Use `generativeqc.experimental.f12_focal.one_electron_sources` with the same
request to obtain the all-electron overlap and core Hamiltonian. The request
selects the installed shell library for its backend. Pass `library=path` to
use a specific SDK build. Both matrix axes use
the same OBS-first order as the two- and three-center integrals.

```python
from generativeqc.experimental.f12_focal import one_electron_sources

request = next(r for r, _ in plan.steps if r.role == "hf_cabs")
data = one_electron_sources(request)
```

`data.d_overlap[atom, axis]` and `data.d_hcore[atom, axis]` are positive
Cartesian derivatives in bohr⁻¹ and Eh/bohr. They include movement of each
atom's OBS and OptRI functions and the attraction to the moving nucleus.
Set `derivatives=False` for energies that need only S and Hcore.

Use `generativeqc.experimental.f12_focal.df_reference(request)` to solve the
cold all-electron `rhf` or `uhf` root in the request's real-spherical orbital
basis. Both Coulomb and exchange use its JKFit record and a relative metric
cutoff of 1e-10. The result contains the DF-HF energy in Eh, its analytic
Cartesian forces in Eh/bohr, the OBS overlap and core Hamiltonian, and the
canonical coefficients, orbital energies, physical Fock and density of each
spin channel. RHF returns one channel with a doubly occupied density. UHF
returns alpha then beta. `forces=False` requests an energy and orbitals at a
displaced geometry. The CUDA provider evaluates the DF operators with native
host SCF control. CABS singles and their nuclear response are separate terms.

After a failed UHF primary solve, the provider tries the level-shifted
first-order recovery before refining the primary determinant with second-order
steps. The solver examines real internal orbital curvature at the converged
UHF determinant. It spans the full occupied–virtual rotation space when it
fits the 60-vector subspace bound.
For larger spaces the bounded Davidson search samples directions, and a
converged Ritz value describes the explored modes. Use the `seed` argument
to follow a particular determinant across nearby geometries.

The provider controls electronic execution and is responsible for the
same-reference CCSD and MP2 differences, convergence and analytic gradients.
The experimental dispatcher requires a provider for electronic work.
Availability of electronic methods, elements and their molecular responses is
established through independent endpoint tests of the provider.

The explicit `conventional_df_post_mp2` provider supplies the combined
DF-CCSD(T) minus DF-MP2 energy and analytic force from one native group. Route
`post_mp2` to this provider and supply your F12 provider for the other roles.
Use root identity `rhf` or `uhf`, correlation space `all-active` and an identical
JKFit and MP2Fit record for the conventional increment. The restricted
DF-CCSD(T) provider requires CUDA. The caller must match the selected cold
reference with the rest of the focal calculation.

For a finite-difference control, plan each displaced geometry and pass its
`post_mp2` request to `conventional_df_post_mp2_energy`. This executes the
same native group with energy as the sole requested property. Keep the
request identity alongside the scalar result. At the central geometry use
`conventional_df_post_mp2` to obtain the analytic forces.

For the complementary orbital space, pass the returned one-electron result
to `focal_cabs_projector(request, one)`. Its `frame` columns are orthogonal
to the full orbital basis in the combined overlap metric. Record its `rank`
and pass `expected_rank` when evaluating displacements. A matrix supplied as
`cotangent` weights the invariant kernel `frame @ frame.T`. For a fixed
weight, `einsum("ij,axij->ax", result.overlap_bar, one.d_overlap)` gives
the positive nuclear derivative of the weighted kernel.

Full-rank CABS overlap responses and the CABS frame checks evaluate their host
products in double-double precision with Ozaki splitting. Inputs outside the
Ozaki range, from overflow-adjacent magnitudes or an unresolvable dynamic range,
are accumulated exactly up to 2^18 multiply-adds per product. A larger product
raises `CompensatedProductWorkLimit` with its shape, trigger and multiply-add
count instead of returning a lower-precision result.


For fixed-cusp unrestricted CC equations, call
`provider.ucc_fixed_sources(pair_request)` on the all-active UHF pair step.
The returned `FocalUCCFixedSources` contains the canonical DF reference,
retained CABS rank, F12Fit identity and the `feeds` dictionary. Its six spin
orders each have `w_vo_*`, `w_vv_*` and `c_f12_*` tensors in the reference's
OBS frame. `fixed_geminal_energy` contains V+B−X in Eh. The extended Fock
and positive exchange use JKFit, and the robust geminal projections use
F12Fit. The CC amplitude and Lambda solvers consume these equation inputs.
Call `provider.ucc_equation_sources(pair_request)` to generate both
`result.fixed` and the ordinary Hamiltonian dictionary `result.ordinary`
from one DF-UHF solve. JKFit defines the Fock matrices, MP2Fit the
antisymmetrized ordinary two-electron blocks, and F12Fit the geminal
projections. `result.mp2_fit_id` identifies the ordinary correlation fit.
The coordinate geometry determines every input to these two dictionaries.

Call `solution = provider.ucc_stationary(pair_request)` for a cold
all-active UHF fixed-cusp UCCSD-F12b solution at that geometry.
`solution.sources.fixed.reference` holds the DF-UHF determinant,
`solution.primal` holds the correlation energy (Eh), physical amplitudes
and convergence history, and `solution.dual` holds the qualified Lambda
multipliers and stationarity residuals. The primal total adds the DF-UHF
and stationary correlation energies. CABS singles is a separate contribution.
A provider constructed with `MolecularFocalProvider.cuda(...)` uses its
GPU architecture, cache and memory budget for the CC residual and adjoint
equations. On CPU, the stationary equations use the native FP64 TensorIR
executor. Optional `options` and `lambda_options` accept `SolverOptions`
and `UCCSDLambdaOptions` from `generativeqc.experimental.uccsd_f12`.
With `source_cotangents=True`, the solve also returns
`solution.source_cotangents.ordinary_weights` and `fixed_source_weights`.
These are the derivatives of the stationary CC Lagrangian with respect
to the active canonical MO input tensors in `solution.sources`.
`provider.ucc_ordinary_source_forces(pair_request, solution)` contracts the
38 ordinary Hamiltonian weights with their nuclear derivatives in Eh/bohr.
It uses the original stationary spin orbitals, JKFit for the Fock response,
and MP2Fit for the 32 antisymmetrized Coulomb blocks. The returned array is
the ordinary-input contribution to the stationary correlation force.

Call `result = provider.uccsd_f12b(pair_request)` to obtain the complete
single-basis all-active UHF fixed-cusp UCCSD-F12b molecular energy in Eh and
Cartesian forces in Eh/bohr. This calculation uses one determinant for the
DF-UHF reference, CABS singles, and both CC Hamiltonian source families.
`result.hf_cabs_energy` and `result.stationary.primal.correlation_energy`
sum to `result.energy`. The corresponding force sum is
`result.hf_cabs_forces + result.ordinary_cc_forces + result.fixed_cc_forces`.
The stationary record retains the surface and basis identities, spin
occupations, orbital coefficients, separate fitting identities, amplitudes
and multipliers. The CBS focal plan selects its post-MP2 and triples
providers independently.

Use the installed `MolecularFocalProvider` to evaluate a plan step directly
from its geometry. A single-card CUDA setup looks like this:

```python
from generativeqc.experimental.f12_focal import MolecularFocalProvider

provider = MolecularFocalProvider.cuda(
    architecture="sm_89",
    cache_directory=cache_directory,
)
hf = provider(next(req for req, _ in plan.steps if req.role == "hf_cabs"))
pair = provider(next(req for req, _ in plan.steps if req.role == "mp2_f12_correlation"))
```

The CUDA provider discovers the installed libraries. To use an SDK build,
pass the two shared-library paths as `shell_library` and `response_library`.
`budget_mib` is a per-component device budget, 1536 MiB by default. The
provider passes the same value to the DF reference, the TensorIR graph runner,
the native conventional DF-MP2 Delta-CV kernels and the CC-F12 pair pullbacks.
The DF reference applies it separately to its detached output, its native
capacity and its CUDA DF workspace. A kernel whose minimum workspace exceeds
it raises instead of using a fixed native allowance.

The budget does not limit the device memory of the whole card. Components
that keep memory resident, the separate reference allowances and the CUDA
context add to it. For example, a C5H12O2 Delta-CV energy and force
calculation in ha-pwCVTZ (581 orbital functions) with the default budget
reached 5443 MiB on an L20, as reported by `nvidia-smi`, including about
297 MiB of CUDA context. Production callers should pass a budget sized for
the card explicitly rather than rely on the default. Whole-card accounting
and capacity checks before an endpoint starts are the responsibility of the
pipeline that schedules it, not of the provider.

The HF+CABS term returns the DF-HF energy and forces plus the relaxed spin
singles contribution for an RHF or UHF reference. The CUDA fitted HF path
accepts small-core Br and I in its hydride, C/H monohalide and one-water domains. Select the scalar ECP on the
orbital record and use Coulomb-only fitting records. The pair term supports
all-active singlet RHF and UHF and
returns ordinary DF-MP2 correlation plus the explicit 3C(FIX) increment.
For a singlet RHF request, both spin channels use the same restricted
spatial orbitals. Their coefficient derivatives add before the restricted
orbital response.
Both contributions within that term share one physical determinant. The JK,
ordinary correlation and explicit correlation fits follow their respective
basis records. Matching identities share the fitted integrals and their
nuclear derivatives. The returned forces are Cartesian and in Eh/bohr.
Supply `high_order` for a custom conventional post-MP2 evaluator when
dispatching component requests individually.

For an existing reference and OBS-first integral set, call
`reference_cabs_singles(request, reference, one, complement=frame,
three_center=B, metric=M, whitener=W, relax=True)`. Here B has axes
(JKFit, AO, AO), and W M Wᵀ is the identity. The result contains the total
energy, both spin corrections and the four relaxed weights for S, Hcore,
B and M. With `relax=False`, the returned coefficient and complementary
kernel weights expose the separate electronic and overlap derivatives.

For the ordinary MP2 part of an all-active singlet RHF or UHF pair request, use
`benchmarks.f12_focal_ump2.molecular_df_ump2(request, ...)` from a source
checkout. The shell and one-electron libraries take the same path arguments
as the HF+CABS provider. On CUDA supply the CUDA metric and response
libraries and an FP64 TensorIR runner. The result contains the HF energy and
forces, the αα, αβ and ββ ordinary MP2 energies, and the relaxed correlation
forces in Eh/bohr. It uses the request's JKFit for the determinant and its
own MP2Fit for the pair integrals. The F12 correction must also be evaluated
before returning the complete `mp2_f12_correlation` component.

For precomputed OBS-only integral sources, pass a canonical
`FocalReference` and two `FocalCoulombSource` values to
`reference_df_ump2(request, reference, jk, mp2)`. Each source declares
the surface, OBS identity, fitting identity and backend, and provides
the raw three-center integrals, full-rank metric and matching whitener.
The result contains the three spin energies and six relaxed weights for
overlap, core, JKFit and MP2Fit. Contract these energy derivatives with
their corresponding nuclear integral derivatives and negate the sum.

For the explicit 3C(FIX) contribution from retained spin pairs, call
`benchmarks.f12_focal_increment.molecular_f12_increment(request, ...)`.
This source-checkout provider takes an all-active singlet RHF or UHF pair request and the
same native library and runner arguments as the ordinary provider. It uses
the full joint OBS/OptRI basis in both the JKFit Fock and the independently
fitted Gaussian operators. Its result contains the explicit F12 increment,
its relaxed force, the active-pair count and the fixed CABS rank. Add the
ordinary DF-MP2 correlation once to assemble the `mp2_f12_correlation` term.
Supply `expected_cabs_rank` for consistent displaced-geometry checks.

For already computed sources, the public experimental function
`reference_f12_increment` takes the request, canonical DF reference,
one-electron result, and two `FocalF12Sources` records. Its three weight
groups differentiate the one-electron, JKFit and F12Fit operators,
respectively. The Coulomb, geminal, squared geminal, geminal-Coulomb and
double-commutator operators in F12Fit each provide their own two- and
three-center nuclear derivative.
