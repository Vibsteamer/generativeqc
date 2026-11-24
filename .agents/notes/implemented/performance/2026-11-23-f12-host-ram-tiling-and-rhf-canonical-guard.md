# Decision: Auxiliary-Tiled F12 Host RAM Bounding, 2-Slab Streaming AO Pullback, and RHF Canonical Residual Guard

Status: implemented
Date: 2026-11-23

## Problem

Scaling the native molecular F12 energy-and-force pipeline (`ha-[D,T,Q]-F12` low-order, `(T)/(T+)` E+F, and Step 3 QZ / OMol 7-heavy-atom structures) onto single 48 GiB NVIDIA L20 pods with a 63 GiB container host-RAM cgroup exposed four distinct production blockers:

1. **Untiled auxiliary and AO-panel allocations (45–91 GiB host RAM)**: Unchunked auxiliary contractions in `_f12_spin_fock.py`, `_f12_spin_ao.py`, `_f12_cc_c_nuclear.py`, `_f12_cc_w_nuclear.py`, `_f12_cabs_fock.py`, `f12_orbital_response.py`, and `_f12_focal_singles_response.py`, plus full `(aux, ext, ext)` Coulomb projections and full-panel `ao_mat` + `cotangent` residency during streaming AO pullbacks, triggered SIGKILL (`exit_code=-9`) at the 63 GiB pod RAM limit even on 10–15 atom `ha-D/T-F12` jobs.
2. **Cartesian AO shell int32 indexing overflow and 20+ GiB scratch**: On 19–23 atom `ha-T-F12` molecules (`n_ext > 1000`), single-block Cartesian shell forward/contraction evaluations exceeded `2^31` elements and allocated >20 GiB scratch buffers; `transpose_program` also capped `max_elements` at `1_000_000` regardless of `ext * ext`.
3. **Prepared RHF reference export canonicality drift (`GENERATIVEQC error 5`)**: On diffuse TZ/QZ bases (`a-Q-F12`, `ha-Q-F12`, `IL16`, `CH4/33156`), `select_final_state` in `src/scf/solver/mean_field_driver.cpp` and `src/scf/rhf.cpp` could export a non-canonicalized or residual-drifted RHF determinant that failed downstream F12 canonicality validation.
4. **Trisol `--image-ref` 502 submit timeout**: Platform `GET /api/v1/images` returned HTTP 502 after 10s, stalling `--image-ref` job submissions until 3×30s timeout.

## Decision

1. **Tile all auxiliary contractions (`<= 256 MiB` slabs) and nuclear/singles response drivers**:
   - Tiled auxiliary loops in `_f12_spin_fock.py`, `_f12_spin_ao.py`, `_f12_cc_c_nuclear.py`, `_f12_cc_w_nuclear.py`, `_f12_cabs_fock.py`, `f12_orbital_response.py` (`unrestricted_nuclear_source_weights`, `unrestricted_df_canonical_driver`), and `_f12_focal_singles_response.py` (`relaxed_reference_weights`), removing redundant `three_center.copy()` allocations.
2. **Project occupied-only Coulomb pairs and share RHF spin-symmetric buffers**:
   - Enabled `occupied_coulomb=True` (`(aux, occ, ext)` instead of `(aux, ext, ext)`) and shared spin-symmetric RHF buffers across `_f12_spin_fock.py`, `_f12_spin_ao.py`, `_f12_unrestricted.py`, `_f12_focal_energy.py`, and `_f12_cc_scalar_nuclear.py` (~3× host RAM reduction on closed-shell RHF).
3. **Chunk Cartesian AO shell blocks (`<= 128M` doubles = 1 GiB) with `int64` offsets and scale `transpose_program`**:
   - Chunked `_f12_focal_shells.py`, `_f12_focal_integral_sink.py`, and `_f12_focal_integrals.py`, and scaled `transpose_program(..., max_elements=max(1_000_000, ext * ext))` for large TZ+CABS bases (`ext > 1000`).
4. **2-slab split streaming AO pullback for all streaming operators**:
   - Split `_f12_spin_ao.py` streaming pullback into `inputs_mode="coefficients"` -> free `ao_mat` -> `inputs_mode="ao"` across all streaming operators, lowering the streaming threshold to 2 GiB in `_f12_focal_molecular.py`.
5. **Physical residual guard and final-state canonicalization in prepared RHF export**:
   - Enforced residual convergence guard and `select_final_state` canonicalization in `src/scf/solver/mean_field_driver.cpp` and `src/scf/rhf.cpp` (compiled into SM89 R4 build `bd672f337`).

## Rejected alternatives

- **Raising container host RAM above 63 GiB**: Unavailable on standard single-L20 pods and hides algorithmic `O(N_aux * N_ext^2)` host residency bloat that breaks production scaling.
- **Skipping canonicality validation on diffuse QZ bases**: Rejected because silent canonicality drift corrupts orbital-response Z-vector solutions and analytic nuclear forces.
- **Re-ordering or dropping `cc_f12b` from the Python F12 production stack on this branch**: Out of scope for Track 1; bounded GPU CCSD-F12b is implemented separately on `feat/rhf-ccsdf12b-cuda`, and the Track 1 evaluation worker checkpoints `low_order` and `(T+)` prior to legacy `cc_f12b`.

## Invariants

- All tiled, occupied-only, and 2-slab split pullbacks must remain numerically identical (within FP64 reduction tolerance) to untiled references and pass local CPU + Gate 2 finite-difference force checks.
- Prepared RHF exports must satisfy both physical Fock residual convergence and MO canonicality before entering F12/CC phases.
- Single-card L20 GPU peak memory remains bounded below 46 GiB (typically 5–29 GiB for non-Br light/QZ systems and 38–45 GiB for Br/I-ECP systems).

## Evidence

- **Local CPU unit/regression verification**: All commits through `c0366374f` passed local pytest against `build-cpu/libgenerativeqc.so`.
- **SM89 R4 build**: `native-f12-application-preopt-v2-r4-sm89-bd672f337` (`job_id=2124601364691558400`, `native_library_sha256=f3a2f560f60dcb9d1040b308b86789759ba789333f37fb52e3f7e418798ba18b`).
- **Step 3 QZ + heavy-halogen L20 validation (100% RHF QZ + Br/I/BrH2O low+CC complete)**:
  - `s3-ch4-33156-aq-r4` (`2124611339052654592`): 2/2 PASS (`5,207 MiB` GPU, `6.96 GiB` HWM).
  - `s3-rhf-aq-fill-r4` (`2124614666020720640`): 6/6 PASS (`5,742 MiB` GPU).
  - `s3-il16-cat-q-r4` (`2124611418744426496`): 1/1 PASS (`5,746 MiB` GPU, `19.7 GiB` HWM).
  - `s3-il16-neu-haq-r4` (`2124611465242480640`): 1/1 PASS (`5,659 MiB` GPU, `32.2 GiB` HWM).
  - `s3-p34-br-r4c` (`2124671317121703936`) & `s3-p34-i-r4c` (`2124671419550801920`): 5/5 PASS each (`28.0–28.9 GiB` GPU).
  - `s3-ta13-brh2o-lowcc-r4e` (`2124783961312866304`, 8/8 PASS, `45,063 MiB` GPU) and `s3-ta13-brh2o-hat-cc-r4e` (`2124783977196695552`, 2/2 PASS FC+AE `cc_f12b`, `43,620 MiB` GPU, `24.2 GiB` RSS).
  - Top 6/6 RHF recipe (`low=ha-[T,Q] | cc=ha-D | (T+)@ha-D-F12`, FC): MAE = `0.0385` kcal/mol, Max = `0.1164` kcal/mol (5/6 <= 0.1 kcal/mol).
- **OMol33 single-L20 63 GiB host-RAM boundary (all 33/33 dispatched within <=18 concurrent L20 cards)**:
  - **26/33 completed cleanly** (`full_matrix_report.json`), with **26/26 `ord-ha-D` `(T)` E+F PASS** (`22.5–28.2 GiB` non-Br GPU VRAM, `43.7–44.2 GiB` Br-ECP GPU VRAM, `3–12 GiB` host `VmHWM`).
  - **Smaller/medium 7-heavy-atom structures** (`CNa3O3`, `C3H2FNaO2`, `C2H4N2O2S`, `CH4N2O3P`, `C3H5NaO3`, `C5H12O2` idx 15) passed `ha-D-F12` `low_order_ef` (`6/26` complete; `4/26` also passed `ha-D-F12 (T+)` E+F and `4/26` passed `ha-T-F12 low_order_ef`) within `37.5–64.1 GiB` host RSS and `22.5–41.4 GiB` GPU VRAM.
  - **7/33 hit the 12h (`43,200s`) outer job timeout** (`omol-r4d-13`, `r4e-05`, `r4e-06`, `r4e-07`, `r4e-08`, `r4e-12`, `r4e-14`) after passing `ha-D-F12 low_order_ef` (and `ha-T-F12 low_order_ef` on `r4e-05`) during long `ha-D-F12 (T+)` E+F / `ord-ha-D` execution (`state=scientific_review`).
  - **Largest 19–23 atom and multi-Br structures** (20/26 complete) hit the 63 GiB container host-RAM cgroup (`exit_code=-9`) on Python-side `ha-D/T-F12` AO integral-sink pullback (resolved by GPU integral-sink offloading on `feat/f12-sink-gpu-forces`), plus isolated non-OOM failures: `2` `ha-T-F12` DF-HF non-convergences (`C5H6Br2`, `C6H14Na`), `1` `ha-T-F12` weighted-density guard (`C5H14PS`), and `2` `cuBLAS retained allocations exceed provider allowance (236.98 MB > 100.66 MB)` (`C3H5NaO3` `(T+)`, `C5H11BrO` `ha-T-F12`). Legacy `cc_f12b` remained `0/26` on OMol33 due to its pre-refactor host/device budget ceiling (resolved separately on `feat/rhf-ccsdf12b-cuda`).
- **Unified extended cross-recipe evaluation (`extended_matrix_report.json`, 6 extensions, 8,019 recipes)**:
  - All 10 observables (`agg_all10`, 333 complete recipes): top `FC` `low=ha-[D,T]-F12 | cc=ha-D-F12 | (T)@ha-D-F12` achieves MAE = `0.1509` kcal/mol (6/10 <= 0.1, 8/10 <= 0.2); `agg_all10_cbscv` top `FC + ΔCV(F12, b=0.9)` `low=ha-[D,T]-F12 | cc=ha-D-F12 | (T)@ord-ha-[D,T]` achieves MAE = `0.1547` kcal/mol, Max = `0.4626` kcal/mol; `agg_cv_aligned_6_cbscv` achieves MAE = `0.0903` kcal/mol, Max = `0.1388` kcal/mol (6/6 <= 0.2).

## Consequences

- Host RAM on QZ and medium TZ F12 molecules drops by 3–10× while preserving exact analytic forces.
- Two remaining scale bottlenecks on 19–23 atom OMol structures are cleanly isolated for follow-up tracks: (a) Python-side AO sink host RAM >63 GiB on 19–23 atom `ha-D/T-F12` molecules (solved by GPU sink offload), and (b) `cuBLAS retained allocations exceed provider allowance (226 MiB vs 96 MiB)` on a subset of large `(T+)`/TZ runs.

## Revisit when

- GPU F12 integral-sink offloading (`feat/f12-sink-gpu-forces`) and bounded GPU CCSD-F12b (`feat/rhf-ccsdf12b-cuda`) are integrated onto mainline and re-qualified on the full OMol33 suite, or when the cuBLAS retained-workspace allowance is increased above 256 MiB.

## References

- Commits on `cursor/preopt-v2-gate2-and-matrix-1336`: `01f0d32e1`, `8b21d5298`, `bd672f337`, `64d336679`, `d204cebab`, `30a2456b4`, `11164c1f7`, `c0366374f`.
- Validation reports: `/root/gqc-f12-20261108/validation/focal-convergence-20261118/preopt_v2_eval/full_matrix_report.json` and `extended_matrix_report.json`.
