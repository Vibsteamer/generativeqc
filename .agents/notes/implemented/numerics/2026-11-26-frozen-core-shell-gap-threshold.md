# Use a 0.05 Eh gap for frozen–active orbital admission

Status: implemented
Date: 2026-11-26

## Decision

Check each spin channel before correlation. Require the magnitudes of
the gaps that separate frozen from active occupied orbitals to be at
least 0.05 Eh. A frozen partition through a near-degenerate shell
depends on the chosen canonical representation. The gap floor also
bounds inverse frozen–active response denominators by 20 Eh⁻¹.

Version 1 defined the cut by an energy-ordered orbital count. For an
explicit frozen count that rule still applies, with the highest frozen
orbital and lowest active orbital defining the gap. A refusal reports
nearby admissible counts. The chemical valence spaces now select
canonical orbitals by [atomic shell identity](2026-11-27-frozen-core-shell-identity.md).
Their numerical check applies to every frozen–active pair.

## Evidence

The independent PySCF DF-HF spectra in
`tests/data/frozen_core_gap/nax_df_hf_spectra.json` cover 60 NaBr and NaI
cases (neutral, cation and anion, DZ/TZ, and a DZ bond-length scan from
0.8 to 2.0 times equilibrium). The old energy-ordered count of nine
frozen orbitals bisected the halogen d shell in all 100 spin channels.
The largest split at that count was 0.01896 Eh. The smallest shell
boundary gap with 14 frozen orbitals was 0.45546 Eh. These two bounds
bracket 0.05 Eh without relying on one bond length or charge.

A separate earlier screen found cases where O or F valence s levels
interleaved with the Na 2p shell. That ordering motivated the
versioned shell-identity approach and the selection of only the Na
1s core. The 60 halide spectra qualify the numerical threshold on
their recorded bases and occupations.

## References

- `python/generativeqc/_f12_frozen_core.py`
- `tools/generate_frozen_core_gap_references.py`
- `tests/python/test_frozen_core_shell_gap_halide_oracle.py`
- `tests/python/test_frozen_core_shell_gap_refusal_advice.py`
