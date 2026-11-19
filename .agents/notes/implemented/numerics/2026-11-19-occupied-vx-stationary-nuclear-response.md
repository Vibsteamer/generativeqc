# Nuclear response of the occupied V and X geminal tables

Implemented on 2026-11-19 for the full-rank, closed-shell OBS+CABS qualification.

The stationary CC cotangents for V and X have four occupied legs. Transpose the existing all-occupied projector graph separately for each source. This preserves all ordered pairs, the OBS×OBS sector and both occupied×CABS sectors. Reverse the common Coulomb fit through the same positive-definite metric solve. Each raw three-center integral contributes to both orbital-coefficient legs. The assembled AO weights use the original, unsymmetrized Frobenius convention.

The occupied and virtual orbitals follow the RHF response in a parallel-transport gauge. Orthogonality to the moving OBS determines the OBS component of each CABS derivative. The one-sided AO overlap derivative fixes the internal CABS gauge. These equations assume the OBS shells embed in a complete combined basis. A truncated CABS or a changing rank needs its own response.

The real-water test uses 5 occupied orbitals, 43 virtual orbitals, 158 combined functions, 198 fitting functions and 23,435 independent CC amplitudes. All nine Cartesian coordinates have signed points at 0.0004 and 0.0002 bohr. With the Cholesky controlled reference, the maximum V and X source errors are 4.08×10⁻¹⁰ and 1.15×10⁻⁹ Eh/bohr across both steps. The corresponding stock spectral-reference V errors are 7.36×10⁻⁶ and 1.93×10⁻⁵ Eh/bohr. Both calculations give the same analytic V gradient. Keep the metric convention and the individual source errors alongside the common 2×10⁻⁶ Eh/bohr gate.

The CPU oracle contracts first derivative shells of the neutral Gaussian geminal, its square, its Coulomb product and the Coulomb operator. The molecular sum includes V, X, Fock and ERI analytically, and includes signed derivatives of the external V, C, B, HF and CABS singles. Replacing another signed term requires checking the individual term and the sum against the same reconverged roots. The source-weight and complementary-frame APIs are exercised by `tests/python/test_f12_occupied_vx_nuclear.py`.
