# Select focal frozen cores by atomic shell identity

Status: implemented
Date: 2026-11-27

The first focal frozen-core definition froze Na 1s, 2s and 2p. In the
140-molecule sodium study, 29 canonical spectra placed an O, F or Cl valence
s orbital below part of the Na 2p core. The first-by-energy partition raised
the DF-MP2 correlation energies by 6.3–126 mEh relative to partitions that
kept the full sodium shell frozen. The C2FNaO scan crossed an avoided
crossing. Freezing by identity under the first definition could change the
chosen canonical orbitals between adjacent scan points. [Earlier shell-gap
qualification](2026-11-26-frozen-core-shell-gap-threshold.md)

Definition version 2 correlates Na 2s and 2p in both chemical valence
spaces. Na retains the 1s core in the valence and CV-active spaces. Br and
I retain the production (n−1)spd core after subtracting any ECP pairs.
NaI can still have Na 2s below the I 4d canonical manifold. The partition
must follow atomic identity across that ordering.

Project each occupied molecular spin frame onto the embedded,
Löwdin-orthonormalized spherical atomic HF core. Freeze the canonical
orbitals of majority core character, require a complete core span, and
apply the 0.05 Eh absolute gap gate to every actual frozen–active pair.
The atom reference uses GenerativeQC's spherical restricted atomic solver
with the same orbital basis and scalar ECP. Its process-local cache is
bounded and geometry independent. The atomic solve supplies the selection
projector. The molecular calculations use the converged molecular
determinant.

Store the occupied column permutation on the reference and apply it to
the detached arrays and their native owner together. The determinant
projector, AO density, Fock, virtual frame and nuclear derivatives of the
Hamiltonian are invariant under an occupied permutation. ΔCV embeds its
CV-active and valence choices as nested prefixes. ΔSR selects in each
canonical reference and applies the same gap criterion on the two sides.
Explicit frozen counts continue to use canonical energy order.

The gap criterion bounds the inverse denominator of the frozen–active
response by 20 Eh⁻¹. An alternative based on the current frozen–active
response amplitude admitted gaps as small as 0.0033 Eh. Small amplitudes
at one geometry did not bound the curvature along the bond scans.
A different alternative would freeze the projected atomic subspace
itself. It changed some otherwise admitted DF-MP2 energies by as much
as 17 mEh and would require a derivative of the moving atomic projector.
Keeping only Na 1s releases the full 2s/2p subshell instead of changing
the canonical derivative model or loosening the denominator criterion.

A chemically selected frozen space must leave a correlated occupied orbital
in each populated spin channel. Releasing the full Na 2s and 2p shells
helps preserve an active space for highly charged Na compounds.

The correlated space identity includes the definition version.
Production assembly takes the version and space from the typed valence
result and matches the corrective terms against it. For versioned Na
basis input hashes and the endpoint contract, see the [focal user
guide](../../../../docs/user/ccsdt_cbs.md).

Prior evidence and the orbital-basis rationale are recorded by
[Curtiss et al.](https://doi.org/10.1063/1.1366337),
[Iron, Oren and Martin](https://doi.org/10.1080/0026897031000094498), and
[Hill and Peterson](https://pubs.rsc.org/en/content/articlelanding/2010/cp/c0cp00020e).
