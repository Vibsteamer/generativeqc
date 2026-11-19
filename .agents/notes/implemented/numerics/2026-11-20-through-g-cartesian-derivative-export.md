# Orbital g in the Cartesian one-electron derivative export

The singleton and batch exporters admit orbital g and share the same pair kernel. The 5Z triplet H2 reference uses the exported Cartesian overlap and core-Hamiltonian derivative matrices after transformation to the public basis. Its frozen, weighted one-electron gradient can pass a separate generated-kernel check while the export still produces incorrect entries.

The retained dual-number recurrence uses a through-f Hermite workspace. A g orbital raises the bra index to four. The kinetic operator can raise the ket index to six, and the recurrence reads through Hermite index eleven on the zero boundary. An f–g nuclear attraction needs Coulomb order seven and a g–g attraction needs order eight. The export selects these bounds for any pair containing g. Lower-angular pairs keep the existing instantiations and the global direct-integral limits retain their established sizes.

The density-fitting regression checks full Cartesian overlap and core derivative matrices against independently displaced values at two geometries. Both export paths exercise moving basis centers and an external nuclear center. Scientific qualification of a molecular force also requires the final physical density, energy-weighted density, two-electron response and nuclear term.
