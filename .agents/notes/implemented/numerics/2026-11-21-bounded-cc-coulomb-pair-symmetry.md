# Symmetric Coulomb pairs in bounded molecular CC

The bounded CCSD-F12b virtual ladder assumes equal Coulomb factors for orbital pairs `(p,q)` and `(q,p)`. On the frozen CH₃Br/ha-D RHF and IH₂/ha-T UHF inputs, offline replay of the MO projection produced roundoff at the 10⁻¹² scale. Inverting Coulomb metrics with condition numbers 1.46 × 10⁸ and 5.10 × 10⁸ amplified the skew above the existing 10⁻¹⁰ packed-factor gate. The original L20 replays stopped at that gate before amplitude or Lambda actions.

The bounded ordinary Hamiltonian now projects the raw OBS Coulomb tensor onto symmetric orbital pairs before the metric solve. The same projection is part of the AO-to-MO graph for its pullback. The transposed graph acts on the sum of direct raw and implicit fit cotangents. The raw-input check and the existing packing tolerance remain in force. Dense ordinary validation and the explicit F12 operator sources retain their existing graphs.

On the frozen geometries and their independently archived canonical orbitals, the projected fitted factors pass the original tolerance. Sampling 2048 matched pair columns changed the reconstructed Coulomb blocks by at most 5.6 × 10⁻¹² Eh. This figure characterizes the offline comparison. The L20 stationary replay still qualifies the native implementation and its nuclear response.
