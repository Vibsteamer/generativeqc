# Molecular spin-resolved fixed-cusp CC sources

The equation sources use two independent fits. Apply JKFit to the extended
Fock and positive exchange, and use the F12Fit physical Coulomb metric with
all geminal operators for the pair columns and external OBS projection.
A shared panel requires complete basis identity. The returned canonical
reference determines the spin frames of all 18 W/C blocks and the common
V+B−X scalar. MP2Fit does not enter these CC equation sources.

The public forward path reuses the retained OBS/OptRI frame and the fixed
pair equations on the selected backend. On OH a/D with separate 150/152
fitting dimensions, a real PySCF/Libint reference transformed to the same
canonical occupied/virtual frames checks every W_vo, W_vv and C element
as well as the scalar. This qualification covers the inputs to the CC
stationarity solve.
