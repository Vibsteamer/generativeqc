# Fitted scalar ECP references for methyl bromide and methyl iodide

Status: implemented
Date: 2026-11-20

The first methyl bromide L20 trial stopped at the fitted-HF hydride topology
gate. It reached no SCF or F12 evaluation. The methyl-halide basis records and
independent DF references were already prepared.

The Python entry, native basis template, fitted HF preparation and CUDA spin
domain recognize the same C1H3X1 topology. Carbon and hydrogen are all electron.
Bromine uses a ten-electron core and iodine a twenty-eight-electron core. The
new requested states are the neutral RHF singlet and the vertical UHF doublet
cation. Scalar ECP and shell/fit backend checks still apply to each calculation.
The shipped presets use separate JK fitting and correlation fitting and carry
the element-specific OptRI choices. The universal JK fitting record in these
presets is a diagnostic choice.

Host basis admission covers both cardinalities and both states. Physical
qualification requires an independent occupied-space check and signed energy
differences for the exact requested force scope. The conventional triples
and fixed-geminal correlation layers have their own angular and storage
limits. A successful fitted reference therefore qualifies that reference
alone.
