# Complementary h shells in the all-electron UHF center

The matched-snapshot UHF molecular benchmark uses the same physical spherical shell
partition as the closed-shell center for overlap and the core Hamiltonian. The low
shell solver evaluates all low–low pairs. The explicit h library evaluates every
remaining pair, including h–h. The same geometry-local batch supplies the compact
nuclear Jacobian to the reverse pass. The contraction includes both spin channels
through the common, response-corrected AO weights and includes nuclear attraction
center motion.

The geometry input can fix `options.cabs_relative_cutoff` for the projector. The
original default is 1e-9. The benchmark checks the supplied complementary rank and
the matched central UHF orbital and Coulomb fits. The 5Z triplet H₂ protocol uses
1e-11 with the full aug-cc-pwCV5Z-RI complementary basis.

The physical qualification requires a matched UHF snapshot, measured L20 execution,
and all six molecular force components against independently reoptimized energies
at two steps per geometry. Host checks on the h-containing S/Hcore matrices and
nuclear derivatives cover the shell assembly for both radii.
