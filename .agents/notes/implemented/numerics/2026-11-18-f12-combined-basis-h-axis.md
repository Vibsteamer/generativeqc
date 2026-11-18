# Combined-basis h functions in the DF angular moment polynomial

The QZ-F12 carbon OptRI basis adds an h shell to the combined orbital and
complementary basis. In a three-center integral this function can be on either
electron-one center, including both centers. The existing auxiliary-h operator
admits h on the electron-two fitting center.

An explicit maximum orbital angular momentum of five in the shared axis
moment builder covers h/h/h values and a single basis-factor derivative.
At most one of the first two powers may be raised to six. Translation gives the
third center derivative. The full h/h/h bound is order 15 for values and 16 for
the derivative. The methane control uses a g-limited fitting basis and its
highest total orders are 14 and 15. The existing CUDA operator inventories and
molecular admission checks keep their original domains. Separate device
qualification is needed for contraction, one-electron matrices and molecular
force assembly.

The regression integrates the two correlated Gaussian variables with a
tensor-product Gauss–Hermite rule. It checks the angular polynomial over
multiple radial nodes and verifies illegal raises and undeclared domains.

## Molecular source variant

The molecular shell builder's explicit `--cabs-h` option emits the FP64 angular
and radial bounds jointly for the raw Cartesian forward and its weighted
nuclear derivative. The same contraction, CCA ordering, rejection checks and
allocation ledger are used for Coulomb, the geminal, geminal/r, geminal-squared
and the double-commutator operator. The new library and build-cache directory
have distinct names. A single shared object can be selected by full path for
either backend and coexists with the conventional lower-angular library.
The conventional through-g kernel and the physical orbital DF method domain
keep their original bounds. A QZ-F12 one-electron forward and nuclear pullback
still require separate h-shell support.
