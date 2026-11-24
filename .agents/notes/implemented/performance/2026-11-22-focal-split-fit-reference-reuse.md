# Canonical reference reuse for a conventional triples fitting basis

When the public molecular focal recipe includes conventional (T) with an independent correlation fit, the complete basis identity differs from the HF+CABS and MP2-F12 basis identity. The DF-HF determinant depends on the orbital basis and JKFit. A second solve in the same geometry can waste work and can introduce a different canonical frame.

The molecular recipe reuses a retained determinant for a triples-only basis with matching surface, execution backend, orbital basis and JKFit. The (T) role retains its declared correlation fit and full basis identity. F12 and CABS evaluation continue to use the complete role-specific basis records and their own ranks. A user-supplied seed provider continues to control each basis independently. For pure-energy displacements from an analytic center, reuse requires a shared center determinant and an identical occupied-overlap threshold.

The regression evaluates nonzero LiH (T) with an independent correlation fit. The analytic reference and both signed energy pairs use the same role-specific source and give the same hydrogen-z force within the existing tolerance. A stricter center seed for the (T) role still enforces its individual overlap requirement.
