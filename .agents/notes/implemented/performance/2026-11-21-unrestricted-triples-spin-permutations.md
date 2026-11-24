# Unrestricted triples energy with unique same-spin tuples

Status: implemented
Date: 2026-11-21

The conventional unrestricted (T) energy contracts two antisymmetric
numerators with a denominator that depends only on their occupied and virtual
sets. Each repeated same-spin index contributes zero. Permuting either set
changes both numerator signs together. The solver now visits increasing
occupied and virtual tuples separately within each spin group and restores
the omitted multiplicity. The primal work admission uses that same tile count.
The existing analytic reverse traversal and its capacity checks retain their
own work counts.

Real-spherical f orbital shells may enter the all-electron full DF
unrestricted CC route via the existing integral providers. The gate retains
its existing boundary for ECP systems and does not expand the supported
Cartesian or partially fitted routes. Direct and canonical-reference triples
pipelines remain subject to their own capacity checks.

The independent PySCF fixtures in tests/python/test_f12_focal_ucc_triples.py
and tests/python/test_uccsd_public.py cover energy and derivative consistency.
For standard a-D and ha-D OH, the primal tile counts fall from 16,384,704
to 2,782,864 and from 13,105,560 to 2,229,110 respectively. These counts are
algorithmic work measurements. The coupled-cluster iterations, integral
transform and response are still separate costs.
