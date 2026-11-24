# Full-basis CABS one-electron batch through i

Status: implemented
Date: 2026-11-20

The i-pair qualification uses a separate lower provider. An ECP molecular
source needs the effective-charge one-electron matrices in the same combined
OBS and OptRI AO order, including pairs whose two shells have angular
momentum below i.

The explicit full-basis i batch uses the existing real-spherical recurrence,
projection and packed nuclear Jacobian for all triangular shell pairs. The
caller provides effective charges with the original elements and independent
residual ECP data. These point-charge integrals exclude the residual local and
nonlocal ECP. Existing high-only h and i operations remain available.

The full batch shares the element ordering and arbitrary real weight
contraction with the high-only batch. It reserves the complete Cartesian
scratch and packed output before starting the CUDA allocation. Using one
bounded batch avoids an artificial lower-space element substitution or a
second AO ordering and charge convention. Its cost grows with the number
of lower pairs. It remains opt-in for retained Br/I controls until the
live residual-ECP provider and molecular F12 response are qualified.

The focused ABI test compares the native S/Hcore values and their contracted
nuclear derivative to independent libcint integrals. It also checks the
input-domain failures and the preservation of the high-only path.
