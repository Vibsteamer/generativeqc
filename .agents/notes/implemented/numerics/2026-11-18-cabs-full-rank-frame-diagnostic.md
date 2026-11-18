# Direct constraint derivative for a complete CABS frame

The molecular gradient can contain a small residual in large, nearly linearly dependent orbital and OptRI bases. This optional benchmark computes a second overlap cotangent from the same physical molecular energy weights in a single central replay.

Let (C) contain all complementary columns, (R) span the fixed, embedded orbital AO space, and (S) be the overlap. Both frames are orthonormal in (S), with (R^T S C=0). For a frame-invariant scalar and (G=\partial E/\partial C), (C^T G) must be symmetric. Differentiating these constraints gives

\[
\bar S=-\operatorname{sym}(RR^T G C^T)-\tfrac12 C\operatorname{sym}(C^TG)C^T .
\]

Contracting through (R) and (C) avoids forming large products of the projector kernel on both sides of its cotangent. The diagnostic checks the full Gram matrix using the compensated physical metric and checks the gauge condition. A truncated CABS uses the original projector derivative, which includes the motion of the discarded subspace. The normal molecular endpoint retains its previous numerical route.

Tests exercise well- and ill-conditioned full spaces using an independent inverse-action derivative and finite differences, and a reduced-rank fallback. The comparison driver writes the true frame weight and both overlap cotangents for a component-level directional and Cartesian audit against the same molecular source.
