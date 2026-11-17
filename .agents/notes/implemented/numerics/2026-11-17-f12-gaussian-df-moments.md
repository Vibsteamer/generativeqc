# Native radial integrals for fitted Slater geminals

Status: implemented at the raw Gaussian integral boundary
Date: 2026-11-17

## Geometry and radial moments

Reuse the Coulomb product-center binder and its ordinary Hermite prefactor for the two-center auxiliary metric, three-center orbital-pair/auxiliary terms and four-center controls. For the auxiliary metric and fitting cases the unit slots have exponent and angular momentum zero. The integral operator supplies only the normalized radial seeds. This keeps coordinate shifts and Gaussian-product derivatives identical across operators.

Let ρ be the reduced exponent of the two electronic Gaussian products, T = ρ|P−Q|², z the exponent of exp(−z r₁₂²), g = z/(ρ+z) and h = ρ/(ρ+z). The nth Hermite seed is (−∂/∂T)ⁿ applied to the zeroth seed. For the Gaussian it is

    M_n = √π h exp(−gT) gⁿ / (2√(ρ+z)).

For exp(−z r₁₂²)/r₁₂, transform the Coulomb Gaussian integral with u = s/(ρ+z+s²)¹ᐟ². The result is

    M_n = h exp(−gT) ∫₀¹ (g+h u²)ⁿ exp(−h T u²) du.

Use a single ordinary Boys vector at hT and repeated positive binomial averaging. This eliminates per-moment numerical quadrature in the CUDA operator. The third primitive uses −∂/∂z of the Gaussian seed for r₁₂² exp(−z r₁₂²). Its product with 4 z₁ z₂ at z = z₁+z₂ produces the dot product of the two radial gradients on electron 1.

## Fit identity and cross terms

A six-Gaussian fit is prepared from β in bohr⁻¹. Scale each primitive exponent by β². The raw Psi4 and the 1/β-normalized Slater conventions are explicit choices. The fixed-amplitude contraction and its integral convention must be qualified together. Geometry changes keep the chosen β fixed. Products use 21 distinct terms when both references denote the same fit and 36 for two different fits. The mixed products require both exponents, including in the gradient product. Accumulate all primitive terms with compensated FP64 summation and apply the shared geometry prefactor once. [Psi4 1.11 numeric fit](https://github.com/psi4/psi4/blob/v1.11/psi4/src/psi4/libmints/mintshelper.cc) · [A normalized Slater factor in CC theory](https://doi.org/10.1063/1.4963748)

## Qualification

`tests/python/test_f12_radial_integrals.py` uses independent real-space radial integration for physical 2c/3c/4c s primitives and their nuclear-center derivatives, with displacements of 3×10⁻⁴ and 10⁻⁴ bohr. It tests radial orders 0 through 20 across 270 input sets. Forty-eight contracted 2c/3c cases include the two normalizations, three ordered β combinations (0.9/2.2, 2.2/0.9 and 1.4/1.4), four operators, translation or directional derivative checks, and independent Psi4 Gaussian coefficients. Increasing the real-space oracle quadrature from 320 to 480 nodes changes the sampled values by less than 10⁻¹³ relative to max(1, |integral|). Resolving the tightest primitive matters for the gradient product.

Run the same test on an allocated CUDA device with `GENERATIVEQC_TEST_CUDA=1`. The device build and host build use identical scientific functions. The tested radial order counts moments. Certification of a complete g/h/i shell or an ECP molecule also requires the corresponding contracted spherical angular blocks, transforms and response.
