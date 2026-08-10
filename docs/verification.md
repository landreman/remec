# Verification records

## Milestone 1.3 — Sovinec numerical-pollution regression

"Sovinec" refers to C. R. Sovinec, A. H. Glasser, T. A. Gianakon, et al.,
"Nonlinear magnetohydrodynamics simulation using high-order finite elements,"
*Journal of Computational Physics* **195** (2004) 355–386,
https://doi.org/10.1016/j.jcp.2003.10.004. Its anisotropic-conduction test
measures spurious cross-field transport introduced by a discretization whose mesh
is not aligned with the field. The physical perpendicular diffusivity is set to
zero, so the measured effective perpendicular diffusivity is numerical pollution.

The benchmark is the translated unit-square form of the `DESIGN.md` §8.3 test:
\(\psi=\sin(\pi x)\sin(\pi y)
=\cos(\pi(x-1/2))\cos(\pi(y-1/2))\),
\(\mathbf b=(\partial_y\psi,-\partial_x\psi)/|\nabla\psi|\),
\(Q=Q_0\psi\), \(\kappa_\parallel=1\), and \(\kappa_\perp=0\).
Thus the field is tangent to closed contours of the source. With homogeneous
Dirichlet data, the discrete central amplitude defines
\(\kappa_{\perp,\mathrm{num}}=Q_0/(2\pi^2\chi_h(1/2,1/2))\).

`test_sovinec_pollution_decreases_with_order_and_refinement` reads all nine rows
of `tests/manufactured/sovinec_pollution.csv`, recomputes them within relative
tolerance \(10^{-5}\), and
requires strict decreases at each adjacent order and refinement. The finest-pair
rates use \(\log_2(\kappa_{\perp,\mathrm{num}}(h)/
\kappa_{\perp,\mathrm{num}}(h/2))\). The algebraic diagnostic is the free-DOF
Euclidean residual divided by the larger of one and the free-DOF source norm; all
runs in the acceptance table must remain at or below \(10^{-6}\). Extended scans
return any finite residual so degradation can be recorded instead of aborting;
NaN or infinity still fails loudly. This residual criterion only validates the
direct solve and is not used as evidence of low pollution. Independent structural
diagnostics also require \(\int_\Omega(|\mathbf b|^2-1)^2\,dV<10^{-12}\) and
\(\int_\Omega(\mathbf b\cdot\nabla\psi)^2\,dV<10^{-12}\), so field normalization
and source tangency are not certified only by the recorded CSV values. A scaling
case with \(\kappa_\parallel=10\) and \(Q_0=3\) checks the expected amplitude,
effective-diffusivity, and dimensionless-ratio scalings. The implementation
derives \(\nabla\psi\) and the \(2k^2\) Laplacian eigenvalue from the same
coefficient function used as the source, and the test independently checks the
eigenvalue \(2\pi^2\). CSV comparisons use relative tolerance \(10^{-5}\), about
100 times the largest Linux/macOS variation measured in review. Every acceptance
row also requires \(\kappa_{\perp,\mathrm{num}}/\kappa_\parallel<0.2\), which
rejects an isotropic substitution independently of the recorded values.

| Degree \(p\) | Elements | \(h=1/4\) | \(h=1/8\) | \(h=1/16\) | Finest-pair rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 → 128 → 512 | 7.351e-2 | 1.967e-2 | 5.002e-3 | 1.975 |
| 2 | 32 → 128 → 512 | 1.627e-3 | 1.101e-4 | 7.193e-6 | 3.936 |
| 3 | 32 → 128 → 512 | 8.805e-6 | 1.622e-7 | 2.761e-9 | 5.877 |

At fixed mesh, raising \(p\) from 1→2 reduces pollution by factors 45.2,
178.7, and 695.4 from coarse to fine; raising \(p\) from 2→3 gives factors
184.8, 678.5, and 2605.2. Mutation checks confirmed that rotating the field
from tangent to normal and omitting normalization of \(\mathbf b\) both make
the regression fail.

The current benchmark deliberately uses a dedicated spatially varying, rank-one
M4a assembly because the milestone 1.1/1.2 verification helper accepts only a
constant direction and strictly positive \(\kappa_\perp\). Milestone 1.5 owns
unifying those paths bit-for-bit and restoring the §8.1 per-piece energy diagnostics.
Milestone 1.4 owns the finite-anisotropy axis and the full scheduled scan; this PR
keeps the fast, hardest-case \(\kappa_\perp=0\) table in PR CI. Extended scans
return finite residuals for diagnosis, but still fail loudly when the central
amplitude is non-finite or non-positive because the pollution metric is then invalid.

## Milestone 1.2 — oblique anisotropic K

The manufactured solution is χ=sin(πx)sin(πy) with homogeneous Dirichlet
data and the constant oblique conductivity
\(\mathbf K=2\mathbf I+5\mathbf b\otimes\mathbf b\),
\(\mathbf b=(3/5,4/5)\).  The source is evaluated analytically as
\(-\nabla\cdot(\mathbf K\nablaχ)\).  The automated test
`test_oblique_anisotropic_manufactured_convergence` reads the machine-readable error
table in `tests/manufactured/oblique_anisotropic_rates.csv`, requires L² rate at least
\(p+0.8\) and K-energy rate at least \(p-0.2\) on the finest refinement
pair, and checks each recorded error within 5%.  `test_oblique_solution_reports_separate_parallel_and_perpendicular_energy`
checks that both M4a contributions are reported separately and sum to the total.
The diagnostic uses a second, direction-sensitive manufactured field
\(\chi=\sin(\pi x)\sin(2\pi y)\) and checks each analytic contribution, so swapped
labels or a missing transverse projection fail.  This linear verification source changes
sign near the boundary and is not intended as an admissible non-negative reference source.
The conductivity ratio \(\kappa_\parallel/\kappa_\perp=3.5\) is deliberately mild;
these rates make no extreme-anisotropy or pollution claim.

| Degree \(p\) | Elements (coarse → fine) | L² rate | K-energy rate |
| --- | ---: | ---: | ---: |
| 1 | 72 → 288 | 1.887 | 0.965 |
| 2 | 72 → 288 | 3.054 | 1.968 |
| 3 | 72 → 288 | 4.089 | 3.008 |

## Milestone 1.1 — isotropic Poisson on `Slab2D`

The manufactured solution is \(\chi=\sin(\pi x)\sin(\pi y)\) with homogeneous
Dirichlet data and \(S_{\rm ref}=2\pi^2\chi\).  This is the isotropic unit-conductivity
reduction of note equation (M4a), \(-\Delta\chi=S_{\rm ref}\).  The automated test
`test_isotropic_poisson_manufactured_convergence` reads the machine-readable error table,
requires L² rate at least \(p+0.8\) and energy rate at least \(p-0.2\) on its finest
refinement pair, and checks each error against the recorded value within 5%; the results are in
`tests/manufactured/isotropic_poisson_rates.csv`. It also checks the homogeneous boundary
trace and the free-DOF direct-solve residual at roundoff.

| Degree \(p\) | Elements (coarse → fine) | L² rate | Energy rate |
| --- | ---: | ---: | ---: |
| 1 | 72 → 288 | 1.955 | 0.981 |
| 2 | 72 → 288 | 2.992 | 1.978 |
| 3 | 72 → 288 | 4.053 | 3.005 |

## Milestone 0.2 — common utilities

| Contract | Measured result | Automated test |
| --- | --- | --- |
| Named block norms | raw `(5, 12)` blocks scale to `(5, 6)` and combine to `sqrt(61)` | `test_named_block_norms_apply_physical_scales_before_combining` |
| Deterministic configuration | equivalent mappings serialize byte-identically, pin a SHA-256 digest, and reject invalid values | `test_config_serialization_is_canonical_for_mappings_and_dataclasses`, `test_config_serialization_rejects_noncanonical_values` |
| Structured timing | JSON event includes fields, outcome, and non-negative seconds; reserved fields are rejected | `test_structured_events_and_timer_emit_machine_readable_json` |
| Thread configuration | `3` is applied; `0` is rejected before NGSolve; the real setter runs in a subprocess | `test_thread_configuration_validates_before_calling_ngsolve`, `test_thread_configuration_executes_the_ngsolve_api_in_a_subprocess` |
| Checkpoint metadata | schema-1 JSON round-trips byte-identically with REMEC/NGSolve versions; invalid and future schemas are rejected | `test_checkpoint_metadata_round_trips_normalization_and_runtime_configuration` |
