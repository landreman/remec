# Verification records

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
