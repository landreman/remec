# Verification records

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
| 1 | 80 → 296 | 2.063 | 1.021 |
| 2 | 80 → 296 | 2.883 | 1.922 |
| 3 | 80 → 296 | 4.075 | 3.088 |

## Milestone 0.2 — common utilities

| Contract | Measured result | Automated test |
| --- | --- | --- |
| Named block norms | raw `(5, 12)` blocks scale to `(5, 6)` and combine to `sqrt(61)` | `test_named_block_norms_apply_physical_scales_before_combining` |
| Deterministic configuration | equivalent mappings serialize byte-identically, pin a SHA-256 digest, and reject invalid values | `test_config_serialization_is_canonical_for_mappings_and_dataclasses`, `test_config_serialization_rejects_noncanonical_values` |
| Structured timing | JSON event includes fields, outcome, and non-negative seconds; reserved fields are rejected | `test_structured_events_and_timer_emit_machine_readable_json` |
| Thread configuration | `3` is applied; `0` is rejected before NGSolve; the real setter runs in a subprocess | `test_thread_configuration_validates_before_calling_ngsolve`, `test_thread_configuration_executes_the_ngsolve_api_in_a_subprocess` |
| Checkpoint metadata | schema-1 JSON round-trips byte-identically with REMEC/NGSolve versions; invalid and future schemas are rejected | `test_checkpoint_metadata_round_trips_normalization_and_runtime_configuration` |
