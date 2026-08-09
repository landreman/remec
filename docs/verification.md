# Verification records

## Milestone 0.2 — common utilities

| Contract | Measured result | Automated test |
| --- | --- | --- |
| Named block norms | raw `(5, 12)` blocks scale to `(5, 6)` and combine to `sqrt(61)` | `test_named_block_norms_apply_physical_scales_before_combining` |
| Deterministic configuration | equivalent mappings serialize byte-identically, pin a SHA-256 digest, and reject invalid values | `test_config_serialization_is_canonical_for_mappings_and_dataclasses`, `test_config_serialization_rejects_noncanonical_values` |
| Structured timing | JSON event includes fields, outcome, and non-negative seconds; reserved fields are rejected | `test_structured_events_and_timer_emit_machine_readable_json` |
| Thread configuration | `3` is applied; `0` is rejected before NGSolve | `test_thread_configuration_validates_before_calling_ngsolve` |
| Checkpoint metadata | schema-1 JSON round-trips byte-identically with REMEC/NGSolve versions; invalid and future schemas are rejected | `test_checkpoint_metadata_round_trips_normalization_and_runtime_configuration` |
