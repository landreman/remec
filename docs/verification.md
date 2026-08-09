# Verification records

## Milestone 0.2 — common utilities

| Contract | Measured result | Automated test |
| --- | --- | --- |
| Named block norms | `(5, 12)` blocks combine to `13` | `test_named_block_norms_report_each_block_and_the_combined_norm` |
| Deterministic configuration | equivalent mappings serialize byte-identically and hash identically | `test_config_serialization_is_canonical_for_mappings_and_dataclasses` |
| Structured timing | JSON event includes the requested fields and non-negative seconds | `test_structured_events_and_timer_emit_machine_readable_json` |
| Thread configuration | `3` is applied; `0` is rejected before NGSolve | `test_thread_configuration_validates_before_calling_ngsolve` |
| Checkpoint metadata | schema-1 metadata round-trips exactly; schema 2 is rejected | `test_checkpoint_metadata_round_trips_normalization_and_runtime_configuration` |
