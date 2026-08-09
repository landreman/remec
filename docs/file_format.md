# Checkpoint file format

## Schema 1 metadata envelope

`CheckpointMetadata.to_json()` produces canonical UTF-8 JSON: sorted object keys and
compact separators. Readers accept schema version `1` only and reject unsupported or
malformed metadata with `CheckpointVersionError`.

Required fields are:

| Field | Type | Purpose |
| --- | --- | --- |
| `schema_version` | integer (`1`) | Compatibility gate |
| `configuration` | object | Canonical normalization and runtime configuration |
| `state_names` | array of strings | Names of state vectors carried by a later payload |
| `git_commit` | string | Checked-out source revision; CI SHA or `unknown` only when Git is unavailable |
| `platform` | string | Platform identifier |
| `remec_version` | string | Installed REMEC package version |
| `ngsolve_version` | string | Installed NGSolve version |

This envelope intentionally contains metadata only. Future checkpoint payload writers
must preserve it and add mesh, FE space definitions, accepted state vectors, harmonic
basis/flux coefficients, profiles, transport parameters, iteration history, and saved
diagnostics as required by `DESIGN.md` §24.
