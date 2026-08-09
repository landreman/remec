# STATUS — milestone ledger

Single source of truth for what is done and what is next. Agents read this first and
update it as the last step of every milestone. Authoritative milestone definitions and
acceptance criteria live in `DESIGN.md` §25; this file tracks state and measurements.

Legend: `[ ]` not started · `[~]` in progress (branch open) · `[x]` merged and green in CI

A milestone may only start when every milestone in the previous phase is `[x]`
(`DESIGN.md` §25). Phase 7 may run in parallel with Phase 8.

---

## Phase 0 — repository and conventions

- [x] **0.1** Package skeleton — `DESIGN.md` §20, §23 · note: n/a
  <br>Acceptance: `pip install -e .` + `pytest` in a clean env, no compiler, no MPI, on Linux and macOS.
  <br>Measured: local macOS / CPython 3.12.2 / NGSolve 6.2.2606 — `make check`,
  `make smoke`, and `make wheel-smoke` pass. Additional review hardening is in progress
  before the PR is merged.
  <br>Next: use the project `.venv` (isolated CPython 3.12); the supplied Conda Python
  interpreters segfault during pytest collection in this terminal environment.
- [~] **0.2** Common utilities — `DESIGN.md` §6, §21, §24 · note: n/a (normalization is `DESIGN.md` §6)
  <br>Acceptance: block norms, structured logging, timing context, deterministic config serialization, thread config, checkpoint metadata; round-trip tested.
  <br>Measured: 10 Phase-0.2 contract tests pass: physically scaled block norms;
  canonical configuration serialization/digest and rejection guards; JSON event/timing
  records; validated NGSolve thread configuration; and byte-identical, versioned
  checkpoint-metadata round-trip. Awaiting PR merge and CI.
  <br>Next: provide the human-console log adapter at a solver run entry point; this
  milestone supplies the machine-JSON event stream only.

## Phase 1 — anisotropic scalar kernel

- [ ] **1.1** Isotropic Poisson on `Slab2D` — `DESIGN.md` §8.1, §16.2 · note: §4
  <br>Acceptance: manufactured convergence at the expected order in L² and energy norms.
  <br>Measured: —
- [ ] **1.2** Oblique anisotropic K — `DESIGN.md` §8.1, §8.2 · note: §4
  <br>Acceptance: parallel/perpendicular diagnostics; order scans.
  <br>Measured: —
- [ ] **1.3** Pollution benchmark — `DESIGN.md` §8.3 · note: §4
  <br>**Phase gate.** Acceptance: machine-readable table; measured pollution decreases systematically with order *and* refinement. No coupled work proceeds until this holds.
  <br>Measured: —
- [ ] **1.4** Closed-field and island frozen-field tests — `DESIGN.md` §8.3, §22 · note: §4, §7
  <br>Measured: —
- [ ] **1.5** Refactor to `AnisotropicDiffusionSolver` — `DESIGN.md` §8.4
  <br>Acceptance: interface extracted, results bit-for-bit unchanged.
  <br>Measured: —

## Phase 2 — level-set volume and transplant

- [ ] **2.1** Mollified V_χ — `DESIGN.md` §12.1, §12.2 · note: §6
  <br>Acceptance: analytic circle/sphere; monotone tabulation.
  <br>Measured: —
- [ ] **2.2** Profiles + transplant — `DESIGN.md` §12.5 · note: §6
  <br>Acceptance: exact enclosed-volume and layer-cake tests.
  <br>Measured: —
- [ ] **2.3** Differentiable map — `DESIGN.md` §12.6 · note: §6
  <br>Acceptance: JVP agrees with finite differences.
  <br>Measured: —
- [ ] **2.4** Cut-cell reference (optional) — `DESIGN.md` §12.4
  <br>Measured: —

## Phase 3 — M3 kernel

- [ ] **3.1** Direct-u weak form, frozen (B, p) — `DESIGN.md` §9.1 · note: (M3)
  <br>Measured: —
- [ ] **3.2** SUPG + manufactured tests — `DESIGN.md` §9.1 · note: (M3)
  <br>Acceptance: includes the test that fails conspicuously if the `D_u ∇⊥u·∇p` term is dropped (§22).
  <br>Measured: —
- [ ] **3.3** ũ formulation — `DESIGN.md` §9.2 · note: (M3)
  <br>Acceptance: agrees with direct-u.
  <br>Measured: —
- [ ] **3.4** D_u^{1/3} layer scaling — `DESIGN.md` §9, §22 · note: (M3)
  <br>Acceptance: measured δ ∝ D_u^{1/3}; resolution requirements documented.
  <br>Measured: —

## Phase 4 — compatible magnetic kernel

- [ ] **4.1** de Rham space/order pairing — `DESIGN.md` §7.1 · note: §5
  <br>Measured: —
- [ ] **4.2** Gauge-fixed curl–curl — `DESIGN.md` §7.3, §11 · note: §5
  <br>Acceptance: manufactured magnetostatics; gauge null-space handled.
  <br>Measured: —
- [ ] **4.3** Harmonic flux field on an analytic torus — `DESIGN.md` §7.2 · note: §5
  <br>Measured: —
- [ ] **4.4** Divergence-free current projection — `DESIGN.md` §10 · note: (M2), (M4)
  <br>Acceptance: discrete ∇·B at roundoff; Ampère compatibility.
  <br>Measured: —

## Phase 5 — reduced end-to-end solver

- [ ] **5.1** Axisymmetric reduced model — `DESIGN.md` §16.3 · note: §11
  <br>Measured: —
- [ ] **5.2** Damped Picard — `DESIGN.md` §13.1, §13.3
  <br>Measured: —
- [ ] **5.3** Anderson with fallback — `DESIGN.md` §13.3
  <br>Measured: —
- [ ] **5.4** Staged continuation — `DESIGN.md` §14.4
  <br>**Phase gate.** Acceptance: axisymmetric benchmark vs. Grad–Shafranov + p₀(V(ψ)) within tolerance.
  <br>Measured: —

## Phase 6 — 3D fixed boundary

- [ ] **6.1** Periodic-torus end-to-end benchmark — `DESIGN.md` §16.2
- [ ] **6.2** Smooth solid-torus mesh — `DESIGN.md` §16.4 (simple torus → shaped Fourier boundary; geometry-error report)
- [ ] **6.3** VMEC/VMEC++ reader + initialization — `DESIGN.md` §17
- [ ] **6.4** DESC reader — `DESIGN.md` §17
- [ ] **6.5** Finite-β fixed-boundary stellarator example — `DESIGN.md` §19, §22 (nightly)
  <br>Acceptance: nested-surface case reproduces p = p₀(V(ψ)) to O(ε_κ); island case shows flattening with measured w_c ∝ ε_κ^{1/4}.

## Phase 7 — extreme-anisotropy upgrade *(may run parallel to Phase 8)*

- [ ] **7.1** AP prototype — `DESIGN.md` §8.4 · **ADR required before code**
- [ ] **7.2** Closed-field AP verification — `DESIGN.md` §8.3
- [ ] **7.3** AP as interchangeable χ solver in Picard — `DESIGN.md` §8.4

## Phase 8 — Newton

- [ ] **8.1** Side-effect-free residual refactor — `DESIGN.md` §14.2
- [ ] **8.2** JFNK prototype with Picard-block preconditioning — `DESIGN.md` §14.3
- [ ] **8.3** Exact local linearization — `DESIGN.md` §14.3
- [ ] **8.4** Nonlocal low-rank V_χ JVP — `DESIGN.md` §12.6, §14.3
- [ ] **8.5** Pseudo-transient globalization and switchover — `DESIGN.md` §14.4
  <br>Acceptance: Picard/Newton agreement test.

## Phase 9 — PETSc experiment *(separate branch; never blocks releases)*

- [ ] **9** Benchmark and ADR — `DESIGN.md` §3.2, §21

## Phase 10 — later physics

- [ ] **10.1** Braginskii transport fields — `DESIGN.md` §15
- [ ] **10.2** Wall-bounded meshes — `DESIGN.md` §16.5
- [ ] **10.3** Free-boundary vacuum coupling
- [ ] **10.4** Open-field/sheath model class · **design ADR with physics sign-off before code**
- [ ] **10.5** Tangent/adjoint sensitivities
- [ ] **10.6** Two-temperature energy equations
- [ ] **10.7** MPI, only if demonstrated necessary

---

## Release gates

- **0.1** — Phases 0–5 complete. See `DESIGN.md` §25 "Release definitions".
- **0.2** — Phase 6 + first AP solver (7.1–7.3) + Anderson.
- **0.3** — Phase 8 + Phase 9 results + shaped wall domains + initial sensitivities.

## Open ADRs blocking work

_(none yet — agents add rows here when they draft one)_

| ADR | Milestone blocked | Question | Status |
|---|---|---|---|
