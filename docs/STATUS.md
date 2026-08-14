# STATUS — milestone ledger

Single source of truth for what is done and what is next. Agents read this first and
update it as the last step of every milestone. Authoritative milestone definitions and
acceptance criteria live in `DESIGN.md` §25; this file tracks state and measurements.

Legend: `[ ]` not started · `[~]` in progress (not ready for review) · `[x]` complete in
the submitted PR (or merged), with its required checks green.  The milestone's `[x]`
change is part of its implementation PR; merging that PR requires no follow-up ledger
edit.

A milestone may only start when every milestone in the previous phase is `[x]` on the
target integration branch (`DESIGN.md` §25). A `[x]` on an unmerged PR does not satisfy
that dependency. Phase 7 may run in parallel with Phase 8.

---

## Phase 0 — repository and conventions

- [x] **0.1** Package skeleton — `DESIGN.md` §20, §23 · note: n/a
  <br>Acceptance: `pip install -e .` + `pytest` in a clean env, no compiler, no MPI, on Linux and macOS.
  <br>Measured: local macOS / CPython 3.12.2 / NGSolve 6.2.2606 — `make check`,
  `make smoke`, and `make wheel-smoke` pass. Additional review hardening is in progress
  before the PR is merged.
  <br>Next: use the project `.venv` (isolated CPython 3.12); the supplied Conda Python
  interpreters segfault during pytest collection in this terminal environment.
- [x] **0.2** Common utilities — `DESIGN.md` §6, §21, §24 · note: n/a (normalization is `DESIGN.md` §6)
  <br>Acceptance: block norms, structured logging, timing context, deterministic config serialization, thread config, checkpoint metadata; round-trip tested.
  <br>Measured: 11 Phase-0.2 contract tests pass: physically scaled block norms;
  canonical configuration serialization/digest and rejection guards; JSON event/timing
  records; validated NGSolve thread configuration; and byte-identical, versioned
  checkpoint-metadata round-trip. Awaiting PR merge and CI.
  <br>Next: provide the human-console log adapter at a solver run entry point; this
  milestone supplies the machine-JSON event stream only.

## Phase 1 — anisotropic scalar kernel

- [x] **1.1** Isotropic Poisson on `Slab2D` — `DESIGN.md` §8.1, §16.2 · note: §4
  <br>Acceptance: manufactured convergence at the expected order in L² and energy norms.
  <br>Measured: merged PR #6; macOS / CPython 3.12.2 / NGSolve 6.2.2606 — finest-pair degree 1:
  L² 1.955, energy 0.981; degree 2: L² 2.992, energy 1.978; degree 3: L² 4.053,
  energy 3.005 (72 → 288 elements; deterministic structured triangles).
  See `tests/manufactured/isotropic_poisson_rates.csv` and `docs/verification.md`.
  <br>Next: milestone 1.2 extends this named-boundary slab to oblique anisotropy.
- [x] **1.2** Oblique anisotropic K — `DESIGN.md` §8.1, §8.2 · note: §4
  <br>Acceptance: parallel/perpendicular diagnostics; order scans.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — constant oblique
  `K = 2I + 5bbᵀ`, `b = (3/5, 4/5)`; finest-pair degree 1: L² 1.887, K-energy
  0.965; degree 2: L² 3.054, K-energy 1.968; degree 3: L² 4.089, K-energy 3.008
  (72 → 288 elements). The solver separately reports positive parallel and
  perpendicular M4a energies. See `tests/manufactured/oblique_anisotropic_rates.csv`
  and `docs/verification.md`.
  <br>Next: the verification kernel remains internal; 1.5 owns the public
  `AnisotropicDiffusionSolver` interface extraction.
- [x] **1.3** Pollution benchmark — `DESIGN.md` §8.3 · note: §4
  <br>**Phase gate.** Acceptance: machine-readable table; measured pollution decreases systematically with order *and* refinement. No coupled work proceeds until this holds.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — with
  κ∥ = 1 and κ⊥ = 0, κ⊥,num/κ∥ decreases on 32 → 128 → 512 elements from
  7.351e-2 → 1.967e-2 → 5.002e-3 (p=1), 1.627e-3 → 1.101e-4 → 7.193e-6
  (p=2), and 8.805e-6 → 1.622e-7 → 2.761e-9 (p=3). Finest-pair rates are
  1.975, 3.936, and 5.877. See `tests/manufactured/sovinec_pollution.csv` and
  `docs/verification.md`.
  <br>Next: milestone 1.4 can reuse the translated Sovinec field
  b = (∂yψ, −∂xψ)/|∇ψ| and should add an independent analytic-island field rather
  than treating this pollution regression as the full closed-field/island suite.
- [x] **1.4** Closed-field and island frozen-field tests — `DESIGN.md` §8.3, §22 · note: §4, §7
  <br>Acceptance: finite-anisotropy closed-field response; independent analytic-island
  manufactured convergence in L² and K-energy norms; smooth field-null handling.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — island finest-pair degree 1:
  L² 1.810, K-energy 0.964; degree 2: L² 3.132, K-energy 1.983; degree 3: L² 4.053,
  K-energy 2.996 (200 → 800 elements). In the degree-3, 512-element closed-field scan,
  κ⊥,num/κ⊥ is 4.8e-6, 1.2e-5, and 6.4e-5 for κ⊥/κ∥ = 1e-1, 1e-2, and 1e-3.
  See `tests/manufactured/analytic_island_rates.csv`,
  `tests/manufactured/closed_field_anisotropy_scan.csv`, and `docs/verification.md`.
  <br>Next: milestone 1.5 should route the constant, rank-one Sovinec, and new smoothly
  floored spatial-field assemblies through one interface without changing their tables;
  preserve the M4a tensor form `K = κ⊥I + (κ∥-κ⊥)b_safe b_safeᵀ` at field nulls,
  add a direct `|b_safe| < 1` symmetry/eigenpair unit contract, and add the §6
  floor-sensitivity study that distinguishes deliberate manufactured-floor activity
  from acceptable production observables.
- [x] **1.5** Refactor to `AnisotropicDiffusionSolver` — `DESIGN.md` §8.4
  <br>Acceptance: interface extracted without changing recorded results.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — public StandardCG
  routes constant, smoothly floored spatial, and rank-one Sovinec M4a paths.
  ADR 0002 records the removal of a redundant normalization; all nine rank-one
  rows match `origin/main` bit-for-bit. The constant direct-tensor path differs only at
  roundoff level (solution-vector maximum below 1e-14) while its CSV remains unchanged.
  Its order/refinement trends remain strict. The direct
  inverse-preconditioner identity defect is below 1e-11. The default pollution gate requires
  κ⊥,num/κ⊥ < 0.1, warning by default and raising `AnisotropyPollutionError` in
  strict mode; κ⊥ = 0 is correctly unsafe. The B-floor observable-sensitivity gate
  defaults to 1% and catches a 100% change at an O(1e-3) observable.
  <br>Next: Phase 2 starts the mollified V_χ operator; keep this solver's M4a
  diagnostics and safety gates as the reusable frozen-field interface. The direct
  `|b_safe| < 1` tensor symmetry/eigenpair contract and paired solver floor-sensitivity
  study remain Phase-1 verification follow-ups before relying on this interface in 3D.

## Phase 2 — level-set volume and transplant

- [x] **2.1** Mollified V_χ — `DESIGN.md` §12.1, §12.2 · note: §6
  <br>Acceptance: analytic circle/sphere; monotone tabulation.
  <br>Measured: macOS / CPython 3.12.2 / NumPy 2.4.6 — the zero-level circle and
  sphere volumes have relative errors 5.18e-4 and 7.76e-4, respectively; their
  mollified co-area densities agree within 1.70e-2 and 2.56e-3. The sphere
  quadrature-resolution scan 24 → 48 → 96 measures adjacent rates 2.078 and
  2.058. The endpoint identities hold exactly, raw endpoint residuals are checked,
  and the 65-level table is strictly monotone with uniform enclosed-volume samples. See
  `tests/manufactured/mollified_sphere_volume_rates.csv` and `docs/verification.md`.
  <br>Next: 2.2 should add the FEM quadrature-extraction pass and consume
  `MollifiedVolumeMap` through a monotone `VolumeProfile` transplant; 2.3 owns the
  nonlocal JVP and must calibrate the `minimum_gradient_fraction=1e-3` critical-level
  safeguard against tabulation spacing before using its mollified derivatives in Newton.
- [x] **2.2** Profiles + transplant — `DESIGN.md` §12.5 · note: §6
  <br>Acceptance: exact enclosed-volume and layer-cake tests.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — direct superlevel-volume
  measurement of the transplanted field realizes seven targets with a 9.77e-5 maximum
  absolute error (required below 2.0e-3); all eight compact quadratic-B-spline layer-cake residuals are below 9.1e-5
  (required below 3.0e-3). The NGSolve mapped-quadrature pass integrates the unit square to 1.0
  and its M4b BSpline transplant has 0.5 ± 0.03 mean pressure.
  <br>Next: 2.3 owns the nonlocal ``delta V_chi`` JVP; the current BSpline represents
  only M4b's local composition and NGSolve differentiates it symbolically.
- [x] **2.3** Differentiable map — `DESIGN.md` §12.6 · note: §6
  <br>Acceptance: JVP agrees with finite differences.
  <br>Measured: macOS / CPython 3.12.2 / NumPy 2.4.6 — the ADR-0003 frozen-width
  quasi-Newton `(V_derivatives)` surface-average JVP agrees with its central finite
  difference at four manufactured levels with maximum absolute error 4.51e-11
  (step 1e-6). A variable-gradient live-width rebuild differs by 1.24e-4 relative,
  below the 2.0e-4 regression bound. The JVP is accumulated level-by-level in O(N+M)
  memory and uses direct H′εwᵢ quadrature weights, never a PCHIP derivative.
  <br>Next: 2.4 remains optional; later Newton work can consume this array-backed
  low-rank quasi-Newton action while retaining symbolic differentiation for M4b's
  local term. Revisit ADR 0003 Option 2 if nonlinear convergence is poor.
- [x] **2.4** Cut-cell reference (optional) — `DESIGN.md` §12.4 · note: §6 (M4b), §8.1
  <br>Acceptance: optional high-order implicit-domain reference agrees with analytic sharp
  circle volumes and calibrates the mollified map under refinement.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 / xfem 2.1.2606 —
  geometry-order-3 sharp-circle errors are 4.752e-6, 2.971e-7, and 2.390e-8 on
  128, 512, and 2048 triangles, with adjacent rates 3.999 and 3.636. The
  mollified-to-sharp differences are 3.202e-3, 8.016e-4, and 2.004e-4 with
  adjacent rates 1.998 and 2.000. See
  `tests/manufactured/cutcell_circle_rates.csv` and `docs/verification.md`.
  <br>Next: `CutCellVolumeReference` is deliberately a direct sharp-volume evaluator
  (`volume` plus `total_volume`), not the solver-facing differentiable map; retain
  `MollifiedVolumeMap` for inverse tabulation and the `(V_derivatives)` Newton action.

## Phase 3 — M3 kernel

- [x] **3.1** Direct-u weak form, frozen (B, p) — `DESIGN.md` §9.1, §9.4 · note: (M3), §5.5
  <br>Acceptance: both regularization gradients (∇⊥ default, full ∇ isotropic variant)
  selectable at runtime and applied consistently to the (M2) flux, the (M3) weak form,
  and the final `D_u ∇ᵣu·∇p` term; the choice recorded in config digest, logs, and
  checkpoint metadata; with the full-∇ variant, diagnostics report
  J∥/B = u − (D_u/B)b·∇u.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — a strong-form
  manufactured solution transcribed directly from (M3) has L² error 1.216e-6
  for both variants; algebraic free-DOF relative residuals are 3.19e-17
  and 2.12e-17. The frozen benchmark is exactly divergence-free, and its nonzero
  drive/reaction/final-correction L² norms are 4.246e-1, 1.47–1.61e-2, and
  1.232–1.239e-2. Pointwise M2 reconstruction and full-∇ J∥/B agree with independent
  formulas to 1e-12; the runtime choice round-trips through digest, structured logs,
  and checkpoint metadata. A divergence identity pins the M3 drive's `2/B_safe³`
  coefficient to 3.6e-16, and its certified L² norm agrees with the assembled
  diagnostic to 1e-12. B-floor activity is 1.70e-16 with sampled min |B| = 2.236;
  setting the floor to 1.0 raises the live diagnostic to 1.241e-1.
  <br>Next: milestone 3.2 should add centralized SUPG with a complete strong residual
  for both variants. Retain the strong-form oracle: applying the same wrong drive factor
  and reaction sign to the implementation and weak assembly check raises its L² errors
  to 1.287 (∇⊥) / 1.284 (full ∇), while deleting the final M3 correction from the
  implementation also makes the algebraic assembly check fail. The divergence identity
  rejects `2→3` and `B_safe³→B_safe²` oracle mutations by 8.32e-2 and 2.81e-1;
  coordinated implementation/assembly/oracle mutations shift the assembled drive norm
  by +50% and +163%, so the certified-to-assembled comparison catches both.
  Before evolving-field studies, pair the floor-activity measurement with an
  observable-sensitivity warning/error gate.
- [x] **3.2** SUPG + manufactured tests — `DESIGN.md` §9.1, §9.4 · note: (M3), §5.5
  <br>Acceptance: includes the test that fails conspicuously if the `D_u ∇ᵣu·∇p` term is dropped (`DESIGN.md` §22); all manufactured tests run for both gradient variants.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — complete-residual SUPG
  finest-pair L² rates for degrees 1/2/3 are 1.970 / 3.029 / 4.036 with ∇⊥ and
  1.968 / 3.039 / 4.033 with full ∇; every degree clears the standard `p + 0.8`
  gate. P1 has zero element-interior Hessian, so the full-∇ strong diffusion residual
  vanishes and ∇⊥ retains only projector derivatives; the bounded stabilization still
  retains optimal L² rate.
  Aligned-advection SUPG on/off and transverse-
  diffusion manufactured cases pass for both variants. In the dedicated final-term
  case, the correct L² errors are 1.234e-6 / 1.234e-6 (∇⊥ / full ∇); deleting
  `D_u ∇ᵣu·∇p` raises them to 4.217e-2 / 4.230e-2. See
  `tests/manufactured/m3_supg_rates.csv` and `docs/verification.md`.
  <br>Next: milestone 3.3 should apply the direct-u/ũ transformation to the same
  centralized SUPG residual and retain both `stabilization` and gradient-variant
  provenance paths.
- [x] **3.3** ũ formulation — `DESIGN.md` §9.2, §9.4 · note: (M3), §5.5
  <br>Acceptance: agrees with direct-u; transformed source terms use the selected ∇ᵣ; verified for both variants.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — reconstructed-ũ and
  direct-u relative L² disagreements are at most 6.26e-16 (∇⊥) and 8.64e-15
  (full ∇), below the 1e-10 gate. Finest-pair degree 1/2/3 L² rates are
  1.970 / 3.030 / 4.036 (∇⊥) and 1.968 / 3.040 / 4.033 (full ∇); all clear p+0.8.
  An independent quadratic-pressure transcription measures distinct nonzero profile-
  diffusion source norms 2.012e-1 (∇⊥) and 2.400e-1 (full ∇), and verifies every
  shifted source to 1e-12 relative tolerance. See
  `tests/manufactured/m3_utilde_rates.csv` and `docs/verification.md`.
  <br>Next: milestone 3.4 should use the preferred homogeneous ũ solve while retaining
  the direct-u cross-check, both gradient variants, complete shifted SUPG residual, and
  physical-u/M2 reconstruction when measuring the D_u^{1/3} layer width.
- [ ] **3.4** D_u^{1/3} layer scaling — `DESIGN.md` §9, §22 · note: (M3), §5.5
  <br>Acceptance: measured δ ∝ D_u^{1/3} for both gradient variants; resolution requirements documented.
  <br>Measured: —
- [ ] **3.5** Gradient-variant comparison study (∇⊥ vs full ∇) — `DESIGN.md` §9.4 · note: §5.5
  <br>Acceptance: on shared frozen-(B, p) benchmarks including a resonant layer and a
  field-misaligned mesh: measured O(ε_J) relative cross-variant agreement at fixed D_u
  and a common D_u → 0 limit; machine-readable comparison tables for assembly/
  refactorization reuse, linear-iteration counts, monotonicity/oscillations and layer
  smearing, misalignment sensitivity, and parallel grid-noise damping, recorded in
  `docs/verification.md`. Default remains ∇⊥ unless changed by an ADR citing these
  measurements.
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
| 0003 | 2.3 | Must the M4b mollifier-width JVP differentiate `epsilon = c h |grad chi|`? | Option 1 accepted |
