# STATUS — milestone ledger

Single source of truth for what is done and what is next. Agents read this first and
update it as the last step of every milestone. Authoritative milestone definitions and
acceptance criteria live in `DESIGN.md` §25; this file tracks state and measurements.

Legend: `[ ]` not started · `[~]` in progress (not ready for review) · `[x]` complete in
the submitted PR (or merged), with its required checks green.  The milestone's `[x]`
change is part of its implementation PR; merging that PR requires no follow-up ledger
edit.

> **2026-08-15 model revision — read before choosing work.** The August 14
> `u=F(p)+ũ` current-profile prescription is mathematically ineffective at finite
> `D_u` and is superseded by note §5.4 and `DESIGN.md` §9.2. Completed milestones
> 3.1–3.4 remain valid verification of the unconstrained/G≡0 M3 kernel, SUPG, and local
> layer scaling. Milestone 3.3's F-shift is retained only as the negative control showing
> that changing F does not change physical u; it is not current-profile support. The next
> work is 3.5 (normalized p₀(s)/I₀(s) contracts and shell moments), then 3.6 (unknown-G
> bordered M3–M3b). Do not begin the gradient comparison, now 3.7, or any coupled phase
> until both corrective milestones pass. The authoritative note is
> `docs/20260815-01_Regularized_3D_MHD_equilibrium.tex`; the August 14 note is historical.

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
- [x] **1.4** Closed-field and island frozen-field tests — `DESIGN.md` §8.3, §22 · note: §4.3
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

- [x] **2.1** Mollified V_χ — `DESIGN.md` §12.1, §12.2 · note: §8.2, §8.4
  <br>Acceptance: analytic circle/sphere; monotone tabulation.
  <br>Measured: macOS / CPython 3.12.2 / NumPy 2.4.6 — the zero-level circle and
  sphere volumes have relative errors 5.18e-4 and 7.76e-4, respectively; their
  mollified co-area densities agree within 1.70e-2 and 2.56e-3. The sphere
  quadrature-resolution scan 24 → 48 → 96 measures adjacent rates 2.078 and
  2.058. The endpoint identities hold exactly, raw endpoint residuals are checked,
  and the 65-level table is strictly monotone with uniform enclosed-volume samples. See
  `tests/manufactured/mollified_sphere_volume_rates.csv` and `docs/verification.md`.
  <br>Next: 2.2 added the FEM quadrature-extraction pass and consumed
  `MollifiedVolumeMap` through a monotone `VolumeProfile` transplant; that legacy
  profile takes dimensional V, so milestone 3.5 owns its migration to normalized s.
  2.3 owns the
  nonlocal JVP and must calibrate the `minimum_gradient_fraction=1e-3` critical-level
  safeguard against tabulation spacing before using its mollified derivatives in Newton.
- [x] **2.2** Pressure profile + transplant (legacy dimensional-V API) — `DESIGN.md` §12.5 · note: §6.1, §8.3
  <br>Acceptance: exact enclosed-volume and layer-cake tests.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — direct superlevel-volume
  measurement of the transplanted field realizes seven targets with a 9.77e-5 maximum
  absolute error (required below 2.0e-3); all eight compact quadratic-B-spline layer-cake residuals are below 9.1e-5
  (required below 3.0e-3). The NGSolve mapped-quadrature pass integrates the unit square to 1.0
  and its M4b BSpline transplant has 0.5 ± 0.03 mean pressure.
  <br>Next: 2.3 owns the nonlocal ``delta V_chi`` JVP; the current BSpline represents
  only M4b's local composition and NGSolve differentiates it symbolically. The measured
  transplant remains valid, but its public `VolumeProfile` coordinate is dimensional V
  and MUST be migrated—not silently reinterpreted—in milestone 3.5.
- [x] **2.3** Differentiable map — `DESIGN.md` §12.6 · note: §8.2, §9.1
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
- [x] **2.4** Cut-cell reference (optional) — `DESIGN.md` §12.4 · note: §8.1
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

- [x] **3.1** Direct-u weak form, frozen (B, p) (revised-model G≡0 kernel) — `DESIGN.md` §9.1, §9.4 · note: §5.2, §5.5
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
  <br>Revision note: this verified the unconstrained direct-u equation and remains the
  G≡0 oracle for corrected (M3); it did not implement (M3b) or a current-profile input.
  Milestone 3.6 will reuse it when checking the constrained formulation. Retain the
  strong-form oracle: applying the same wrong drive factor
  and reaction sign to the implementation and weak assembly check raises its L² errors
  to 1.287 (∇⊥) / 1.284 (full ∇), while deleting the final M3 correction from the
  implementation also makes the algebraic assembly check fail. The divergence identity
  rejects `2→3` and `B_safe³→B_safe²` oracle mutations by 8.32e-2 and 2.81e-1;
  coordinated implementation/assembly/oracle mutations shift the assembled drive norm
  by +50% and +163%, so the certified-to-assembled comparison catches both.
  Before evolving-field studies, pair the floor-activity measurement with an
  observable-sensitivity warning/error gate.
- [x] **3.2** SUPG + manufactured tests (revised-model G≡0 kernel) — `DESIGN.md` §9.1, §9.4 · note: §5.2, §5.5
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
  <br>Revision note: retain the centralized complete strong residual and both provenance
  paths. Milestone 3.6 must add the −G′B·∇s term and apply every D_u term to ũ.
- [x] **3.3** Algebraic F(p)-shift equivalence (historical negative control; not a current closure) — `DESIGN.md` §9.2, §9.4 · note: §5.4 "A change of variables does not restore the freedom"
  <br>Historical acceptance: the shifted solve agrees with direct-u; transformed source
  terms use the selected ∇ᵣ; verified for both variants. Revised interpretation: this
  equivalence is evidence that prescribed F cancels from physical u, not that it imposes
  a mean current.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — reconstructed-ũ and
  direct-u relative L² disagreements are at most 6.26e-16 (∇⊥) and 8.64e-15
  (full ∇), below the 1e-10 gate. Finest-pair degree 1/2/3 L² rates are
  1.970 / 3.030 / 4.036 (∇⊥) and 1.968 / 3.040 / 4.033 (full ∇); all clear p+0.8.
  An independent quadratic-pressure transcription measures distinct nonzero profile-
  diffusion source norms 2.012e-1 (∇⊥) and 2.400e-1 (full ∇), independently verifies
  the other three shifted sources to 1e-12 relative tolerance, and pins diffusion by
  direct-u agreement. See
  `tests/manufactured/m3_utilde_rates.csv` and `docs/verification.md`.
  Direct-u/ũ agreement remains below 3.0e-16 with deliberately active B floors of
  0.1 and 1.0 after matching the symmetric Galerkin projection convention exactly.
  <br>Revision note: keep this path private/legacy until 3.6 uses it for the required
  two-F cancellation test. `PrescribedCurrentProfile` and its identifier MUST be
  removed or explicitly deprecated from the production API/checkpoint schema in 3.5;
  the corrected inputs are p₀(s) and I₀(s), while G(s) is solved.
- [x] **3.4** D_u^{1/3} layer scaling (G≡0 local-layer limit) — `DESIGN.md` §9, §22 · note: §5.3–§5.5
  <br>Acceptance: measured δ ∝ D_u^{1/3} for both gradient variants; resolution requirements documented.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — fitted physical-(M2)
  `J_parallel/B` FWHM exponents are 0.346160 (∇⊥) and 0.345392 (full ∇), within 0.04
  of 1/3. Across D_u = 0.0025 → 0.04, FWHM widths grow from 0.129309 → 0.337937
  and 0.129302 → 0.337151, respectively. The lowest adjacent exponents, 0.338771
  and 0.338644, are closer to 1/3 than the highest pair (0.357415 and 0.355391),
  documenting the remaining systematic residual as finite-D_u behavior. This milestone
  claims only the width half of the note's two-part `layer_width` scaling. Operational
  FWHM is 4.772–4.949 times the unit-prefactor inner scale `(D_u/(40π))^(1/3)` and
  spans 8.275–21.628 normal elements (required at least 6). At the thinnest case, a
  64×16 → 96×24 h-refinement changes FWHM by only 5.789e-5 relative. The M3 space is
  genuinely periodic, with direct-u/utilde seam values agreeing to 1e-11. Free-DOF
  relative residuals stay below 4.36e-17; physical-u, M2-current, and J_parallel/B
  cross-checks pass. Legacy prescribed-F identifiers participate in the configuration
  digest and structured solve records; this provenance is historical and is replaced by
  normalized I₀(s), G-basis, and shell-grid provenance in milestones 3.5–3.6. See `tests/manufactured/m3_layer_scaling.csv`,
  `tests/manufactured/m3_layer_mesh_refinement.csv`, and `docs/verification.md`.
  <br>Next: milestone 3.5 is the normalization/shell-moment correction. Milestone 3.7 may
  later reuse this periodic machinery and physical-M2 Fourier observable as a scaling
  baseline, but not this slab as evidence for cross-variant agreement: B_x=0 makes the
  layer-normal ∇⊥ and full-∇ operators identical. A later coupled production driver owns
  invoking `assess_layer_resolution` once it can estimate δ and local normal mesh scale.
- [x] **3.5** Normalized profile contracts + shell-current moments — `DESIGN.md` §12.1–§12.5, §9.2, §24 · note: §5.4, §6.1, §8
  <br>Acceptance: replace the dimensional-V `VolumeProfile` public/checkpoint contract
  with p₀(s) on exactly [0,1]; add cumulative `ToroidalCurrentProfile` I₀(s) with
  I₀(0)=0; reject ambiguous/dimensional coordinates; update the serialization contract
  and increment only a schema that already persists the superseded semantics;
  use one shared s=V_χ/V_Ω field for both profiles. Implement cumulative and shellwise
  I_tor integrals with mollified layer-set weights for all (M2) current components and
  verify against analytic circle/annulus or toroidal-surrogate integrals, h/quadrature
  refinement, endpoint/partition identities, and a domain-volume rescaling test showing
  identical p₀(s)/I₀(s) semantics. Legacy prescribed-F state is rejected or explicitly
  migrated only as non-production verification provenance.
  <br>Measured: macOS / CPython 3.12.2 / NumPy 2.4.6 — public analytic and
  piecewise-linear pressure/current profiles now use only normalized s∈[0,1]; pressure
  is non-increasing, cumulative I₀ enforces I₀(0)=0 while allowing reversal, and
  checkpoint profile records require `coordinate_kind="normalized_volume"`. The
  schema remains 1 because its earlier metadata envelope persisted no profile payload;
  ambiguous dimensional-V and legacy prescribed-F records are rejected. The shared
  `s=V_χ/V_Ω` evaluator drives both M4b pressure transplantation and independent
  mollified M3b shell moments. On the circular toroidal surrogate, total-current
  maximum cumulative errors are 2.683e-4 → 6.668e-5 → 1.665e-5 for 24 → 48 → 96
  radial cells at quadrature order 6, with adjacent rates 2.008 and 2.002. At fixed
  48 cells, quadrature orders 1 → 2 → 3 reduce the error 5.599e-4 → 1.578e-4 →
  6.173e-5. At 96 cells, component errors are 7.801e-5 (parallel), 3.899e-5
  (diamagnetic), and 1.949e-5 (regularizing). The endpoint, component-sum, and shell-
  partition identities are imposed by construction; independent analytic cumulative
  and shellwise values are tested. Radius 1 → 2.75 domain rescaling changes the
  scaled I₀(s) result by 3.55e-15 and sampled s by 8.88e-16. See
  `tests/manufactured/shell_current_moment_rates.csv` and `docs/verification.md`.
  <br>Next: milestone 3.6 should pass its independently reconstructed physical M2
  component samples to `mollified_shell_current_moments`, reuse the exact shared s
  field, and compare those diagnostic rows—not its C_u/C_G matrices—with the input
  `ToroidalCurrentProfile`. Its shell grid must satisfy both enforced local resolution
  checks: at least three radial cells and at least two mapped mollifier widths per
  shell. The deprecated F-shift path is deliberately absent
  from `remec.solvers` exports but remains in its implementation module for the required
  two-F cancellation negative control.
- [x] **3.6** Constrained unknown-G bordered M3–M3b solve — `DESIGN.md` §9.1–§9.4 · note: §5.4, (M2)–(M3b), §9
  <br>Acceptance: for both ∇⊥ and full-∇ variants, solve jointly for homogeneous ũ and
  G(s) with G(1)=u_b; apply −D_u∇ᵣũ consistently in (M2), (M3), SUPG, diagnostics, and
  shell constraints; realize I_tor(s)=I₀(s) to solver tolerance, confirmed by an
  independent shell-integral evaluation of the reconstructed (M2) current (not by reusing
  the solve's constraint matrices). Automated tests MUST
  show (a) two distinct old F(p) profiles with the same edge value give the same physical
  u, (b) two distinct I₀(s) profiles give their respective currents, and (c) deleting
  −G′B·∇s, dropping the −(μ₀G/B²)B·∇p reaction coupling, diffusing full u, or omitting
  any current contribution makes a test fail.
  Include h/p/N-shell convergence, N-doubling stability, the bordered Schur solve, and a
  D_u scan holding I₀ fixed while the multiplier-current/mean-ũ diagnostics approach the
  note's regular limit.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — the unknown-G
  piecewise-linear border uses one A factorization plus a shell-sized Schur solve and
  independently reconstructs all parallel, diamagnetic, and regularizing (M2) moments.
  Across the coupled h/p table, maximum relative residuals are 1.131e-16 (M3) and
  1.063e-16 (M3b). At p=2, 20 → 28 subdivisions gives physical-u L² rates 2.0041
  (∇⊥) and 2.0042 (full ∇); the p=1 → 2 errors decrease from 3.379e-4 → 2.016e-4
  and 3.869e-4 → 2.054e-4, with the p=3 results at the second-order mollified-shell
  ceiling. Doubling 4 → 8 shells changes the physical field by 2.145e-4 (∇⊥) and
  2.176e-4 (full ∇) in relative L² while every finest shell spans 3.991 local
  cells/mollifier widths. Two distinct I₀ profiles on the nondegenerate coupled state
  realize their independently evaluated currents below 1e-10; both G couplings and a
  nonzero ũ are exercised. Two old F profiles with one edge value cancel below 1e-10.
  In a fixed-I₀ family with D_u-dependent bounded G′, D_u=0.08 → 0.04 → 0.02 reduces
  ‖D_uG′∇ᵣs‖₂ from 2.850e-2 → 1.361e-2 → 6.750e-3 (∇⊥) and 2.874e-2 →
  1.372e-2 → 6.804e-3 (full ∇); maximum |⟨ũ⟩| falls from 3.987e-2 to 9.379e-3.
  The constrained path reports physical J∥/B, including the full-gradient ũ correction.
  See `tests/manufactured/m3_constrained_rates.csv`,
  `tests/manufactured/m3_constrained_du_scan.csv`, and `docs/verification.md`.
  <br>Next: milestone 3.7 should retain this exact shared-s/PCHIP construction and
  independent current evaluator, use genuinely field-misaligned/resonant benchmarks
  and record reuse/iteration, oscillation, smearing, misalignment, and parallel-noise
  metrics without changing the default ∇⊥ choice absent an ADR. The p=2 → 3 plateau is
  the inherited second-order mollified-shell ceiling; the §12.4 cut-cell reference is
  the route for deciding whether sharper shell integration removes it.
- [x] **3.7** Gradient-variant comparison study (∇⊥ vs full ∇), constrained formulation — `DESIGN.md` §9.4 · note: §5.5
  <br>Acceptance: on shared frozen-(B,p,s,I₀) benchmarks including a resonant layer and a
  field-misaligned mesh: each variant realizes the same I₀(s); measured O(ε_J) relative
  cross-variant agreement at fixed D_u and a common D_u→0 limit; machine-readable
  comparison tables for assembly/refactorization reuse, bordered linear-iteration
  counts, monotonicity/oscillations and layer smearing, misalignment sensitivity, and
  parallel grid-noise damping, recorded in `docs/verification.md`. Default remains ∇⊥
  unless changed by an ADR citing these measurements.
  The D_u→0 study must be emergent on one fixed frozen state and fixed I₀/drive; monitor
  ‖D_uG′∇ᵣs‖ and ⟨ũ⟩_s, and explicitly report or reject an inadmissible target rather
  than inferring regularity from a per-D_u manufactured family.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — on one fixed
  resonant `(B,p,s,I₀,drive)` state, D_u = 0.04 → 0.02 → 0.01 reduces the relative
  physical-u cross-variant difference from 2.0207e-2 → 1.0489e-2 → 5.1913e-3,
  with difference/ε_J = 1.0104 / 1.0489 / 1.0383 and adjacent decay rates 0.946 /
  1.015. A fixed B_x=0.01 makes the bordered P block nonzero. This fixed-εκ target is
  explicitly rejected as a strict admissible D_u→0 sequence: εκ/ε_J = 0.25 / 0.50 /
  1.00, maximum |⟨ũ⟩| grows 8.74e-3 → 3.539e-2, and
  D_u max|⟨ũ⟩| stays within 1.02 at about 3.5e-4. Thus the scan establishes the
  cross-variant O(ε_J) gap, not a regular common physical limit; milestone 3.6 records
  the vanishing-mean admissible-family check. Both variants independently realize the
  same I₀ below 2.99e-17 relative, while the multiplier-current norm falls from about
  1.55e-2 → 4.06e-3. The resonant P coupling is carried by the advection term
  (‖B·∇G‖₂ = 3.86e-3 → 4.06e-3); the misalignment table separately pins its nonzero
  reaction contribution. The nonuniform ∇φ makes the regularizing
  toroidal-current norm nonzero and variant-distinct (1.109e-2 / 9.955e-3 at
  D_u=0.02). At that D_u, the ∇⊥/full layer FWHM values are 0.41050 / 0.40702
  (9.85 / 9.77 normal cells), both have one radial turning point, and full ∇ lowers
  fifth-harmonic parallel-noise transfer by 4.49%; all scan rows clear six FWHM cells.
  On aligned and 22.5°-misaligned 20² → 28² controls,
  coarse-to-fine changes are 1.4068e-2 / 1.2687e-2 and 1.9625e-2 / 1.6215e-2,
  giving misalignment amplifications of 1.395 / 1.278. Each frozen solve records one
  A assembly/factorization, five direct responses, and four within-call factorization
  reuses; Krylov iterations are not applicable and the preconditioner is none.
  Across the resonant rows, full ∇ assembles A about 2.3 times faster; bordered-solve
  differences are below the run-to-run timing spread. Cross-iteration reuse is not
  implemented. The evidence does not justify changing the default from ∇⊥. See
  `tests/manufactured/m3_gradient_du_limit.csv`,
  `tests/manufactured/m3_gradient_misalignment.csv`, and `docs/verification.md`.
  <br>Next: Phase 4 starts the compatible de Rham magnetic kernel. Preserve the runtime
  gradient selection and its independently reconstructed physical (M2) current through
  the later H(div) projection; milestone 4.4 must retain these M3b shell moments rather
  than certifying only divergence. The actual Picard driver in Phase 5 should remeasure
  full-∇ regularization-block reuse before considering any default change.

## Phase 4 — compatible magnetic kernel

- [x] **4.1** de Rham space/order pairing — `DESIGN.md` §7.1 · note: §6 (M1)
  <br>Acceptance: on 6- and 48-tetrahedron contractible cubes at base orders 0--3,
  independently verify every grad/curl/div mapping and both successive-derivative
  identities below 1e-12, with exact Euler characteristic one; reject element families
  whose NGSolve order convention has not been established.
  <br>Measured: macOS / CPython 3.12.2 / NGSolve 6.2.2606 — the affine tetrahedral
  sequence is `H1(p+1) -> HCurl(p) -> HDiv(max(p-1,0)) -> L2(max(p-2,0))`.
  Across eight mesh/order rows the maximum individual mapping defect is 3.42e-15,
  maximum `curl(grad)` defect is 1.03e-13, and maximum `div(curl)` defect is 2.53e-14;
  every alternating global dimension is exactly one. See
  `tests/manufactured/de_rham_pairing.csv` and `docs/verification.md`.
  <br>Next: milestone 4.2 should reuse these tetrahedral offsets for its H1 gauge and
  HCurl vector-potential spaces. On curved tetrahedra measure the (M1) magnetic
  invariant directly as `div(curl(A_h))`: NGSolve's ordinary scalar L2 is a weak
  diagnostic/constraint space, not the Piola density-mapped strong image of `div(HDiv)`.
- [ ] **4.2** Gauge-fixed curl–curl — `DESIGN.md` §7.3, §11 · note: §6 (M1)
  <br>Acceptance: manufactured magnetostatics; gauge null-space handled.
  <br>Measured: —
- [ ] **4.3** Harmonic flux field on an analytic torus — `DESIGN.md` §7.2 · note: §6 (M1)
  <br>Measured: —
- [ ] **4.4** Divergence-free current projection — `DESIGN.md` §10 · note: (M1)–(M3b), §5.4
  <br>Acceptance: discrete ∇·B at roundoff; Ampère compatibility; projected current
  preserves the prescribed (M3b) shell moments I_tor(s)=I₀(s) to the stated tolerance.
  <br>Measured: —

## Phase 5 — reduced end-to-end solver

- [ ] **5.1** Axisymmetric reduced model — `DESIGN.md` §16.3 · note: §11, especially §11.2
  <br>Acceptance: reduced equations include both p₀(s) and I₀(s), and recover the note's
  Grad–Shafranov enclosed-current relation for at least two current profiles.
  <br>Measured: —
- [ ] **5.2** Damped Picard — `DESIGN.md` §13.1, §13.3 · note: §9
  <br>Acceptance: cycle uses one s field, p=p₀(s), bordered M3–M3b, and a
  shell-moment-preserving projection; convergence includes both profile residuals.
  <br>Measured: —
- [ ] **5.3** Anderson with fallback — `DESIGN.md` §13.3
  <br>Measured: —
- [ ] **5.4** Staged continuation — `DESIGN.md` §14.4 · note: §9, §11.2
  <br>**Phase gate.** Acceptance: axisymmetric benchmark vs. Grad–Shafranov with
  p=p₀(s(ψ)) and I_tor=I₀(s(ψ)) within tolerance for at least two I₀ targets.
  <br>Measured: —

## Phase 6 — 3D fixed boundary

- [ ] **6.1** Periodic-torus end-to-end benchmark — `DESIGN.md` §16.2 · note: §6, §9
- [ ] **6.2** Smooth solid-torus mesh — `DESIGN.md` §16.4 (simple torus → shaped Fourier boundary; geometry-error report)
- [ ] **6.3** Poincare plots: compute data via field line tracing, save and load
  data, functions to make plots.
- [ ] **6.4** VMEC/VMEC++ reader + initialization — `DESIGN.md` §17
  <br>Acceptance: imports p₀(s) and derives cumulative enclosed toroidal current I₀(s)
  on the same normalized-volume grid; does not pass through a legacy F(p) profile.
- [ ] **6.5** DESC reader — `DESIGN.md` §17
  <br>Acceptance: same normalized p₀(s)/I₀(s) contract as 6.4.
- [ ] **6.6** Finite-β fixed-boundary stellarator example — `DESIGN.md` §19, §22 (nightly)
  <br>Acceptance: nested-surface case reproduces p=p₀(s(ψ)) to O(ε_κ) and
  I_tor=I₀(s(ψ)) to constraint/discretization tolerance; island case shows flattening
  with measured w_c∝ε_κ^{1/4}; diagnostics include the current-profile residual.

## Phase 7 — extreme-anisotropy upgrade *(may run parallel to Phase 8)*

- [ ] **7.1** AP prototype — `DESIGN.md` §8.4 · **ADR required before code**
- [ ] **7.2** Closed-field AP verification — `DESIGN.md` §8.3
- [ ] **7.3** AP as interchangeable χ solver in Picard — `DESIGN.md` §8.4

## Phase 8 — Newton

- [ ] **8.1** Side-effect-free residual refactor — `DESIGN.md` §14.2 · note: §9.1
  <br>Acceptance: state includes G coefficients and residual includes every M3b shell row.
- [ ] **8.2** JFNK prototype with Picard-block preconditioning — `DESIGN.md` §14.3 · note: §9.1
- [ ] **8.3** Exact local linearization — `DESIGN.md` §14.3
- [ ] **8.4** Nonlocal low-rank level-set JVPs — `DESIGN.md` §12.6, §14.3 · note: §5.4, §9.1
  <br>Acceptance: JVP covers p₀(s), G(s), shell weights, and I_tor constraints as well
  as V_χ; agrees with finite differences for all blocks.
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

- **0.1** — Phases 0–5 complete, including corrected normalized p₀(s)/I₀(s) and
  constrained M3b. See `DESIGN.md` §25 "Release definitions".
- **0.2** — Phase 6 + first AP solver (7.1–7.3) + Anderson.
- **0.3** — Phase 8 + Phase 9 results + shaped wall domains + initial sensitivities.

## Open ADRs blocking work

_(none yet — agents add rows here when they draft one)_

| ADR | Milestone blocked | Question | Status |
|---|---|---|---|
| 0003 | 2.3 | Must the M4b mollifier-width JVP differentiate `epsilon = c h |grad chi|`? | Option 1 accepted |
