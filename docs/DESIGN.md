# remec — software design and implementation plan

**Status:** accepted initial architecture (synthesis of two independent design drafts; see Appendix A for adjudications)
**Primary backend:** Netgen/NGSolve
**Primary language:** Python
**Primary initial mode:** interpretive, fixed closed boundary
**Mathematical source of truth:** `docs/20260814-01_Regularized_3D_MHD_equilibrium.tex` (referred to throughout as **"the note"**; labels (M1)–(M4b) and section numbers refer to it)

---

## 1. Purpose and audiences

This document defines the software architecture, numerical strategy, implementation
sequence, and verification requirements for **remec**, a code solving the
transport-regularized 3D MHD equilibrium model (M1)–(M4b) of the note.

Audiences: (1) human developers; (2) reviewers checking that an implementation respects
the mathematical model; (3) AI coding agents implementing one small, tested milestone at
a time.

Authority: **the note is authoritative for the equations, physics, and asymptotic
scalings; this document is authoritative for software architecture and sequencing.** If
the two appear to conflict, do not silently choose one: open an issue or an architecture
decision record (ADR, in `docs/adr/`) quoting the relevant passages. Small NGSolve API
discrepancies discovered during implementation (an argument name, a missing solver
option) go in `docs/dev_notes.md`; anything that changes a decision in this document
requires an ADR.

The immediate goal is **not** a production stellarator code. It is a trustworthy sequence
of independently verified numerical kernels and reduced-dimensional solvers establishing:
accurate strongly anisotropic diffusion without field-aligned coordinates; a stable
discretization of the regularized current-continuity equation; a compatible magnetic
representation with discrete divergence at roundoff; accurate, differentiable evaluation
of the nonlocal level-set-volume map V_χ; a modular Picard iteration; a credible path to
coupled Newton; and `pip install remec` with no compiler and no MPI.

## 2. Normative language

**MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, **MAY** are used in the usual
requirements sense. A MUST is a non-negotiable invariant unless changed by an accepted
ADR. A SHOULD is the expected implementation; a well-documented alternative may be
accepted. A MAY is optional.

---

## 3. Executive decisions

### 3.1 Backend and language

remec **MUST** use: Python for the public API and orchestration; Netgen/NGSolve for
finite-element spaces, variational forms, meshes, compiled kernels, shared-memory
parallelism, and native linear algebra; NumPy/SciPy for small dense algebra,
interpolation, profiles, Anderson acceleration, and diagnostics; pytest for verification;
HDF5-compatible output for portable data plus NGSolve-native data where needed.

NGSolve was selected because binary wheels exist on PyPI for Linux, macOS, and Windows,
and useful shared-memory parallelism is available through `TaskManager` without MPI,
supporting the required installation path:

```bash
python -m pip install remec
```

The default installation **MUST NOT** require PETSc, MPI, a C++ compiler, Fortran, or a
system package manager. Heavy packages (JAX, DESC, SIMSOPT, PETSc) **MUST NOT** be base
dependencies; they are optional extras.

### 3.2 Optional PETSc experiment (deferred by design)

After the native NGSolve implementation has useful end-to-end 3D functionality, an
experimental branch (`petsc-backend`) MAY explore `ngsPETSc`: KSP/SNES, field-split
preconditioning, hypre AMS for the curl–curl block, and one-node scaling. PETSc
**MUST NOT** become a default dependency without benchmark evidence of a substantial
robustness or performance benefit that justifies the environment complexity. Merge
criteria SHOULD include at least one of: robustness on a problem native NGSolve cannot
solve; ≥ ~2× time-to-solution at representative size; access to otherwise-inaccessible
problem sizes; or a major simplification of the Newton implementation — weighed together
with installation, wheel availability, platform support, and CI burden. The native and
PETSc paths MUST share the same residual definitions, state objects, subproblem
interfaces, verification tests, and checkpoint format (see the `solvers/linalg/`
abstraction in Section 20).

### 3.3 Physical scope of the first usable solver

The first usable coupled solver **MUST** target: interpretive mode; a fixed, closed
boundary with **B**·n̂ = 0; prescribed toroidal flux Ψ_t; prescribed p₀(V) with
p₀(V_Ω) = p_b; prescribed net-parallel-current profile F(p); manually specified
regularization/transport coefficients; one shared-memory node.

Explicitly deferred: predictive pressure evolution with a physical heating source
(the solver is shared, but predictive mode is not a v1 deliverable); nonzero **B**·n̂ at
material boundaries; scrape-off-layer and sheath boundary conditions; free-boundary
coupling to coils and an exterior vacuum region; MPI as a required execution mode; GPU;
automatic adjoints; separate electron and ion energy equations.

### 3.4 Nonlinear solution sequence

Implement in this order: (1) damped Picard; (2) Anderson-accelerated Picard;
(3) pseudo-transient / continuation-based globalization; (4) coupled Newton–Krylov;
(5) pseudo-arclength continuation and tangent/adjoint capability. The Picard subsolvers
MUST be designed so they can later serve as blocks of a Newton preconditioner.

### 3.5 Magnetic representation

The 3D fixed-boundary solver **MUST** use a vector potential internally:

    B = Ψ_t B_h + ∇×A

with A in an H(curl) Nédélec space; B_h a divergence-free harmonic field carrying unit
toroidal flux; a gauge constraint removing the gradient null space. The public API and
output files SHOULD expose **B**, **J**, p, u, χ directly; users should not need **A**
unless they ask for it.

### 3.6 Extreme anisotropy

High-order continuous Galerkin is the **baseline**, not the final proof of robustness.
remec **MUST NOT** claim reliable support for a given anisotropy ratio merely because
linear residuals converge: support requires a numerical-perpendicular-diffusion
measurement showing artificial cross-field diffusion comfortably below the requested
physical κ⊥ (Section 9.3). The architecture MUST permit a later asymptotic-preserving
(AP) two-field formulation; before remec claims robust operation near the physical range
κ∥/κ⊥ ~ 10⁸–10¹⁰, an AP method or an equally convincing alternative MUST be implemented
and compared against the standard formulation.

### 3.7 Decision summary with revisit triggers

| # | Decision | Rationale | Revisit trigger |
|---|---|---|---|
| D1 | Python + NGSolve backend | pip binary wheels on all platforms, no MPI/compiler; symbolic forms + `AssembleLinearization`; high-order curved H¹/H(curl)/H(div); `TaskManager` threads | Backend is the measured bottleneck, or GPU/multi-node becomes a requirement (→ PETSc branch evidence, ADR) |
| D2 | Threads only by default | Single-node target | Memory exceeds one node; PETSc branch shows large gains |
| D3 | A internal, B = Ψ_t B_h + ∇×A, B public | Discrete ∇·(∇×A) = 0 at roundoff; Ψ_t exact via harmonic basis | none anticipated |
| D4 | Anisotropic baseline: high-order CG + direct measurement of pollution; AP as required later milestone | High order controls pollution at moderate ε_κ (NIMROD: p ≥ 3 non-aligned); AP addresses ε_κ→0 conditioning/accuracy | Claims at ε_κ ≲ 10⁻⁸, or iterative-solver conditioning blocks production sizes |
| D5 | M3 baseline: CG + SUPG; upwind-DG fallback | Layers kept ≥ ~6 cells by policy, where SUPG-CG behaves; far fewer DOFs | Oscillations or layer smearing during D_u continuation |
| D6 | Linear solves: direct factorization permitted (and default) for 2D/axisymmetric/small-3D verification; iterative + preconditioning is the planned production 3D path; never assemble the coupled Jacobian | Direct is robust to anisotropy conditioning and fastest to correctness at small scale, but does not scale to multi-million-DOF 3D blocks | Configurable DOF threshold; preconditioner benchmark program (Sec. 9.5) |
| D7 | V_χ: gradient-scaled mollified Heaviside (differentiable) for solves; volume-uniform tabulation; monotone PCHIP inverse; optional ngsxfem cut-cell reference | Note Sec. 8; Tornberg–Engquist consistency; histogram is non-differentiable | none |
| D8 | Divergence-free H(div) current projection between (M2) and Ampère, correction monitored | Pointwise J is not div-conforming; Ampère needs a compatible RHS | Correction fails to converge under refinement → mixed u–J formulation |
| D9 | Picard as named operator objects; Newton reuses them as preconditioner blocks | Note Sec. 9.1 | none |
| D10 | Interpretive first-class; predictive a variant of the same solver | Note's priority; positivity automatic via composition | none |
| D11 | Fixed closed boundary v1; BC layer represents general g = B·n̂ internally but the physical model rejects g ≠ 0 | Nonzero g is a distinct open-field-line model class needing sheath physics (note Sec. 6) | SOL milestone |
| D12 | Nondimensional FEM core with a `Normalization` object; SI only at user-facing adapters | Scale separation (κ∥/κ⊥ ~ 10¹⁰, enormous SI κ values) wrecks matrix scaling; the note's knobs (ε_κ, ε_J) are dimensionless | none |
| D13 | Meshes built for element quality (robust curved tet/prism first), NOT from the source equilibrium's flux-coordinate map; deformation-of-a-reference-mesh structure retained | Flux maps degenerate at the axis and shear badly under strong shaping; remec solutions need not retain imported surfaces | Multi-block prism/hex upgrade for high-order efficiency |
| D14 | Newton: Stage A = JFNK with Picard-block preconditioning; Stage B = exact symbolic local Jacobian + explicit low-rank nonlocal V_χ term | JFNK is lowest-risk first; the BSpline transplant mechanism (Sec. 13.5) makes Stage B's local part nearly free | none |
| D15 | Transport coefficients implemented from the note's Sec. 10.2 formulas; PlasmaPy test-only | Few lines; avoids heavy dependency with an API under development | none |
| D16 | Axisymmetric R–Z as a true reduced formulation, early; 3D-axisymmetric cross-check where feasible | Most sensitive end-to-end verification (GS + 1D transport); fastest dev platform; independent check of the note's Sec. 11 reduction | none |
| D17 | Derivatives deferred but architected for: pure residuals, transpose actions, fixed reference mesh, smooth V_χ only, JVP/VJP interfaces | Converged-state adjoint reuses the Newton Jacobian transpose | Optimization use case materializes |
| D18 | M3 regularization gradient runtime-selectable: ∇⊥ (default) or full ∇ (isotropic variant), applied consistently across (M2)/(M3); comparison study required (Sec. 9.4, milestone 3.5) | Note §5.5: variants agree to O(ε_J); full ∇ gives a fixed SPD **B**-independent Laplacian (assembly/preconditioner reuse, monotone stencils, no ∂op/∂**B** Newton block, damps parallel grid noise); ∇⊥ is the derived kinetic closure and preserves u = J∥/B exactly | Measured evidence of a clear winner → change the default via ADR |

---

## 4. Governing model (summary; the note is authoritative)

With **b** = **B**/B, ∇⊥f = ∇f − **b**(**b**·∇f), K = κ⊥I + (κ∥ − κ⊥)**bb**ᵀ,
ε_κ = κ⊥/κ∥:

- **(M1)** ∇·**B** = 0, μ₀**J** = ∇×**B**.
- **(M2)** **J** = u**B** + (**B**×∇p)/B² − D_u ∇⊥u.
- **(M3)** **B**·∇u − ∇·(D_u∇⊥u) = (2/B³)**B**·(∇p×∇B) − (μ₀u/B²)**B**·∇p
  + (μ₀D_u/B²)∇⊥u·∇p. The final right-hand-side term is part of the model and
  **MUST NOT** be omitted: it is required for (M3) to be exactly ∇·**J** = 0 for the
  current (M2), which is what makes Ampère's law integrable.
- **(M2/M3 regularization-gradient variants — note §5.5):** the regularizing gradient
  ∇ᵣ in (M2) and (M3) is runtime-selectable between ∇⊥ (default; the note's derived
  closure) and the full ∇ (isotropic variant): **J** = u**B** + (**B**×∇p)/B² − D_u∇ᵣu
  and **B**·∇u − ∇·(D_u∇ᵣu) = (2/B³)**B**·(∇p×∇B) − (μ₀u/B²)**B**·∇p
  + (μ₀D_u/B²)∇ᵣu·∇p. The same ∇ᵣ MUST be used in the (M2) flux, the (M3) left-hand
  side, and the (M3) final right-hand-side term — a mixed pairing leaves an O(D_u)
  residual in ∇·**J** and MUST be treated as an error. With ∇ᵣ = ∇, u is the auxiliary
  solved field and J∥/B = u − (D_u/B)**b**·∇u; diagnostics and bootstrap-type F(p)
  closures MUST use the reconstructed **J**. Requirements and the comparison program
  are in Section 9.4. This licence is specific to (M3); it does **not** extend to
  (M4a)/(M4).
- **(M4a)** ∇·(K∇χ) = −S_ref, S_ref > 0 (default S_ref = 1 after nondimensionalization).
- **(M4b)** p(**r**) = p₀(V_χ(χ(**r**))), V_χ(χ̂) = ∫_Ω H(χ − χ̂) d³r. No independent
  pressure DOFs exist in interpretive mode.
- **(M4)** predictive: the same operator applied to p with a physical source S_p ≥ 0.

**Boundary conditions (v1):** **B**·n̂ = 0, Ψ_t prescribed, χ = 0, u = F(p_b) on ∂Ω.
The preferred solved variable for (M3) is ũ = u − F(p) with homogeneous data ũ = 0; the
transformed source terms **MUST** be transcribed from the note's Eq. (utilde_equation)
— an agent MUST NOT improvise them. Until that transcription is verified, an early kernel
MAY solve directly for u with Dirichlet data u = F(p_b).

**Nonzero B·n̂:** the boundary API SHOULD eventually represent prescribed
g = **B**·n̂ with ∮g dA = 0, but this is not merely another fixed-boundary option: open
field lines require parallel-loss/sheath conditions on the transport equations. General
g MAY be represented internally; the initial physical model MUST reject g ≠ 0;
open-field-line physics MUST be a distinct future model class. Do not conflate it with
free-boundary mode (coil field + emergent edge), which is a separate future module.

**Key scalings** (resolution policy and extrapolation): barrier/flattening width
w_c ∝ ε_κ^{1/4}; current-layer width δ ∝ D_u^{1/3}; |**b**·∇χ|/|∇χ| = O(ε_κ L∥/L⊥).
Production strategy per note Sec. 9: run at moderate anisotropy (ε_κ ~ 10⁻⁶–10⁻⁸) with
layers resolved, scan ε_κ and D_u, extrapolate with the known scalings.

Character of the sub-problems: (M4a) self-adjoint elliptic, extremely anisotropic —
the central numerical risk; (M3) nonsymmetric advection–diffusion, first order along
**b**, elliptic transverse, hypoelliptic (uniformly elliptic in the full-∇ variant of
note §5.5); (M1)+(M2) div–curl/curl–curl; (M4b) explicit
nonlocal composition, local + low-rank linearization.

---

## 5. Numerical invariants (monitored or preserved by construction)

1. **Magnetic divergence:** ∇_h·(∇_h×A_h) = 0 algebraically to roundoff; a regression
   test MUST measure this independently of nonlinear convergence.
2. **Current continuity:** raw (M2) currents are projected to a discretely
   divergence-free H(div) field (Sec. 11); the relative projection correction
   ‖J_h − J_raw‖/‖J_raw‖ MUST be recorded and MUST converge to zero under refinement —
   a large or non-convergent correction means the M3 discretization and M2
   reconstruction are inconsistent.
3. **Profile realization:** the transplant MUST reproduce p₀(V) to
   quadrature/interpolation accuracy; test the layer-cake identity
   ∫φ(p)d³r = ∫φ(p₀(V))dV for a spline family of test functions φ, and the endpoint
   identities V_χ(0) = V_Ω, V_χ(max χ) = 0.
4. **Positivity/boundedness:** interpretive p MUST remain within the range of p₀ at all
   iterates. Monitor min/max p, min B, whether any B-floor is active, and monotonicity
   of the fitted V_χ and of p₀.
5. **Resolved layers:** a converged algebraic solve is not a resolved solution. Estimate
   and report the number of local element widths across w_c and δ; a production run
   MUST warn (strict mode: fail) when either falls below `min_layer_cells`
   (default 6).
6. **Global balances** (note Sec. 6, "Global consistency checks"): total power
   (∫S_ref or ∫S_p^eff = Γ(0)) vs. boundary heat flux; net toroidal current through a
   poloidal cross-section vs. the value encoded in F; toroidal flux error vs. Ψ_t.

---

## 6. Nondimensionalization, units, and small-B protection

The FEM core SHOULD solve nondimensional equations. A `Normalization` object MUST record
at least: reference length L₀; reference field B₀; pressure scale B₀²/μ₀; current-density
scale J₀ = B₀/(μ₀L₀); vector-potential scale B₀L₀; u scale 1/(μ₀L₀); D_u scale B₀L₀
(so ε_J = D_u/(B̄L̄) as in the note); and the transport/source scales of the χ equation.
Normalization metadata MUST be written to every checkpoint. User-facing physical models
MAY accept SI (via `scipy.constants`; temperatures in eV converted explicitly at the
boundary), but MUST convert to the documented nondimensional form before assembly. Early
verification milestones MAY accept dimensionless inputs only; SI adapters come after the
dimensionless kernels pass manufactured-solution tests.

**Small-B protection:** expressions in 1/B, 1/B², 1/B³ need protection during failed
nonlinear iterates. Use the smooth floor B_safe = sqrt(**B**·**B** + B_floor²) —
this is a smooth regularization, consistent with the no-hard-clipping rule (D17), not a
branch. `B_floor` MUST be configurable, default far below the physical minimum, and its
activity monitored: a solution is unacceptable if the floor materially affects the
converged residual or observables. Every occurrence of B⁻¹ MUST use the same safe norm
and the same diagnostic.

---

## 7. Finite-element spaces and compatible magnetics

### 7.1 Spaces

Baseline: χ (and predictive p) and ũ in continuous H¹, default order 3; orders 3–5 MUST
be supported for anisotropy studies. A in `HCurl`; projected **B** and **J** stored in
`HDiv` when a stored field is required; `L2` for divergence constraints/diagnostics;
scalar H¹ gauge multiplier chosen to form a stable mixed pair. Mesh geometry order
comparable to the FE order (curved elements). Static condensation (`condense=True`)
SHOULD be used for high-order spaces where it helps.

**De Rham pairing caution:** the exact space/order pairing that yields the commuting
discrete sequence MUST be established by small de Rham-sequence tests (∇, ∇×, ∇· mapping
between the chosen spaces) **before** the 3D solver is built. Agents MUST NOT assume
that equal integer `order` arguments across NGSolve spaces automatically produce the
desired sequence.

Coefficients (**b**, B_safe, κ's, D_u, p) enter forms as NGSolve
`CoefficientFunction`s evaluated at quadrature points; **b** is a CF of the current
A (+ Ψ_t B_h), so its Newton linearization is available symbolically. Compile
frequently reused coefficient expressions.

### 7.2 Harmonic flux field B_h

A solid torus has one nontrivial harmonic magnetic component; the toroidal flux cannot
be represented by the curl of a single-valued A with homogeneous tangential data. The
geometry module MUST provide a normalized **B**_h with (to discretization accuracy)
∇·**B**_h = 0, ∇×**B**_h = 0, **B**_h·n̂ = 0, and unit toroidal flux through a chosen
cut surface. For a triply periodic box, three independent mean-flux components are
required (the constant unit vectors).

**Recommended first construction (simple and testable):** start from
**B**₀ = ∇φ_cyl = ê_φ/R (curl-free, div-free, unit circulation; the torus MUST link the
z-axis), then restore tangency with one scalar Neumann solve: find ψ ∈ H¹ with
∫∇ψ·∇w = −∮_{∂Ω}(**B**₀·n̂)w for all w (fix the constant); set **B**_h ∝ **B**₀ + ∇ψ and
normalize the flux to 1. Acceptable alternatives: a mixed harmonic-field solve, or a
scalar potential with a prescribed jump across an explicit cut surface. Whatever method
is chosen MUST be validated on a simple analytic torus (unit tests: weak curl and div
residuals, tangency, flux = 1) before any shaped-stellarator use. This is a dedicated
milestone; shaped 3D calculations MUST NOT begin before it passes.

### 7.3 Gauge and flux exactness

Initial magnetostatic solve: mixed Coulomb gauge —
(∇×A, ∇×v)/μ₀ + (∇λ, v) = (J, v), (A, ∇q) = 0 — with essential n̂×A = 0 and λ = 0.
With a compatible (projected) J the multiplier is ≈ 0; report ‖λ‖ as a diagnostic.
Alternative gauge strategies MAY be used only after demonstrating uniqueness of **B**,
robust solver convergence, no contamination of Ampère's law, and preserved fluxes. A
small mass penalty on A MUST NOT be introduced without quantifying its perturbation of
the field equation.

Flux exactness: with n̂×A = 0 on ∂Ω, the toroidal flux of ∇×A through a poloidal
cross-section S is ∮_{∂S}A·dl = 0 since ∂S ⊂ ∂Ω carries zero tangential A; the harmonic
term therefore carries the prescribed flux **exactly**, with no constraint equation.
The precise NGSolve tangential-trace implementation MUST be verified with a manufactured
vector potential and a toroidal-flux benchmark before becoming a reusable utility.

---

## 8. Anisotropic reference-potential solver (the central numerical risk)

### 8.1 Baseline weak form and discretization

For fixed **B**: find χ_h ∈ H¹₀ with ∫∇v·K∇χ dV = ∫v S_ref dV for all v. The
implementation MUST assemble the parallel piece κ∥(**b**·∇χ)(**b**·∇v) and the
perpendicular piece κ⊥∇⊥χ·∇⊥v separately so diagnostics can report each contribution.
High-order CG (default order 3; scans over order at fixed mesh and mesh at fixed order
MUST be supported), curved geometry where relevant, quadrature order high enough that
coefficient/geometry integration is not the dominant error. SPD system ⇒ preconditioned
CG, or a sparse direct factorization within the Section 21 solver policy (direct is the
default for 2D/axisymmetric/small-3D; the factorization is reused across
source-refinement sub-iterations within one outer step and refactorized once per field
update).

### 8.2 What high order does and does not solve

High order reduces truncation error and artificial perpendicular diffusion — this is the
established mechanism (NIMROD: accurate anisotropic conduction at realistic κ∥/κ⊥ for
p ≥ 3 without mesh alignment). It does **not** guarantee uniform conditioning or accuracy
as ε_κ → 0, especially with closed field lines, islands, nearly invariant regions, or
unresolved layers. The baseline solver is therefore named `StandardCG`, not
`ExtremeAnisotropySolver`.

### 8.3 Numerical-pollution regression (the acceptance metric)

The repository MUST include a Sovinec-style steady test measuring an effective numerical
perpendicular diffusivity: closed-field-line **b** (circular field lines / analytic
island), κ⊥ = 0, source Q = Q₀cos(kx)cos(ky); the steady central amplitude gives
κ⊥,num = Q₀/(2k²χ(0)). For every supported order, report κ⊥,num/κ∥ vs. mesh size and
anisotropy, as a machine-readable regression table. A production run MUST warn (strict
mode: fail, `AnisotropyPollutionError`) when measured or conservatively estimated
κ⊥,num is not sufficiently below the requested physical value; configurable safety
factor, conservative default κ⊥,num < 0.1 κ⊥.

### 8.4 Strategy interface and the AP milestone

```python
class AnisotropicDiffusionSolver(Protocol):
    def solve(self, field, coefficients, source, boundary, initial=None): ...
    def apply(self, x): ...                    # operator action (for Newton/precond.)
    def build_preconditioner(self): ...
    def diagnostics(self) -> dict[str, float]: ...
```

Planned strategies: `StandardCG`; `ReducedPollutionCG` (Günter-patterned variant, MAY);
`IteratedTwoFieldAP` (required milestone, Phase 7); `InteriorPenaltyDG`/`HDG` (MAY —
also the cross-validation discretization and the wall-geometry option). The AP
implementation MUST be based on a cited derivation (Deluzet–Narski iterated two-field or
a documented alternative) selected in an ADR **before coding begins**; agents MUST NOT
invent an AP saddle system from memory. When AP is used inside Newton, prefer keeping
the residual in plain form and the AP machinery inside the preconditioner, preserving
the property that the χ block of the Jacobian is the same uniformly elliptic operator at
every iterate (note Sec. 9.1).

**Cross-verification doctrine:** agreement between two independent discretizations
(e.g., `StandardCG` vs. HDG or AP) is far stronger evidence for tiny perpendicular
fluxes than mesh convergence of one method. A third, error-uncorrelated referee — a
Lagrangian Green's-function / semi-Lagrangian solve on frozen **B**
(del-Castillo-Negrete–Chacón line of work) — SHOULD be added as a standalone
verification tool for integrable and chaotic test fields; it never enters the production
solve path.

### 8.5 Preconditioning program (for the iterative production path)

Native NGSolve candidates: geometric or p-multigrid; BDDC; additive Schwarz / patch
smoothers; line/graph smoothers adapted to the anisotropy direction; static
condensation. The chosen preconditioner MUST be benchmarked on: straight aligned fields;
oblique non-aligned fields; closed field lines; a magnetic island; a chaotic test field
— reporting iteration counts vs. mesh size, order, and anisotropy. (hypre AMS and
field-split alternatives belong to the PETSc branch evaluation.)

---

## 9. Regularized current-continuity solver (M3)

### 9.1 Weak form and SUPG

Unstabilized: ∫v **B**·∇u + ∫D_u ∇ᵣv·∇ᵣu = ∫v R(B, p, u, D_u), with R the complete
right-hand side of (M3) and ∇ᵣ the selected regularization gradient (∇⊥ by default,
full ∇ as the isotropic variant; Sec. 9.4); every B⁻¹ uses B_safe. The baseline adds SUPG stabilization
with streamline direction **b**. Requirements: the strong residual used in SUPG MUST
include the parallel advection term, the transverse diffusion, and **every** source and
reaction term (including coefficient derivatives implied by the strong divergence where
needed); the stabilization parameter MUST live in one centralized, unit-tested function
depending on element size along the field, |**B**|, transverse diffusion, and polynomial
order; the stabilization contribution MUST be separately reported in diagnostics.

### 9.2 ũ formulation

Prefer ũ = u − F(p) (homogeneous BC). The transformed equation MUST be transcribed from
the note's Eq. (utilde_equation) and verified line-by-line against the direct-u
formulation on a manufactured case (the two MUST agree to discretization accuracy).

### 9.3 DG fallback and linear solvers

If SUPG oscillates or smears the D_u^{1/3} layer, implement upwind interior-penalty DG
conforming to the same solver interface and sharing the manufactured solutions. The M3
block is nonsymmetric: GMRES with a native preconditioner, or direct within the
Section 21 policy. At least one manufactured test MUST fail conspicuously if the last
(μ₀D_u/B²)∇ᵣu·∇p term of (M3) is omitted; this test MUST run for both gradient
variants.

### 9.4 Regularization-gradient variants (∇⊥ vs. full ∇)

The note's derived closure is the transverse −D_u∇⊥u; note §5.5 licenses an isotropic
variant −D_u∇u that is equivalent to the order of the retained physics (variant
solutions differ by O(ε_J) relative) and is expected to be numerically simpler: for
constant D_u it is a fixed SPD Laplacian independent of the evolving **B** (assembly,
factorization, and preconditioner reuse across nonlinear iterations; monotone stencils
without mixed derivatives; no differentiation of **bb** inside the divergence; no
∂(operator)/∂**B** Newton block; damping of parallel grid-scale noise that the
transverse operator leaves untouched). Whether these advantages are significant in
practice is an open experimental question this section exists to answer.

Requirements:

- The M3 solver MUST implement both variants behind one runtime option, e.g.
  `regularization_gradient="perpendicular" | "full"`, default `"perpendicular"` (the
  note is authoritative for the physical model; changing the default requires an ADR
  citing the milestone-3.5 measurements).
- The selected ∇ᵣ MUST be applied consistently in: the (M2) constitutive flux and the
  J_raw construction (Sec. 10); the (M3) weak form; the SUPG strong residual; the final
  (μ₀D_u/B²)∇ᵣu·∇p right-hand-side term; and the ũ-transformation source terms
  transcribed from Eq. (utilde_equation) (with ∇ᵣ replacing ∇⊥ in its
  ∇·(D_u F′(p)∇⊥p) term). Mixing operators between any of these breaks exact
  ∇·**J** = 0 at O(D_u) and MUST be treated as an implementation error, not a
  tolerance issue.
- The choice MUST be recorded in the run configuration digest, structured logs, and
  checkpoint metadata.
- With `"full"`, u is the auxiliary solved variable; J∥/B = u − (D_u/B)**b**·∇u MUST be
  the quantity reported by parallel-current diagnostics and consumed by bootstrap-type
  F(p) closures.
- All Section 9.1–9.3 manufactured tests, and the D_u^{1/3} layer-scaling
  demonstration, MUST pass for both variants.
- **Comparison study (milestone 3.5):** on shared frozen-(**B**, p) benchmarks
  (including a resonant-layer case and a deliberately field-misaligned mesh), measure
  and record as machine-readable tables: (a) the relative difference between the two
  converged solutions at fixed D_u, verifying the O(ε_J) expectation and common
  D_u → 0 limits; (b) assembly/refactorization counts and wall time per nonlinear
  iteration; (c) linear-iteration counts and preconditioner behavior; (d) presence of
  oscillations / monotonicity violations and layer smearing; (e) sensitivity to
  mesh–field misalignment (the ∇⊥ tensor discretization incurs spurious parallel
  diffusion ≤ ~D_u × misalignment error — Günter et al., Sharma–Hammett — which the
  full-∇ variant accepts by construction); (f) damping of parallel grid-scale noise.
  Results go in `docs/verification.md`. Because the two variants share no
  discretization of the regularizing term, their agreement is also a strong
  cross-verification of the M3 kernel itself.
- **Scope limitation:** this licence is specific to (M3), where parallel transport is
  advective with an O(1) coefficient so added parallel diffusion is subdominant by
  ε_J k∥L̄ ≪ 1. It MUST NOT be applied to (M4a)/(M4), where both directions are
  diffusive and the anisotropy ratio is the physics: isotropic contamination at the
  κ∥ level destroys w_c ∝ ε_κ^{1/4} and the barrier structure (Sec. 3.6, Sec. 8).

---

## 10. Current construction and divergence-free projection

Construct J_raw = u**B** + (**B**×∇p)/B_safe² − D_u∇ᵣu at quadrature points, where ∇ᵣ
MUST match the regularization gradient selected for the M3 solve (Sec. 9.4), with each
term separately accessible for diagnostics/output. Then solve the constrained
projection: (J_h, v) + (λ_h, ∇·v) = (J_raw, v), (∇·J_h, q) = 0, with J_h ∈ H(div),
natural J_h·n̂ = 0 on the closed boundary, and explicit handling of any global current
component not fixed by the local constraint. Diagnostics MUST include divergence norm
before/after, relative projection correction, net current integrals, and the Ampère
compatibility residual. The projection is linear ⇒ differentiable ⇒ safe inside Newton.
If the correction does not converge rapidly under refinement, replace the sequential
M3-plus-projection construction with a mixed solve coupling u, **J**, and the
continuity multiplier (ADR required).

---

## 11. Magnetic update

Given compatible J_h, solve the gauged curl–curl problem for A and set
B_candidate = Ψ_t B_h + ∇×A. Under-relax **on the vector-potential coefficients** with
the harmonic flux coefficient held fixed — A^{k+1} = (1−α)A^k + αA_candidate — which
automatically preserves the prescribed flux and the curl representation. Each magnetic
solve MUST report: Ampère residual; magnetic divergence; boundary normal-field residual;
gauge residual ‖λ‖; toroidal-flux error; magnetic energy; min/max B. Biot–Savart /
virtual casing is NOT part of the fixed-boundary solve; a future free-boundary module
MAY use a virtual-casing library, boundary elements, direct quadrature of the FEM
current, or a coupled vacuum FEM region, and MUST remain isolated behind the boundary
interface.

---

## 12. Level-set volume map and pressure transplant

### 12.1 Interface (independent of the nonlinear solver)

```python
class LevelSetVolumeMap:
    def build(self, chi, mesh, options) -> "LevelSetVolumeMap": ...
    def volume(self, chi_level): ...
    def inverse_level(self, volume): ...
    def evaluate_volume_coordinate(self, points_or_quadrature): ...
    def jvp(self, delta_chi): ...
    def diagnostics(self) -> dict[str, float]: ...
```

### 12.2 Mollified evaluation (default; differentiable)

V_χ^ε(χ̂) = ∫ H_ε(χ − χ̂) dV with value-space width scaled to a fixed **spatial** width:
ε_χ ∝ c·h·|∇χ| locally (c ≈ 1–2 cells), with safeguards near critical points. A constant
width in χ-space MUST NOT be the default (spatially inconsistent smoothing where |∇χ|
varies; Tornberg–Engquist). Implementation: one pass extracting quadrature values of χ,
|∇χ|, and weights w_i to NumPy, then vectorized over levels; this same data yields the
derivatives ∂V_χ^ε/∂χ_i = H_ε′(χ_i − χ̂)w_i and ∂V_χ^ε/∂χ̂ = −Σ H_ε′w_i ≈ −ρ(χ̂)
(co-area density) needed for the JVP.

### 12.3 Tabulation, inversion, critical levels

Tabulate at levels approximately **uniform in enclosed volume**, not in χ (across
flattened regions V_χ is nearly a step; the inverse χ̂(V) is tame). Enforce monotonicity
with a monotone interpolant (PCHIP or I-spline); provide both V_χ(χ̂) and χ̂(V); never
numerically differentiate the near-step direction. Optional adaptive level refinement
near detected critical (separatrix) values. Cheap mandatory checks per build:
endpoint identities, spline monotonicity, spot agreement of −dV/dχ̂ with an independent
co-area integral.

### 12.4 Cut-cell reference (optional extra)

An `ngsxfem`-based high-order implicit-domain integration path SHOULD be added behind
the same interface for: accurate final V_χ evaluation; verification of the mollified
method; separatrix/critical-level studies; layer-cake moments. `ngsxfem` MUST remain an
optional extra so the minimal installation stays simple.

### 12.5 Profiles and the transplant mechanism

```python
class VolumeProfile(Protocol):
    def value(self, volume): ...
    def derivative(self, volume): ...
    def validate(self, total_volume, edge_value=None): ...
```

Implementations: tabulated monotone profile; analytic callable; spline; normalized
profile on s = V/V_Ω. p₀ MUST be non-increasing on [0, V_Ω] with p₀(V_Ω) = p_b; the
interpretive edge-vacuum plateau (note Sec. 7.3) is simply a p₀ constant for V ≥ V_p.
The composed map g = p₀∘V_χ SHOULD be represented as a monotone 1D spline wrapped as a
differentiable NGSolve 1D CoefficientFunction (`BSpline`) applied to χ, so that p,
∇p = g′(χ)∇χ, and the **local** Newton linearization g′(χ)δχ all flow through symbolic
differentiation automatically.

### 12.6 Linearization

δp = g′(χ)δχ (local) + p₀′(V_χ)·δV_χ (nonlocal; dense along level sets, low rank in the
tabulation-level index; one level-set-averaging pass per application, built from the
H_ε′w_i data). Picard does not need this derivative. The Newton milestone MUST provide
either (a) a finite-difference JVP of the complete smooth residual (Stage A), or (b) the
exact local Jacobian plus the explicit low-rank nonlocal JVP (Stage B). The histogram
volume map MUST NOT be used in Newton (discontinuous under small coefficient changes).

---

## 13. Picard iteration

### 13.1 Cycle (per iterate k)

1. form **B**^k, **b**^k; 2. evaluate transport coefficients on the state; 3. solve
(M4a) for χ^k; 4. build V_{χ^k}; 5. p^k = p₀(V_{χ^k}(χ^k)); (optionally apply the
under-relaxed, positivity-floored S_ref source-refinement update of note Sec. 6.4
item iii before step 3); 6. solve (M3) for ũ^k, set u^k = F(p^k) + ũ^k; 7. build
J_raw^k; 8. project to divergence-free J_h^k; 9. magnetic update for A_candidate;
10. damping or Anderson; 11. residuals, physical diagnostics, layer-resolution
diagnostics; 12. checkpoint at the configured cadence.

### 13.2 Convergence criteria and norms

Convergence MUST require more than a small state update. Default stopping logic
requires ALL of: normalized M1/Ampère, M3, and M4a residuals below tolerance; state
update norm below tolerance; profile-transplant error below tolerance; divergence and
flux invariants within tolerances; no active numerical floors materially affecting the
solution. Use physically scaled per-block norms; a raw Euclidean norm over concatenated
DOFs MUST NOT determine convergence (blocks differ in units, counts, and scales).

### 13.3 Damping and Anderson

Begin with scalar under-relaxation (log damping decisions and rejected steps);
block-specific damping only if needed. Anderson acceleration: backend-independent Python
module on flattened block vectors through an adapter; configurable depth (~5);
regularized least squares with restart on ill-conditioning; fallback to damped Picard;
tested on simple fixed-point problems before use in remec; MUST preserve fixed
harmonic-flux coefficients and homogeneous essential BCs.

---

## 14. Newton and continuation

### 14.1 Entry gates

Do not implement monolithic Newton until: every Picard block has independent
manufactured-solution tests; an end-to-end Picard solve converges in at least one 2D or
axisymmetric benchmark; the complete residual is a pure, side-effect-free evaluation
(no mutation of profiles or accepted state, no hidden damping); restart and diagnostics
are stable.

### 14.2 State and residual

Interpretive Newton state x = (A, χ, ũ, gauge variables), with p eliminated through the
composition; a later mixed state MAY add **J** and a continuity multiplier. The residual
MUST be a pure function of immutable inputs, current state, mesh/spaces, and
continuation parameters.

### 14.3 Two-stage Jacobian plan

**Stage A — JFNK:** finite-difference Jacobian-vector products of the complete smooth
residual (verify JVP convergence as the FD parameter varies); Krylov = GMRES; right
preconditioner = block lower-triangular sweep of the Picard operators (their stored
factorizations/solvers), applied in Picard order.

**Stage B — hybrid exact:** NGSolve symbolic local linearization
(`AssembleLinearization`; the BSpline transplant makes the local composition term
automatic) plus the explicit low-rank nonlocal V_χ JVP (12.6); supply transpose actions
when sensitivity work begins. The coupled Jacobian MUST NOT be assembled as a single
matrix; per-block preconditioner matrices are fine.

### 14.4 Globalization and continuation

At least one robust globalization MUST exist: residual-based backtracking line search,
trust region, or pseudo-transient continuation (σI/Δτ added to diagonal blocks, Δτ grown
as ‖G‖ falls — recovering HINT-like relaxation at small Δτ). Preferred workflow:
Picard/pseudo-transient from the initial guess → switch to Newton inside a configured
basin → continuation in β, D_u, or ε_κ for difficult regimes (natural continuation
first — reuse the previous solution as the initial guess in scans — pseudo-arclength
later). Interpretive mode needs no positivity safeguard at Newton iterates; predictive
mode SHOULD offer a log-p variable rather than clipping. Required test: Picard/Newton
agreement on a common case to tight tolerance.

---

## 15. Transport coefficient models

```python
class TransportModel(Protocol):
    def evaluate(self, state, geometry, volume_coordinate) -> "TransportFields": ...
```

`TransportFields` provides κ∥, κ⊥, D_u, the anisotropy ratio, any applied floors, caps,
or multipliers, and provenance metadata.

- **Manual models first (required for verification and scans):** constant coefficients;
  analytic spatial coefficients; profiles of normalized enclosed volume; direct
  specification of ε_κ + a reference conductivity; direct D_u.
- **Braginskii (later):** the note's Sec. 10.2 formulas — κ∥ ≈ 3.16 n_eT_eτ_e/m_e,
  κ⊥ ≈ 2.0 n_iT_i/(m_iΩ_i²τ_i), D_u ~ B̄ρ_e²/λ_e with a documented O(1) prefactor
  uncertainty (expose it). Users supply n(V), T_e(V), T_i(V) (transplanted through the
  same χ level sets; the outer iteration absorbs the iterate dependence). If p₀ is not
  given separately, construct it from the species profiles; if both are given, validate
  consistency and warn or reject on a configurable tolerance. Unit-test against PlasmaPy
  values (test-only dependency).
- **Caveats the code MUST keep distinct** (in metadata and docs): classical Braginskii
  ⊥ transport vs. neoclassical vs. anomalous/turbulent vs. numerical regularization
  floors; a free-streaming/connection-length cap on κ∥ SHOULD be available
  (κ∥ ≲ α n v_te L_c, α ≈ 0.1); the single total-pressure equation is an effective
  closure, not the two-temperature Braginskii energy system.
- The solver API SHOULD make D_u and ε_κ scans easy; a scientific result SHOULD include
  evidence of insensitivity once layers are resolved. Run logs record physical vs.
  floored values.

---

## 16. Geometry and meshes

### 16.1 Interface and classes

```python
class Geometry(Protocol):
    def build_mesh(self, options) -> "MeshBundle": ...
    def boundary_regions(self) -> dict[str, object]: ...
    def characteristic_length(self) -> float: ...
    def harmonic_basis(self, mesh_bundle) -> list[object]: ...
    def metadata(self) -> dict[str, object]: ...
```

Implement in order: `Slab2D` → `PeriodicBox3D` → `AxisymmetricRZ` →
`SmoothSolidTorus3D` → `WallBoundedToroidalDomain`. Boundary regions MUST be named
(future plates, walls, control surfaces).

### 16.2 Slab and periodic box

The 2D slab is the primary kernel-development environment: Dirichlet and periodic scalar
BCs, prescribed analytic **B** fields, manufactured anisotropic-diffusion and M3
solutions, island-like test fields. The periodic box uses Netgen periodic
identifications and NGSolve `Periodic` spaces; MUST test scalar and vector periodicity,
all three mean-flux components, and high-order compatibility — and do not assume every
helper/preconditioner supports periodic wrappers; test each selected solver.

### 16.3 Axisymmetric R–Z

A genuine reduced formulation of the note's Sec. 11 equations — NOT a one-cell 3D
wedge. Metric factors MUST be visible in the weak forms and covered by manufactured
solutions. This is the critical end-to-end verification environment (Grad–Shafranov +
1D transport). Where feasible, also cross-check the reduced solver against a 3D run on
an axisymmetric configuration (independent verification of the reduction itself).

### 16.4 Smooth solid torus (adjudicated strategy)

Favor robustness over an ideal structured mesh: (1) sample the Fourier boundary
accurately; (2) build a high-quality curved tetrahedral or prism-dominated volume mesh;
(3) curve the boundary to an order comparable to the FE order; (4) measure and report
the geometry-approximation error (the geometry is "exact" only relative to the
truncated Fourier representation and the polynomial geometry order); (5) later add a
multi-block prism/hex mesh (with a disk topology avoiding a polar axis singularity) for
high-order efficiency and reduced pollution.

**Do NOT use a VMEC/DESC flux-coordinate map as the remec mesh**: it degenerates at the
magnetic axis, can produce badly sheared elements under strong shaping, remec solutions
need not retain the imported surfaces, and wall domains do not follow source
coordinates. The map remains useful for *initialization* (Sec. 17). For future shape
derivatives, meshes SHOULD be constructed as deformations of a fixed reference mesh
(store the reference and the deformation field).

### 16.5 Wall and divertor domains

Netgen/OCC where practical; Gmsh MAY be an optional dependency for CAD-heavy walls.
Sharp corners and flat plates reduce regularity: prefer local refinement (and possibly
locally reduced order) over globally increasing order. Mesh-quality gates: reject
inverted or near-degenerate elements (`MeshQualityError`).

---

## 17. Initialization from DESC, VMEC, and VMEC++

**Readers:** one shared lightweight netCDF reader for VMEC2000 and VMEC++ `wout_*.nc`
(VMEC++ writes the classic format); an h5py reader for DESC native HDF5 (record DESC's
class/version metadata in remec checkpoints) plus support for DESC-exported `wout`.
DESC, SIMSOPT, and VMEC++ Python packages MUST be optional extras; default readers
depend only on `netCDF4`/`h5py`.

**Two transfer paths (both required eventually):**
- *Coordinate-matched:* when the remec mesh was deliberately constructed from the source
  parameterization (e.g., boundary-fitted), use it directly.
- *Physical-space:* evaluate source fields at arbitrary remec quadrature points; this
  generally requires coordinate inversion (vectorized Newton root-finds with good
  initial guesses) or point location. Agents MUST NOT assume inversion is unnecessary
  for general or remeshed domains; it is an initialization-only cost.

**Compatible import pipeline:** (1) evaluate **B** in physical components; (2) project
into the divergence-conforming representation (preserving normal flux / discrete
divergence); (3) separate the harmonic flux component; (4) reconstruct A by the
gauge-fixed curl-constrained solve (same operator as Sec. 7.3 with RHS
(B_target, ∇×v)); (5) initialize p₀(V) from the source equilibrium's own pressure and
V(s) — do NOT keep any inherited flux coordinate as a permanent remec coordinate;
(6) u ≈ **J**·**B**/B²; (7) one consistent M3 solve before coupled iteration. The
importer MUST report transfer errors in: magnetic divergence; boundary normal field;
toroidal flux; magnetic energy; Ampère residual where source current is available.
Bundle one coarse `wout` and one coarse DESC file (< 1 MB each) as regression inputs;
no other large binaries in git.

---

## 18. Public Python API

Small, typed, and independent of low-level NGSolve objects wherever practical.
Illustrative target (names may change; the separation of geometry, physics inputs,
solver options, and output MUST be retained):

```python
from remec import (EquilibriumProblem, FixedClosedBoundary, ManualTransport,
                   PicardOptions, SolverOptions, TabulatedVolumeProfile)
from remec.geometry import SmoothSolidTorus3D
from remec.io import load_vmec

initial = load_vmec("wout_example.nc")
geometry = SmoothSolidTorus3D.from_fourier_boundary(initial.boundary,
                                                    mesh_size=0.08, geometry_order=4)
problem = EquilibriumProblem(
    geometry=geometry,
    boundary=FixedClosedBoundary(toroidal_flux=initial.toroidal_flux),
    pressure_profile=TabulatedVolumeProfile(normalized_volume=initial.s,
                                            pressure=initial.pressure),
    current_profile=initial.current_profile,
    transport=ManualTransport(kappa_parallel=1.0, epsilon_kappa=1e-6, D_u=1e-4),
    initial_state=initial,
)
solution = problem.solve(SolverOptions(
    nonlinear=PicardOptions(damping=0.2, max_iterations=200, anderson_depth=0),
    threads=16, checkpoint_interval=5))
solution.save("example.remec")
solution.write_vtk("example_fields")
```

No global mutable configuration: thread count, tolerances, quadrature order, floors, and
logging live in explicit option objects or a run context. NGSolve meshes, spaces, forms,
and grid functions stay inside internal implementation objects; public profile,
transport, and configuration classes remain importable without NGSolve until a mesh or
solution is actually constructed.

---

## 19. Diagnostics (physics outputs, not just residuals)

Beyond the invariant monitors of Section 5, the diagnostics package MUST provide the
note's interpretive-mode consistency outputs: the effective source
S_p^eff = g′(χ)S_ref − g″(χ)∇χ·K·∇χ (note Eq. Seff) and its level-set variance; the
implied 1D source S̄₀(V) = −d[G(V)p₀′(V)]/dV and total power Γ(0) (admissibility checks
on p₀); the geometric conductance G(V) (note Eq. conductance) — flagging flattened
regions where G is enormous; a field-line tracer with Poincaré sections (SciPy ODE on
the H(div) **B**) for topology visualization; and layer-width estimators for w_c and δ.
These feed the Section 5 balance checks and the verification battery.

---

## 20. Repository structure

```text
remec/
├── README.md                     # points newcomers/agents at docs/DESIGN.md §1
├── LICENSE
├── pyproject.toml
├── docs/
│   ├── 20260814-01_Regularized_3D_MHD_equilibrium.tex
│   ├── DESIGN.md
│   ├── equations.md              # transcribed weak forms with note eq. references
│   ├── verification.md           # test matrix and current regression numbers
│   ├── file_format.md            # checkpoint schema (versioned)
│   ├── dev_notes.md              # NGSolve API discoveries / deviations
│   └── adr/                      # 0001-ngsolve-backend.md, 0002-vector-potential.md,
│                                 # 0003-anisotropic-strategy.md, ...
├── src/remec/
│   ├── __init__.py, config.py, normalization.py, problem.py, solution.py,
│   │   state.py, profiles.py, boundary.py, cli.py
│   ├── geometry/    base.py slab.py periodic_box.py axisymmetric.py
│   │                solid_torus.py wall.py
│   ├── fem/         spaces.py forms.py operators.py quadrature.py projections.py
│   │                ngsolve_utils.py
│   ├── physics/     fields.py transport.py braginskii.py current.py
│   ├── solvers/     anisotropic.py current_continuity.py current_projection.py
│   │                magnetics.py picard.py anderson.py newton.py continuation.py
│   │                linalg/ (base.py native.py petsc_optional.py)
│   ├── levelsets/   mollified.py cutcell_optional.py interpolation.py
│   │                linearization.py
│   ├── io/          checkpoint.py vmec.py desc.py vtk.py
│   └── diagnostics/ residuals.py invariants.py layers.py seff.py conductance.py
│                    poincare.py performance.py
├── tests/           unit/ manufactured/ regression/ integration/ data/
├── examples/        slab_anisotropic.py periodic_island.py axisymmetric.py
│                    fixed_boundary_3d.py
└── benchmarks/      anisotropy/ scaling/ petsc_branch/
```

The structure may be simplified early, but physics, geometry, nonlinear orchestration,
level-set volume, and I/O MUST remain separable.

---

## 21. Shared-memory execution, linear-solver policy, and performance

- All computationally intensive NGSolve work runs inside `TaskManager`; the public
  `threads` option sets/validates thread configuration before entering the context.
- Avoid Python loops over elements/quadrature points: use CoefficientFunctions, compiled
  expressions, NGSolve integration, and vectorized NumPy reductions. The first likely
  Python hotspot is the V_χ tabulation — profile before optimizing; compiled extensions
  only after profiling (and only as optional acceleration).
- **Linear-solver policy (adjudicated):** sparse direct factorization is the default for
  2D, axisymmetric, and 3D blocks below a configurable DOF threshold, and always for
  verification problems (robust to anisotropy conditioning; fastest to correctness).
  The **planned production 3D algorithm is iterative** with the Section 8.5
  preconditioner program; a production-scale design MUST NOT depend on direct
  factorization of multi-million-DOF 3D blocks. Reuse assembled forms, factorizations,
  and preconditioners when coefficients have not changed materially.
- One-node feasibility requires: no assembled coupled Jacobian (ever); static
  condensation and/or matrix-free application where practical at high order; controlled
  Krylov work-vector counts; no duplicated field output; checkpoints that do not hold
  multiple complete state copies. Memory and timing by block MUST be part of benchmark
  runs; optimize the dominant block (expected: anisotropic diffusion) only after
  measuring.
- Correctness and verification precede optimization. Keep output cadence below nonlinear
  iteration cadence. Do not add PETSc because it is standard in HPC; add it only after
  the controlled Section 3.2 benchmarks.

---

## 22. Verification strategy (part of the architecture)

**Unit tests:** parallel/perpendicular projection identities; K symmetry and
eigenvalues (κ∥ along **b**, κ⊥ transverse); profile monotonicity and consistency;
normalization/unit conversions; Braginskii formulas (vs. PlasmaPy, test-only); B_safe;
Anderson algebra; mollifiers and their derivatives; checkpoint round-trips.

**Manufactured solutions:** scalar diffusion — isotropic Poisson; constant oblique
anisotropy; spatially varying anisotropy direction; curved geometry; periodic domain;
closed field lines; island topology; convergence measured in L² and energy norms.
M3 — pure aligned advection; transverse diffusion; reaction terms with **B**·∇p; the
final D_u∇ᵣu·∇p term (with a test that fails conspicuously if it is dropped);
nonconstant B; SUPG on/off; both regularization-gradient variants (∇⊥ and full ∇;
Sec. 9.4), with cross-variant agreement at O(ε_J) at fixed D_u and a common D_u → 0
limit. Magnetics — discrete ∇·∇× = 0; manufactured curl–curl;
gauge null-space handling; boundary normal-field condition; harmonic field + toroidal
flux; current projection + Ampère compatibility. Level sets — analytic circle/sphere;
deformed smooth level sets; multiple nested components; a saddle/separatrix level;
endpoint identities; layer-cake moments; JVP vs. finite differences; mollified vs.
cut-cell comparison.

**Pollution regression:** permanent (Sec. 8.3); small in PR CI, full order/resolution/
anisotropy scans nightly.

**End-to-end and physics regressions (nightly):** axisymmetric benchmark — reduction to
classical Grad–Shafranov + p = p₀(V(ψ)) in the appropriate limit, and reduced-vs-3D
axisymmetric agreement where feasible; one island chain with w_c ∝ ε_κ^{1/4}
(Fitzpatrick threshold); current layer with δ ∝ D_u^{1/3} and bounded J∥;
chaotic-layer pressure flattening; nested-surface limit; interpretive→predictive
consistency (a predictive run driven by the recovered S_p^eff returns the interpretive
solution); Picard/Newton agreement; restart round-trip; wout/DESC import regressions;
finite-β code-to-code comparisons (HINT2/SPEC/PIES) where data are available (later).

---

## 23. CI and releases

**PR CI (GitHub Actions):** Ubuntu and macOS (Windows once NGSolve + remec support is
stable there); supported CPython versions with NGSolve wheels. Each PR: clean-environment
installation; ruff format/lint; type checks on the pure-Python API where practical; unit
tests; small manufactured tests; the small pollution test; a small threaded-execution
test; checkpoint round-trip; wheel build + `pip install dist/*.whl` smoke test.

**Scheduled CI:** anisotropy/order scans; full Sovinec measurements; larger M3 tests;
axisymmetric end-to-end; thread-scaling sanity; memory benchmarks; optional ngsxfem
tests; PETSc-branch tests when that branch exists.

**Packaging:** `pyproject.toml` declares `ngsolve` as a binary-wheel dependency within a
tested version range; optional extras `remec[io]`, `remec[xfem]`, `remec[vmec]`,
`remec[desc]`, `remec[petsc]`, `remec[viz]`, `remec[dev]`. A release is not complete
until its wheel installs and passes the smoke test in a clean environment on each
supported platform. Publish to PyPI via trusted publishing on tags; a conda-forge
feedstock MAY follow the first tagged release as a convenience.

---

## 24. Logging, reproducibility, and error handling

Each run produces structured logs (human console + machine JSON): problem summary; mesh
and FE sizes; thread count; coefficient ranges; nonlinear iteration table; linear
iteration counts; residuals by equation; invariant errors; active floors/caps; layer
resolution estimates; timing by block; memory estimate; checkpoint locations; code and
NGSolve versions, git commit, platform. Randomized tests/meshes store their seeds.

Checkpoints MUST include: mesh (or lossless reference), FE order/space definitions, all
accepted state vectors, harmonic basis and flux coefficients, normalization, input
profiles, transport parameters, iteration history, versions/platform/threads, and the
saved iterate's diagnostics — under a schema version; readers MUST reject unsupported
future major versions clearly. HDF5 for structured data; VTK/VTU for visualization only
(never a restart format).

Domain-specific exceptions: `InvalidProfileError`, `MeshQualityError`,
`FluxCompatibilityError`, `UnresolvedLayerError`, `AnisotropyPollutionError`,
`NonlinearConvergenceError`, `CheckpointVersionError`,
`UnsupportedBoundaryPhysicsError`. Never continue silently after: a nonmonotone volume
map; an inverted element; a failed harmonic-flux construction; incompatible net boundary
flux; a singular linear solve; NaN/Inf in state or residual; a profile evaluated outside
its volume interval. Under-resolution warnings are acceptable in exploratory runs;
strict mode turns them into errors.

---

## 25. Staged implementation plan

Each milestone is sized for one reviewable pull request; a coding agent SHOULD implement
one at a time. Do not start a phase before the previous phase's acceptance criteria pass
in CI.

**Phase 0 — repository and conventions.** 0.1 package skeleton (`pyproject.toml`,
importable `remec`, CI on Linux+macOS, NGSolve smoke test, ruff/pytest, `Normalization`
and option dataclasses, developer instructions; acceptance: `pip install -e .` + `pytest`
in a clean env with no compiler/MPI). 0.2 common utilities (block norms, structured
logging, timing context, deterministic config serialization, thread configuration,
checkpoint metadata).

**Phase 1 — anisotropic scalar kernel.** 1.1 isotropic Poisson on `Slab2D` with
manufactured convergence. 1.2 oblique anisotropic K with parallel/perpendicular
diagnostics and order scans. 1.3 pollution benchmark with machine-readable table —
**no coupled work proceeds until measured pollution decreases systematically with order
and refinement.** 1.4 closed-field and island frozen-field tests. 1.5 refactor into the
`AnisotropicDiffusionSolver` interface without changing results.

**Phase 2 — level-set volume and transplant.** 2.1 mollified V_χ with analytic
circle/sphere tests and monotone tabulation. 2.2 profiles + transplant with exact
enclosed-volume and layer-cake tests. 2.3 differentiable map (JVP vs. finite
differences). 2.4 optional ngsxfem cut-cell reference and comparison.

**Phase 3 — M3 kernel.** 3.1 direct-u weak form with all terms on frozen (**B**, p),
with both regularization gradients (∇⊥ default and full ∇; Sec. 9.4) selectable at
runtime and threaded consistently through every D_u term. 3.2 SUPG + manufactured
tests, run for both variants. 3.3 ũ formulation transcribed and verified against
direct-u (both variants). 3.4 D_u^{1/3} layer-scaling demonstration and resolution
requirements (both variants). 3.5 gradient-variant comparison study per Sec. 9.4 and
note §5.5: measured O(ε_J) cross-variant agreement at fixed D_u, common D_u → 0 limit,
and a machine-readable performance/robustness comparison (assembly reuse, linear-solver
behavior, monotonicity, misalignment sensitivity, parallel grid-noise damping) recorded
in `docs/verification.md`; the default remains ∇⊥ unless changed by an ADR citing
these measurements.

**Phase 4 — compatible magnetic kernel.** 4.1 de Rham space/order-pairing tests.
4.2 gauge-fixed curl–curl with manufactured magnetostatics. 4.3 harmonic flux field on
a simple analytic torus. 4.4 divergence-free current projection + diagnostics.

**Phase 5 — reduced end-to-end solver.** 5.1 axisymmetric reduced model per the note's
Sec. 11. 5.2 damped Picard connecting χ → transplant → M3 → current → magnetics.
5.3 Anderson with fallback and history tests. 5.4 staged continuation in pressure
amplitude, D_u, anisotropy. Acceptance: axisymmetric benchmark vs. Grad–Shafranov +
p₀(V(ψ)) within tolerance.

**Phase 6 — 3D fixed boundary.** 6.1 periodic-torus end-to-end benchmark. 6.2 smooth
solid-torus mesh (simple torus, then shaped Fourier boundary, geometry-error report).
6.3 VMEC/VMEC++ reader + initialization. 6.4 DESC reader. 6.5 reproducible finite-β
fixed-boundary stellarator example with Poincaré/isobar/S_p^eff/G(V) diagnostics;
nested-surface case reproduces p = p₀(V(ψ)) to O(ε_κ); island case shows flattening
with measured w_c ∝ ε_κ^{1/4} (nightly).

**Phase 7 — extreme-anisotropy upgrade.** 7.1 literature-derived AP prototype (ADR
first) on the Phase 1 tests. 7.2 closed-field AP verification (anisotropy-independent
or substantially improved conditioning/accuracy). 7.3 AP as an interchangeable χ solver
in Picard. (Independent of Phase 8; may proceed in parallel. Required before any claim
of robust operation at ε_κ ≲ 10⁻⁸.)

**Phase 8 — Newton.** 8.1 pure side-effect-free residual refactor. 8.2 JFNK prototype
with Picard-block preconditioning. 8.3 exact local linearization via symbolic
differentiation. 8.4 nonlocal low-rank V_χ JVP. 8.5 pseudo-transient globalization and
Picard→Newton switchover; Picard/Newton agreement test.

**Phase 9 — PETSc experiment.** Separate branch; never blocks native releases.
Benchmark installation, CI complexity, KSP/SNES robustness, memory, time-to-solution,
one-node scaling, AMS/field-split gains; record the decision in an ADR (criteria in
Sec. 3.2).

**Phase 10 — later physics.** (1) Braginskii transport fields; (2) wall-bounded meshes;
(3) free-boundary vacuum coupling (virtual casing or coupled vacuum region);
(4) open-field/sheath model class (design ADR with physics sign-off before code);
(5) tangent/adjoint sensitivities; (6) two-temperature energy equations; (7) MPI if
demonstrated necessary.

**Release definitions.** `remec 0.1` (first scientifically useful): pip install on
Linux/macOS without compiler/MPI; 2D slab + axisymmetric; high-order anisotropic χ
solver with pollution benchmark; differentiable V_χ + exact transplant to stated
tolerance; complete M3 with SUPG and both regularization-gradient variants (Sec. 9.4);
compatible magnetics + current projection; damped
Picard; restartable checkpoints; D_u/ε_κ scans; documented axisymmetric end-to-end
verification; clear under-resolution warnings. (A 3D demonstration is desirable but not
required if toroidal magnetics are not yet sufficiently verified.) `remec 0.2`: smooth
3D solid torus; VMEC/VMEC++/DESC initialization; Anderson; a reproducible finite-β
fixed-boundary stellarator case; first AP solver. `remec 0.3`: Newton–Krylov +
continuation; PETSc benchmark results; shaped wall domains; initial tangent
sensitivities.

---

## 26. Guidance for AI coding agents

Before changing code, an agent MUST: (1) read this document; (2) read the relevant note
section; (3) identify the current milestone; (4) state which equations and invariants
the change affects; (5) add or update tests before claiming completion.

Agents MUST: make small, reviewable changes; preserve equation labels in
docstrings/comments and include the mathematical formulas for nontrivial forms; use
explicit types on public APIs; keep NGSolve-specific code behind internal modules;
report solver tolerances and residual definitions; add a regression test for every
fixed numerical bug; distinguish algebraic convergence from discretization accuracy;
preserve restart compatibility or increment the schema version; prefer installed-NGSolve
reality over this document's API assertions, recording discrepancies in
`docs/dev_notes.md` (design-level changes require an ADR).

Agents MUST NOT: omit a term because it appears numerically small; replace a tensor
operator by an isotropic approximation except where explicitly sanctioned — the M3
regularization-gradient option of Sec. 9.4 is sanctioned; (M4a)/(M4) is never;
introduce field-aligned coordinates as a hidden assumption; use a histogram volume map
in Newton; claim extreme-anisotropy capability from residual convergence alone; add
PETSc, MPI, JAX, DESC, or SIMSOPT as base dependencies; assemble a dense/global coupled
Jacobian; use direct 3D factorization as the *planned production* solver; silently clip
pressure, field, or coefficients; conflate free-boundary and open-field-line physics;
expose raw NGSolve internals in the public API unnecessarily. When a source or equation
is ambiguous, open a focused issue or ADR rather than making an undocumented choice.

---

## 27. Known risks and mitigations

1. **High-order FEM stays too polluted:** permanent pollution benchmark; p/h refinement
   and graded meshes; the AP milestone; frozen-field cross-verification (incl. the
   Lagrangian Green's-function referee).
2. **Native preconditioners not robust enough:** test BDDC/multigrid/patch smoothers +
   static condensation; anisotropy-aware smoothers; ngsPETSc experiment; isolated
   linear-solver interfaces.
3. **Harmonic flux / gauge in complex topology:** dedicated simple-torus milestone;
   explicit cut surfaces where needed; topology diagnostics; no shaped-stellarator runs
   before validation.
4. **Discrete M2/M3 incompatibility:** projection with monitored correction; escalate to
   a mixed u–J formulation if the correction does not converge.
5. **V_χ near critical levels:** gradient-scaled mollification; monotone inverse;
   cut-cell reference; volume-based sampling; dedicated separatrix tests.
6. **Strongly shaped mesh quality:** robust tets first; geometry-error diagnostics;
   quality gates; later multi-block prism/hex; graded/adaptive refinement.
7. **One-node memory:** iterative production solvers; static condensation; matrix-free
   where practical; controlled work vectors; block profiling; MPI only if demonstrated
   necessary.
8. **Newton complexity:** Picard first; pure residual architecture; JFNK before the
   exact nonlocal Jacobian; pseudo-transient continuation; optional PETSc branch.

---

## 28. References and resources

**Project sources:** the note (`docs/20260814-01_...tex`); this document; ADRs.

**NGSolve:** documentation (docu.ngsolve.org) — TaskManager, H(curl)/H(div) tutorial,
periodic meshes/spaces, nonlinear problems and `AssembleLinearization`; `ngsxfem`
(level-set/cut-cell integration); `ngsPETSc`.

**Anisotropic diffusion:** Günter, Yu, Krüger & Lackner, JCP 209 (2005) 354; Günter,
Lackner & Tichmann, JCP 226 (2007) 2306; Sharma & Hammett, JCP 227 (2007) 123
(monotonicity limiting and its pollution cost); Sovinec et al., JCP 195 (2004) 355
(high-order FE at realistic anisotropy; the pollution measurement); Lozinski, Narski &
Negulescu and Degond–Deluzet–Narski (asymptotic-preserving schemes); Deluzet & Narski,
Multiscale Model. Simul. (2019) (iterated two-field AP); Giorgiani et al., CPC (2020)
(high-order HDG, non-aligned meshes); "Mesh refinement for anisotropic diffusion in
magnetized plasmas" (arXiv:2210.16442 — layers must be resolved before high-order rates
appear); del-Castillo-Negrete & Chacón, PRL 106 (2011) 195004, Phys. Plasmas 19 (2012)
056112; Chacón, del-Castillo-Negrete & Hauck, JCP 272 (2014) 719; Chacón &
Di Giannatale, JCP (2024) (arbitrary-field Lagrangian Green's function). The AP method
actually implemented MUST be pinned to a primary source in an ADR.

**Level sets:** Tornberg & Engquist, JCP 200 (2004) 462; Saye, SISC 37 (2015) A993.
**Nonlinear solvers:** Knoll & Keyes, JCP 193 (2004) 357 (JFNK); Kelley & Keyes,
SINUM 35 (1998) 508 (pseudo-transient continuation).
**Physics/benchmarks:** Braginskii (1965); Fitzpatrick, Phys. Plasmas 2 (1995) 825;
Rechester & Rosenbluth, PRL 40 (1978) 38; Hanson, Nucl. Fusion 55 (2015) (virtual
casing).
**Equilibrium-code interop:** VMEC++ (github.com/proximafusion/vmecpp; classic
`wout_*.nc`), DESC (github.com/PlasmaControl/DESC; native HDF5 + VMECIO), SIMSOPT,
PlasmaPy (test-only), Gmsh (optional).

---

## 29. Final design principle

remec is built as a sequence of independently verifiable mathematical operators, not as
one large equilibrium solver. The central success criterion is not that a nonlinear
iteration returns a field. It is that the returned field satisfies the compatible
magnetic and current constraints, realizes the prescribed pressure-versus-volume profile,
resolves the regularization layers, and has artificial cross-field transport
demonstrably below the physical transport being modeled.

---

## Appendix A — provenance and adjudications

This document merges two independent drafts (an NGSolve-focused plan, "N", and a
requirements-style plan, "G"). Where they agreed — backend, magnetic representation,
Picard→Newton progression, projection step, mollified V_χ, deferred-physics list, PETSc
branch — the agreement is adopted as settled. Where they disagreed, the adjudications
were:

1. **Units:** G's nondimensional core with a `Normalization` object was adopted over N's
   SI-throughout (scale separation would produce badly scaled matrices; the note's knobs
   are dimensionless); N's SI-at-the-boundary adapters retained.
2. **Meshing:** G's position — never use the source equilibrium's flux-coordinate map as
   the remec mesh; robust curved tet/prism first, multi-block hex later — was adopted
   over N's mesh-from-equilibrium-map proposal (axis degeneracy, shear under strong
   shaping, solutions need not retain imported surfaces). Consequence accepted:
   physical-space field transfer with coordinate inversion is required in general at
   initialization; N's deformation-of-a-reference-mesh principle is retained for future
   shape derivatives, and N's concrete B_h recipe is kept as the recommended first
   construction.
3. **Linear solvers:** reconciled — direct factorization is the default at development/
   verification scale and below a configurable threshold (N), while the *planned
   production* 3D algorithm is iterative and direct-3D-at-scale is prohibited (G).
4. **Newton staging:** G's JFNK-first (Stage A) then exact hybrid (Stage B) adopted over
   N's exact-first preference; N's BSpline transplant mechanism retained because it makes
   Stage B's local part nearly free.
5. **Roadmap:** G's finer phase/milestone granularity adopted; amended so Phases 7 (AP)
   and 8 (Newton) are explicitly independent and may proceed in parallel — AP gates
   extreme-anisotropy *claims*, not Newton.
6. **Small-B floor:** G's B_safe adopted, reconciled with N's no-clipping rule by
   classifying it as a smooth, monitored regularization that must be inactive at
   acceptance.
7. **Axisymmetric mode:** G's true-reduced-formulation requirement adopted; N's
   3D-axisymmetric cross-check retained as a verification item (G's own §22.7 agrees).
8. **CI calibration:** N's tiering adopted — the full 10¹⁰ anisotropy scans run
   nightly, not per-PR (G listed them under PR CI).
9. **Additions kept from N absent in G:** interpretive-mode physics diagnostics
   (S_p^eff, level-set variance, G(V) conductance), the optional S_ref
   source-refinement iteration, the Lagrangian Green's-function cross-verification
   referee, decision revisit-triggers in the summary table, and the flux-exactness
   argument for Ψ_t. Additions kept from G absent in N: normative language and the ADR
   process, the de Rham pairing caution, convergence criteria/block-norm rules, residual
   purity requirements, checkpoint schema/versioning, error taxonomy, logging spec, the
   public-API sketch, the risk register, and the release definitions.
