# Verification records

## Milestone 3.3 — transformed utilde formulation for M3

`CurrentContinuitySolver.solve_utilde` implements the note's preferred split
\(u=F(p)+\tilde u\) with homogeneous \(\tilde u=0\) boundary data. It retains the
milestone-3.2 Galerkin/SUPG operator and moves every \(F(p)\) contribution to the
right-hand side of (M3):

\[
\begin{aligned}
L_{M3}(\tilde u)={}&\frac{2}{B_{\rm safe}^3}
\mathbf B\!\cdot\!(\nabla p\times\nabla B)
-F'(p)\,\mathbf B\!\cdot\!\nabla p
+\nabla\!\cdot\!\left(D_uF'(p)\nabla_r p\right)\\
&-\frac{\mu_0F(p)}{B_{\rm safe}^2}\mathbf B\!\cdot\!\nabla p
+\frac{\mu_0D_u}{B_{\rm safe}^2}
F'(p)\nabla_rp\!\cdot\!\nabla p .
\end{aligned}
\]

Here \(L_{M3}\) is exactly the direct-u left-hand-side operator. The runtime-selected
\(\nabla_r=\nabla_\perp\) or \(\nabla_r=\nabla\) is used in both transformed terms
containing \(D_u\). The Galerkin profile diffusion is moved in its symmetric weak form,
\(-\int D_u\nabla_rv\cdot\nabla_rF\), so it exactly matches the direct-u
\(\nabla v\cdot P^2\nabla u\) block even when the smooth floor makes
\(P=I-\mathbf b_{\rm safe}\mathbf b_{\rm safe}^T\) non-idempotent. The SUPG load and
strong-residual diagnostic use the note-literal single-projection
\(+\nabla\cdot(D_uP\nabla F)\), retaining the documented
\(O(B_{\rm floor}^2/B_{\rm safe}^2)\) Galerkin/SUPG convention. The solver reconstructs
physical \(u\), \(\nabla u\), the
note-(M2) current, and (for the full-gradient variant) physical \(J_\parallel/B\) before
reporting diagnostics. The formulation is included in the configuration digest and
structured solve records. `PrescribedCurrentProfile` carries the frozen \(F(p)\) and
\(F'(p)\) coefficient functions plus explicit
\(\nabla\cdot(F'\nabla_\perp p)\) and \(\nabla\cdot(F'\nabla p)\) coefficients. This
avoids NGSolve's silent zero coordinate derivative for GridFunction-backed pressure
gradients; the runtime variant selects the matching divergence.

The manufactured comparison uses the nonconstant divergence-free field from milestones
3.1–3.2, \(p=x+y\), \(F(p)=0.25+0.3p\), and
\(\tilde u_*=\sin(\pi x)\sin(\pi y)\). The complete strong direct-u residual for
\(u_*=F(p)+\tilde u_*\) prescribes the drive independently. Direct-u and reconstructed
utilde solutions are compared on every mesh in
`tests/manufactured/m3_utilde_rates.csv`; the automated gates require both formulations
to clear the standard finest-pair \(p+0.8\) L2-rate threshold and their relative L2
disagreement to remain below \(10^{-10}\).

| Gradient variant | Degree 1 rate | Degree 2 rate | Degree 3 rate | Maximum relative direct/utilde disagreement |
| --- | ---: | ---: | ---: | ---: |
| ∇⊥ | 1.970 | 3.030 | 4.036 | 6.26e-16 |
| full ∇ | 1.968 | 3.040 | 4.033 | 8.64e-15 |

The finest degree-3 reconstructed-utilde L2 errors are 1.272e-6 (∇⊥) and 1.280e-6
(full ∇), equal to their direct-u errors to the displayed precision. Free-DOF relative
residuals over the table remain below 1.04e-15. A separate quadratic-pressure oracle
\(p=x^2+xy+y^2\) keeps every shifted term nonzero. It independently retranscribes the
advection, reaction, and final-correction sources to relative tolerance \(10^{-12}\);
the explicit diffusion coefficient is cross-checked by the direct-u disagreement gate
and by the distinct variant values:

| Gradient variant | Advection source L2 | Diffusion source L2 | Reaction source L2 | Final-correction source L2 |
| --- | ---: | ---: | ---: | ---: |
| ∇⊥ | 7.820e-1 | 2.012e-1 | 1.207e-1 | 2.688e-2 |
| full ∇ | 7.820e-1 | 2.400e-1 | 1.207e-1 | 3.064e-2 |

This oracle also compares the reconstructed note-(M2) current and physical
\(J_\parallel/B\) pointwise with the direct-u solve and checks homogeneous utilde
boundary values. An active-floor scan at \(B_{\rm floor}=10^{-8},0.1,1.0\) raises
the measured floor-activity L2 norm from 1.70e-16 to 1.43e-3 and 1.24e-1 while the
maximum direct/utilde disagreement remains 2.98e-16. Before the exact symmetric weak
shift, the 0.1-floor perpendicular disagreement was 5.28e-6.

Mutation checks confirmed
that deleting the \(-F'(p)\mathbf B\cdot\nabla p\) source increases the L2 disagreement
from roundoff to 1.037e-1, while replacing the selected perpendicular gradient in the
profile flux by the full gradient raises it to 6.346e-3. Reconstructing
\(J_\parallel/B\) from solved \(\tilde u\) instead of physical \(u\) creates a 0.55
pointwise error for both variants. All three mutations turn the suite red.

## Milestone 3.2 — complete-residual SUPG for M3

`CurrentContinuitySolver` now uses the DESIGN §9.1 SUPG form by default, while
`stabilization="none"` retains the milestone-3.1 Galerkin form. For note equation (M3),
the added residual term is

\[
\int_\Omega \tau\,(\mathbf B\!\cdot\!\nabla v)\left[
\mathbf B\!\cdot\!\nabla u-\nabla\!\cdot(D_u\nabla_r u)
+\frac{\mu_0u}{B_{\rm safe}^2}\mathbf B\!\cdot\!\nabla p
-\frac{\mu_0D_u}{B_{\rm safe}^2}\nabla_r u\!\cdot\!\nabla p
-\frac{2}{B_{\rm safe}^3}\mathbf B\!\cdot(\nabla p\times\nabla B)
\right]dV.
\]

The perpendicular variant expands the complete variable-projector divergence as
\(P_{ij}\partial_{ij}u+(\partial_iP_{ij})\partial_j u\),
\(P=I-\mathbf b_{\rm safe}\mathbf b_{\rm safe}^{T}\); the full-gradient variant uses
the element-interior Laplacian. Thus the SUPG residual contains parallel advection,
diffusion and its coefficient derivatives, reaction, drive, and the final
\(D_u\nabla_r u\cdot\nabla p\) correction for both runtime variants.
For degree 1, the element-interior Hessian is identically zero, so the discrete strong
diffusion residual contains only spatial projector derivatives (none in the full-∇
variant); the checked-in strong-residual norm therefore plateaus. This limitation is
explicit rather than hidden in a weaker gate: the bounded stabilization still measures
the expected P1 L² rate, and all degrees use the repository-standard \(p+0.8\) gate.

NGSolve coordinate differentiation silently returns zero for expressions backed by a
`GridFunction`. A varying GridFunction magnetic field therefore must supply its native
2-by-3 `grad(B)` (or a transposed 3-by-2 matrix). The solver constructs
\(\partial_j(B_i/B_{\rm safe})\) from this matrix and rejects a varying field whose
implicit coordinate derivative is silently zero. A unit test compares the resulting
projector divergence with an independent analytic expression.

The single unit-tested parameter function uses separate high-order streamline and
diffusive scales, \(h_s=h_\parallel/p\) and \(h_d=h_K/p\):

\[
\tau=\left[\left(\frac{2|\mathbf B|}{h_s}\right)^2
+\left(\frac{4D_u}{h_d^2}\right)^2\right]^{-1/2}.
\]

It recovers \(h_s/(2|\mathbf B|)\) and \(h_d^2/(4D_u)\) in the respective limiting
cases. On the current isotropic `Slab2D` verification mesh,
\(h_\parallel=h_K/|\mathbf b_{xy}|\), while the diffusive scale remains \(h_K\).
This prevents the diffusive branch from diverging for a nearly out-of-plane field: as
\(|\mathbf b_{xy}|\) decreases from 1 to 1e-3, sampled \(\tau_{\max}\) remains
1.949e-3–1.953e-3 and the stabilized/unstabilized L² errors agree within 1e-6 at the
smallest fraction for both variants. The stabilization is assembled separately as
well as in the total operator; diagnostics record its free-DOF vector norm, the
element-interior strong-residual L² norm, and the sampled minimum/maximum \(\tau\).
The stabilization mode is included in the configuration digest and structured solve
events. It deliberately remains solver-owned rather than part of `RuntimeOptions` while
this frozen M3 kernel has no checkpointed solve state; milestone 3.3 must revisit that
choice before M3 becomes part of a checkpointable coupled run.

The smooth all-terms manufactured solution is
\(u_*=\sin(\pi x)\sin(\pi y)\) on the nonconstant divergence-free frozen field from
milestone 3.1. Its complete strong (M3) residual prescribes the physical drive. The
checked-in table `tests/manufactured/m3_supg_rates.csv` is recomputed within 5% in PR CI:

| Gradient variant | Degree 1 finest rate | Degree 2 finest rate | Degree 3 finest rate |
| --- | ---: | ---: | ---: |
| ∇⊥ | 1.970 | 3.029 | 4.036 |
| full ∇ | 1.968 | 3.039 | 4.033 |

The automated gates require strict error decrease and a finest-pair L² rate above
\(p+0.8\). Separate manufactured tests cover aligned advection with SUPG both on and
off, transverse diffusion, reaction/nonconstant-field terms, and the final correction;
every case is parameterized over both regularization gradients.

Mutation checks: deleting the final
\(\mu_0D_u\nabla_r u\cdot\nabla p/B_{\rm safe}^2\) term from both Galerkin and SUPG
operators raises the dedicated degree-3 L² errors from 1.234e-6 to 4.217e-2 (∇⊥) and
from 1.234e-6 to 4.230e-2 (full ∇). Deleting the strong diffusion divergence alone
reduces the transverse-case assembled stabilization norm from 3.641e-7 to 6.348e-8
(∇⊥) and from 2.131e-7 to 1.822e-8 (full ∇), violating its 5% pinned contract.
Halving the centralized stabilization parameter makes its exact advection-limit test
report 0.025 instead of 0.05. All three mutations turn the suite red.

## Milestone 3.1 — direct-u frozen-field M3 kernel

`CurrentContinuitySolver` implements the unstabilized direct-u weak form of note
equation (M3) on frozen \((\mathbf B,p)\):

\[
\begin{aligned}
\int_\Omega v\,\mathbf B\!\cdot\!\nabla u
&+D_u\nabla_r v\!\cdot\!\nabla_r u
+\frac{\mu_0}{B_{\rm safe}^2}v u\,\mathbf B\!\cdot\!\nabla p\\
&-\frac{\mu_0D_u}{B_{\rm safe}^2}
v\,\nabla_r u\!\cdot\!\nabla p\,dV
=\int_\Omega \frac{2v}{B_{\rm safe}^3}
\mathbf B\!\cdot\!(\nabla p\times\nabla B)\,dV .
\end{aligned}
\]

The runtime choice is `perpendicular` by default, with
\(\nabla_r=\nabla_\perp\), or `full`, with \(\nabla_r=\nabla\). The same selected
gradient reconstructs note equation (M2),
\(\mathbf J=u\mathbf B+\mathbf B\times\nabla p/B_{\rm safe}^2-D_u\nabla_r u\).
For the full-gradient variant the reported physical parallel-current diagnostic is
\(J_\parallel/B=u-(D_u/B_{\rm safe})\mathbf b_{\rm safe}\cdot\nabla u\), not the
auxiliary solved variable alone. The choice is included in `RuntimeOptions`, so it is
present in canonical configuration digests, both structured solve events, and checkpoint
metadata.

The primary physics oracle in `tests/unit/test_current_continuity.py` transcribes the
strong form of (M3) directly with analytic coefficient derivatives on the spatially
varying, solenoidal frozen field used by the assembly check. It chooses
\(u_*=\sin(\pi x)\sin(\pi y)\), then uses the explicit
`magnetic_magnitude_gradient` input to prescribe the exact M3 drive. This is independent
of the implementation's integration by parts and movement of reaction terms into the
bilinear form. On an order-3 mesh with `maxh=0.0625`, the measured L² errors are
1.216e-6 for both variants, below the single-mesh \(10^{-5}\)
gate. This is an exact-solution regression, not a convergence claim; the p/h sweep
belongs to milestone 3.2.

A divergence identity separately constrains the manufactured oracle's one shared
coefficient, \(2/B_{\rm safe}^3\). With the true analytic \(\nabla|B|\), true
\(\nabla\times\mathbf B\), and a compact boundary-vanishing probe, the weak diamagnetic
flux and the M3 drive/curl expression agree to \(10^{-12}\); the identity is nonzero,
so neither the factor 2 nor the denominator power can be absorbed into the injected drive.
The measured correct discrepancy is 3.6e-16; changing \(2\to3\) raises it to 8.32e-2,
and changing \(B_{\rm safe}^3\to B_{\rm safe}^2\) raises it to 2.81e-1. The test then
ties this independent certificate to the implementation by comparing the certified
drive's L² norm with the solver's assembled `m3_drive_l2` diagnostic to relative
tolerance \(10^{-12}\). Coordinated implementation/assembly/oracle mutations now shift
that diagnostic by +50% for \(2\to3\) and +163% for
\(B_{\rm safe}^3\to B_{\rm safe}^2\), so both are caught end to end.

A secondary algebraic assembly check mirrors the intended weak form and verifies that
the direct solve satisfies it. It is not treated as an independent physics oracle. Its
shared order-2 frozen-coefficient unit-square field is exactly divergence-free, and every
source/reaction component is nonzero:

| Gradient variant | Free-DOF relative residual | M3 drive L² | Reaction L² | Final correction L² |
| --- | ---: | ---: | ---: | ---: |
| perpendicular | 3.19e-17 | 4.246e-1 | 1.615e-2 | 1.239e-2 |
| full | 2.12e-17 | 4.246e-1 | 1.467e-2 | 1.232e-2 |

Both residuals are below the automated \(10^{-11}\) gate. Pointwise tests independently
reconstruct all three M2 current components and the full-gradient \(J_\parallel/B\)
formula to absolute tolerance \(10^{-12}\). The shared smooth floor reports L² activity
1.70e-16 and sampled minimum physical field magnitude \(\sqrt{5}=2.236\). A deliberate
floor of 1.0 raises the activity to 1.241e-1 while leaving the physical minimum unchanged,
so the diagnostic cannot pass as a hard-coded zero.

Mutation checks: changing the drive factor \(+2\) to \(-3\) and flipping the reaction
sign in both the implementation and the mirrored weak assembly check leaves that
secondary check green, but the strong-form oracle fails with L² errors 1.287
(`perpendicular`) and 1.284 (`full`). Deleting the final
\(\mu_0D_u\nabla_r u\cdot\nabla p/B_{\rm safe}^2\) term from the implementation also
makes the algebraic assembly check fail. Reconstructing M2 with a gradient different
from the one selected for M3 makes the pointwise current contract fail conspicuously.

## Milestone 2.4 — optional sharp cut-cell reference

`CutCellVolumeReference` is the optional `remec[cutcell]` implementation of the note's
`(M4b)` sharp reference (Sec. 6), using the sub-cell method in Sec. 8.1:

\[
V_\chi(\hat\chi) = \int_\Omega H(\chi-\hat\chi)\,d\Omega.
\]

It uses `xfem.lsetcurv.LevelSetMeshAdaptation` to map its piecewise-linear cut geometry
to the supplied high-order level set, then integrates its positive domain. It is a
verification/final-evaluation route rather than a replacement for the differentiable
`(mollified_V)` map: it intentionally provides no `(V_derivatives)` JVP. The optional
dependency is installed on five CI matrix legs; the Ubuntu/Python-3.10 leg runs the full
test suite from `remec[dev]` without it, exercising the optional-import skip path. The
smoke jobs separately exercise the minimal wheel installation.

The manufactured circle in the unit-square quadrant has the exact sharp volume
\(\pi(0.6^2-\hat\chi)/4\). The measured zero-level results in
`tests/manufactured/cutcell_circle_rates.csv` are:

| Triangles | Sharp cut-cell error | Cut-cell rate | Mollified–sharp difference | Difference rate |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 4.752e-6 | — | 3.202e-3 | — |
| 512 | 2.971e-7 | 3.999 | 8.016e-4 | 1.998 |
| 2048 | 2.390e-8 | 3.636 | 2.004e-4 | 2.000 |

The automated contract requires every sharp-reference rate to exceed 3.5 and every
mollified-reference difference rate to exceed 1.9. It also checks sharp analytic volumes
at four levels, monotonicity, and the unit-square total volume. Mutation checks confirmed
that replacing the positive cut domain by its complement, or forcing the geometry mapping
to first order, makes both manufactured tests fail.

## Milestone 2.3 — differentiable volume map

`MollifiedVolumeMap.jvp` implements the nonlocal contribution required by
`DESIGN.md` §12.6 and the note's `(V_derivatives)` equation. At requested level values
it applies

\[
\delta V_\chi^\varepsilon(\hat\chi)
=\sum_i H'_{\varepsilon_i}(\chi_i-\hat\chi)w_i\,\delta\chi_i.
\]

ADR 0003 selects the frozen-width quasi-Newton reading: gradient-scaled mollifier
widths are held fixed at the linearization point, then rebuilt only after an iterate is
accepted. This follows the explicit `H'_epsilon w_i` action displayed in the note but
is not the derivative of a functional that simultaneously rebuilds those widths. The
action directly evaluates the smooth quadrature functional rather than differentiating
the volume-uniform PCHIP table, avoiding the ill-conditioned near-step direction. The
latter continues to provide monotone values and inverses for the M4b composition.

The JVP accumulates one level at a time, using O(N_quad + N_levels) working memory;
it does not materialize a dense quadrature-by-level matrix in a Krylov matvec.

`tests/unit/test_differentiable_volume_map.py` uses 4,096 manufactured unit-interval
quadrature samples with constant gradient and spatial size, so the frozen-width contract
isolates the `H'_epsilon w_i` term from width variation. At levels 0.23, 0.37, 0.61,
and 0.74, its central finite difference with step \(10^{-6}\) has maximum absolute
discrepancy \(4.51\times10^{-11}\). It also verifies the default tabulation-level JVP
path and the exact frozen-width identity \(\delta V[1]=\rho=-dV/d\hat\chi\).

A separate variable-gradient manufactured case rebuilds widths on both sides of the
finite difference. Its measured frozen-versus-live relative discrepancy is
\(1.24\times10^{-4}\), below the \(2.0\times10^{-4}\) O(epsilon) quasi-Newton
regression bound. This test intentionally distinguishes the selected frozen-width
functional from the unselected live-width derivative; it does not call them equal.

Mutation check: replacing the `H'_epsilon` surface weights by zero makes all four JVP
values zero while the independently evaluated finite-difference values range from
\(-2.81\times10^{-1}\) to \(2.96\times10^{-1}\), so the contract fails for the
intended omitted nonlocal derivative.

## Milestone 2.2 — profiles and the M4b transplant

`VolumeProfile` supplies analytic and tabulated non-increasing (p_0(V)) profiles;
the latter permits the intentional edge-vacuum plateau. `TransplantedProfile` implements
note equation (M4b),

\[
p(\mathbf r)=p_0\!\left(V_\chi(\chi(\mathbf r))\right).
\]

It validates profile monotonicity and the complete [0, (V_\Omega)] interval,
keeps the result within the prescribed pressure bounds, and exports the local monotone
composition as an NGSolve degree-one `BSpline`. Thus NGSolve can form the local chain
rule term (g'(\chi)\delta\chi); milestone 2.3 will add the nonlocal
(p_0'(V_\chi)\delta V_\chi) contribution.

`extract_ngsolve_quadrature` is the FEM bridge to the array-backed
`MollifiedVolumeMap`. At each mapped integration point it uses
(w_i= w(\mathrm{ip}_i)|\det J_i|), (h_i=|\det J_i|^{1/d}), the coefficient value,
and the supplied analytic gradient. This retains curved-element geometry rather than
forming a discontinuous histogram.

`tests/unit/test_profiles_transplant.py` directly measures the pressure-superlevel
volume of the transplanted quadrature field on seven targets (absolute tolerance
(2\times10^{-3})), rather than algebraically inverting (p_0). It also checks pressure
bounds and monotonicity, and the note's layer-cake relation

\[
\int_\Omega\varphi(p)\,dV=\int_0^{V_\Omega}\varphi(p_0(V))\,dV.
\]

For (p_0(V)=1-V^2) on the manufactured unit interval, an eight-function compact
quadratic B-spline family spanning the full pressure interval has residuals between
(-9.069\times10^{-5}) and (2.600\times10^{-5}), versus a (3\times10^{-3})
threshold. A radial circle data set independently evaluates the first layer-cake
moment (\int p\,dA=2) within 0.02. The NGSolve test extracts a
unit-square quadrature total of 1.0, verifies that the center pressure exceeds the
edge pressure, and measures mean pressure (0.5\pm0.03).

Mutation check: replacing (V_\chi(\chi)) by a constant zero volume causes direct
superlevel-volume realization to fail on all seven targets (all measure 1.0), every
compact B-spline layer-cake moment to fail (maximum residual 0.218), the NGSolve
center/edge ordering to collapse, and the independent radial moment to become 4 rather
than 2. This demonstrates that a test failure reflects omission of the M4b volume
composition, not a solver residual.

## Milestone 2.1 — mollified level-set volume map

`MollifiedVolumeMap` implements the note's equation `(mollified_V)`,

\[
V_\chi^\varepsilon(\hat\chi)=\sum_i w_i H_{\varepsilon_i}(\chi_i-\hat\chi),
\qquad
\varepsilon_i=c h_i\max(|\nabla\chi_i|,\mathrm{gradient\_floor}),
\]

using the compact, moment-matched smooth Heaviside
\(H_\varepsilon\). The gradient-scaled width is deliberately local, so the
regularization has a fixed spatial width rather than an inconsistent fixed width in
level-set-value space. A named `minimum_gradient_fraction` applies a robust fraction
of the sampled maximum gradient at critical points, avoiding a zero-width mollifier;
the number of floored samples is reported. A mandatory co-area consistency diagnostic
warns when the tabulated derivative exceeds its configurable relative-error tolerance.
`coarea_density` implements `(V_derivatives)`,
\(-dV_\chi^\varepsilon/d\hat\chi=\sum_i w_iH'_{\varepsilon_i}\), and a monotone
PCHIP table exposes both `V_χ(χ̂)` and its stable inverse `χ̂(V)`. The map returns the
endpoint identities exactly.

`tests/unit/test_level_set_volume.py` evaluates \(\chi=0.6^2-r^2\) on
Gauss quadrature in \([-1,1]^2\) and \([-1,1]^3\). At the zero level, the exact
circle and sphere volumes are \(\pi(0.6)^2\) and \(4\pi(0.6)^3/3\). The measured
relative errors are 5.18e-4 and 7.76e-4; the independent analytic co-area densities
\(\pi\) and \(2\pi(0.6)\) agree within 1.70e-2 and 2.56e-3. The test also verifies
the endpoint identities, strict monotonicity, a tabulation uniformly spaced in
enclosed volume, unnormalized endpoint residuals, and agreement between the tabulated
derivative and independently assembled mollified co-area density. It additionally
checks the inverse round-trip `inverse_level(volume(level))` within 0.04 over the
analytic circle branch.

The manufactured sphere resolution table is checked in at
`tests/manufactured/mollified_sphere_volume_rates.csv`:

| Quadrature order | Absolute volume error | Adjacent rate |
| ---: | ---: | ---: |
| 24 | 1.2347e-2 | — |
| 48 | 2.9248e-3 | 2.078 |
| 96 | 7.0256e-4 | 2.058 |

The measured rates are consistent with the expected \(O(\varepsilon^2)\) behavior of
the mollified construction. The test recomputes all table errors and rates and rejects
either rate below 1.9. This is a quadrature-resolution study of the map itself; it is not a claim
about a future FEM solve's convergence rate.

Mutation checks: replacing the gradient-scaled width by a fixed chi-space width makes
the circle co-area-density assertion fail (3.328 versus \(\pi\), outside the 3% bound).
Replacing `H'_ε` by zero makes both circle and sphere co-area checks fail. The PCHIP
derivative is independently checked against the co-area density, and the raw endpoint
residuals are checked before endpoint normalization, so neither a wrong spline slope
nor a removed mollified-Heaviside assembly can pass through endpoint identities alone.
The plateau test rejects the former `1e-12` gradient guard: 29% zero-gradient samples
must have a width of at least `1e-4` and a finite density below `3e3`.

NumPy is declared explicitly because §12.2 requires the quadrature samples to be
vectorized. Its temporary `<2.5` bound preserves the project-wide Python-3.10 mypy
target: NumPy 2.5's stubs use Python-3.12-only syntax even when checked by a later
interpreter. The bound can be lifted when those stubs support remec's stated target.
Milestone 2.2 owns the NGSolve quadrature-extraction pass and will adapt this
array-backed map to the complete §12.1 solver-facing interface; the nonlocal JVP remains
Milestone 2.3. That milestone must also calibrate the present `1e-3` critical-gradient
fraction against tabulation spacing before reusing the mollified surface weights in a
Newton derivative; the current map reports its floored-sample count and warns whenever
its mandatory tabulation/co-area check exceeds the configured tolerance.

## Milestone 1.5 — `AnisotropicDiffusionSolver` strategy interface

The public `AnisotropicDiffusionSolver` is the Phase-1 `StandardCG` strategy for
note equation (M4a),

\[
\int_\Omega \nabla v\mathbin\cdot
\left[\kappa_\perp I+(\kappa_\parallel-\kappa_\perp)
\mathbf b_{\rm safe}\mathbf b_{\rm safe}^{T}\right]\nabla\chi\,dV
=\int_\Omega vS_{\rm ref}\,dV.
\]

It exposes `solve`, `apply`, `build_preconditioner`, `diagnostics`, and the
rank-one `measure_sovinec_pollution` entry point while keeping NGSolve objects
behind `remec.fem`. All three Phase-1 paths use the spatial M4a tensor assembly
where applicable; the rank-one path retains its historical quadrature and
machine-readable pollution table. The public result contains only scalar
diagnostics, including a direct sparse-Cholesky inverse identity defect below
1e-11, not NGSolve meshes, fields, or matrices.

For a unit direction, this is algebraically the note's projected M4a form. At
an active smooth floor it intentionally retains the displayed tensor: applying a
second perpendicular projection with \(|\mathbf b_{\rm safe}|<1\) would not be
idempotent and would implement a different operator. ADR 0002 records the
redundant-normalization correction that preserves the original Sovinec baseline
bit-for-bit.
`test_sovinec_common_path_preserves_the_bit_exact_reference_solution` is the
same-process regression that demonstrates this acceptance criterion.

For production safety, `assess_pollution` implements `DESIGN.md` §8.3's default
criterion \(\kappa_{\perp,\rm num}<0.1\kappa_\perp\): it emits
`AnisotropyPollutionWarning` normally and raises `AnisotropyPollutionError` in
strict mode. `assess_floor_sensitivity` applies §6 to paired observables at two
smooth field-floor values; its default 1% tolerance warns (or raises in strict
mode) for a material difference on the observable's own scale. The test suite
demonstrates the rank-one κ⊥=0 unsafe measurement and a 100% difference at an
O(1e-3) observable, as well as safe counterparts.

Mutation checks: removing the M4a tensor contrast fails the new public island
manufactured solve (central response 2.014 rather than 1) as well as the frozen-field
topology regressions; it also makes rank-one Sovinec singular. Rotating the Sovinec
field from tangent to normal makes its source-tangency assertion fail (measured
\(4.93\), required below \(10^{-12}\)). These are physical/discretization
protections rather than residual checks.

## Milestone 1.4 — closed-field and analytic-island frozen fields

This milestone extends the frozen-field verification of note equation (M4a) to a
finite-anisotropy axis and to an independent analytic magnetic island.  The internal
spatial-field weak form is

\[
\int_\Omega \kappa_\perp\nabla\chi\cdot\nabla v
+(\kappa_\parallel-\kappa_\perp)
(\mathbf b_{\rm safe}\cdot\nabla\chi)
(\mathbf b_{\rm safe}\cdot\nabla v)\,dV
=\int_\Omega vS_{\rm ref}\,dV,
\]

with
\(\mathbf b_{\rm safe}=\mathbf B/
\sqrt{\mathbf B\cdot\mathbf B+B_{\rm floor}^2}\), as required by
`DESIGN.md` §6. Thus
\(K=\kappa_\perp I+(\kappa_\parallel-\kappa_\perp)
\mathbf b_{\rm safe}\mathbf b_{\rm safe}^{T}\) remains positive definite and
smoothly becomes isotropic at an exact field null. The solver reports the separate
parallel and perpendicular M4a energies, the free-DOF algebraic residual, and
\(\int_\Omega(1-|\mathbf b_{\rm safe}|^2)^2dV\) as the floor-activity diagnostic.

The closed-field scan uses the translated Sovinec flux
\(\psi=\sin(\pi x)\sin(\pi y)\),
\(\mathbf B=(\partial_y\psi,-\partial_x\psi)\),
\(S_{\rm ref}=\psi\), \(\kappa_\parallel=1\), and
\(B_{\rm floor}=10^{-6}\). Because
\(\mathbf b_{\rm safe}\cdot\nabla\psi=0\) even at finite floor, the exact solution is
\(\chi=\psi/(2\pi^2\kappa_\perp)\). The permanent scheduled table
`tests/manufactured/closed_field_anisotropy_scan.csv` contains all 27 combinations of
degrees 1–3, 32/128/512 elements, and
\(\kappa_\perp/\kappa_\parallel=10^{-1},10^{-2},10^{-3}\). The measured effective
diffusivity minus the physical \(\kappa_\perp\) is positive and decreases strictly
under every adjacent order and mesh refinement. On the finest degree-3 mesh,
\(\kappa_{\perp,{\rm num}}/\kappa_\perp\) is respectively
\(4.8\times10^{-6}\), \(1.2\times10^{-5}\), and \(6.4\times10^{-5}\).
The fast PR test samples the degree-3, 128-element slice, pins its measured central
amplitude and numerical diffusivity, and requires the relative L² error below
\(6.0\times10^{-4}\) throughout the anisotropy axis (measured maximum
\(5.54\times10^{-4}\)). The Laplacian eigenvalue used in both fast and scheduled
metrics is derived from the differentiated flux coefficient rather than hard-coded.

The checked-in scan uses `ngsolve.dx(bonus_intorder=20)`. Raising the rule from
14 to 20 changed the cancellation-sensitive degree-3 numerical diffusivity by at
most 1.21%, while changing 20 to 26 moved it by at most 0.41% across all nine
degree-3 mesh/anisotropy combinations. The table was regenerated at 20. Thus the
quadrature sensitivity is below 1% at the retained rule and is not the dominant
source of the strict order/refinement trends (whose adjacent margins are much larger).
The central amplitude and effective diffusivity are the solver outputs pinned at
relative tolerance \(10^{-5}\). Numerical diffusivity is their cancellation-derived
difference from physical \(\kappa_\perp\), which amplifies platform-level amplitude
variation; it is therefore constrained by positivity and by the strict order/refinement
trends rather than by a misleading table-equality pin. Those trend assertions remain
the acceptance test for the derived metric.

The independent island is generated by

\[
\psi_I=\tfrac12(y-\tfrac12)^2
+\frac{\cos(2\pi x)}{(2\pi)^2},\qquad
\mathbf B_I=(\partial_y\psi_I,-\partial_x\psi_I).
\]

It has an O-point at \((1/2,1/2)\) and X-points at \((0,1/2)\) and
\((1,1/2)\). The manufactured solution is
\(\chi=\sin(\pi x)\sin(\pi y)\), with its source obtained analytically from the
M4a tensor at \(\kappa_\parallel=10\), \(\kappa_\perp=1\), and the deliberately
resolved null floor \(B_{\rm floor}=0.05\). This verification field has nonzero
normal component on the horizontal Dirichlet boundaries, so it is an island chain
with boundary-terminating exterior field lines, not a closed fixed-boundary magnetic
configuration. Because the manufactured source is derived from the floored operator,
the floor is intentionally part of the exact verification problem; this test does not
claim that its visibly active floor is negligible in the §6 production-observable
sense. The milestone 1.5 `Next:` follow-up records that separate sensitivity study. The island
manufactured source is also sign-changing (a 201×201 point scan gives approximately
\([-269.7,177.2]\)); like the milestone 1.2 direction-sensitive source, it is a linear
verification device rather than an admissible positive reference source. The rate
table is `tests/manufactured/analytic_island_rates.csv`.

| Degree \(p\) | Elements (coarse → fine) | L² rate | K-energy rate |
| --- | ---: | ---: | ---: |
| 1 | 200 → 800 | 1.810 | 0.964 |
| 2 | 200 → 800 | 3.132 | 1.983 |
| 3 | 200 → 800 | 4.053 | 2.996 |

`test_analytic_island_manufactured_convergence` requires L² rate at least
\(p+0.8\), K-energy rate at least \(p-0.2\), finite evaluation at every O/X null,
positive recorded floor activity, and a free-DOF relative residual below \(10^{-11}\).
The degree-1 L² rate 1.810 has only 0.010 headroom above its 1.8 gate, so milestone
1.5 must preserve the recorded table when reconciling the spatial-field forms.
The assembly uses `bonus_intorder=20`; energy/error integrals use order 20 and the
sharper floor-activity integral uses order 40. Its value changes by less than
\(5\times10^{-7}\) relatively over the three meshes and is pinned by the CSV.
The 27-case Cartesian scan is marked `slow`; `.github/workflows/nightly.yml` runs
`make test-full` every day and on manual dispatch after the workflow reaches the
default branch, while PR CI keeps the smaller finite-anisotropy slice plus all nine
island convergence solves. The complete 34-test `make test-full` suite was run locally
on this branch before submission.

Mutation checks confirmed that deleting the
\((\kappa_\parallel-\kappa_\perp)\mathbf b\mathbf b^T\) contribution fails the
closed-field regression and all island orders, and deleting \(B_{\rm floor}^2\) from
the safe norm fails all three island convergence regressions. These failures are
discretization errors, not residual failures: the mutated direct solves still converge
algebraically.

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
\(\int_\Omega(\mathbf b\cdot\nabla\psi)^2\,dV<10^{-12}\), so unit normalization
and the 90-degree rotation of \(\mathbf b\) relative to the actual source gradient
are checked structurally rather than only through the recorded CSV. A scaling case
with \(\kappa_\parallel=10\) and \(Q_0=3\) checks the expected amplitude,
effective-diffusivity, and dimensionless-ratio scalings. The implementation
derives \(\nabla\psi\) and the \(2k^2\) Laplacian eigenvalue from the same
coefficient function used as the source, and the test independently checks the
eigenvalue \(2\pi^2\). CSV comparisons use relative tolerance \(10^{-5}\), about
100 times the largest Linux/macOS variation measured in review. Every acceptance
row also requires \(\kappa_{\perp,\mathrm{num}}/\kappa_\parallel<0.2\), which
rejects an isotropic substitution independently of the recorded values.

The regression table is recorded with `ngsolve.dx(bonus_intorder=6)`. Because
\(1/|\nabla\psi|\) is non-polynomial near the isolated field nulls, changing the
quadrature rule changes the pinned amplitudes and requires regenerating the CSV.
Reviewing with `bonus_intorder=14` shifted amplitudes by at most 0.573% while
leaving the finest-pair rates essentially unchanged (1.977, 3.936, and 5.877),
confirming quadrature is not the source of the observed convergence trend.

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
