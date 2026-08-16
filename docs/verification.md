# Verification records

> **Model-revision notice (2026-08-15).** Records for milestones 3.3–3.4 describe the
> former prescribed `u=F(p)+ũ` parameterization. The measurements remain valid for the
> unconstrained/G≡0 M3 kernel and local layer scaling, but `F(p)` does not prescribe the
> physical mean current at finite D_u. Under the authoritative August 15 note, these
> equivalence records are the negative control for `DESIGN.md` §9.2. Production current
> input is cumulative I₀(s), s=V/V_Ω∈[0,1]; the solver must determine G(s) from (M3b).
> No statement below calling the F-shift “preferred” should be read as current design.
> Likewise, milestone 2.2 records the former dimensional-V `VolumeProfile`; its
> numbers remain valid. Milestone 3.5 migrated the public contract and layer-cake
> oracle to p₀(s) and the factor V_Ω∫₀¹·ds.

## Milestone 4.1 — tetrahedral de Rham order pairing

The first compatible-magnetics milestone establishes the NGSolve order convention
needed for note equation (M1).  On tetrahedra the exact affine polynomial complex is

\[
H^1_{p+1}\xrightarrow{\nabla}H(\mathrm{curl})_p
\xrightarrow{\nabla\times}H(\mathrm{div})_{\max(p-1,0)}
\xrightarrow{\nabla\cdot}L^2_{\max(p-2,0)}.
\]

This is deliberately not four spaces constructed with equal `order` arguments.  The
factory rejects non-tetrahedral element families rather than silently applying these
offsets to NGSolve's different tensor-product convention.

For every row below, deterministic random coefficients excite every degree of freedom
in each source space.  An independently assembled L² mass projection measures whether
the gradient, curl, or divergence lies in the next space; the projected fields are then
differentiated again to measure both `curl(grad)` and `div(curl)`.  The alternating
global dimension is exactly one on the contractible cube in every row, as required for
the full complex including the constant scalar kernel.

| cells/axis | tetrahedra | base p | H¹/HCurl/HDiv/L² orders | max mapping defect | curl(grad) defect | div(curl) defect |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| 1 | 6 | 0 | 1/0/0/0 | 2.65e-16 | 1.52e-15 | 8.34e-16 |
| 1 | 6 | 1 | 2/1/0/0 | 4.15e-16 | 1.98e-15 | 1.40e-15 |
| 1 | 6 | 2 | 3/2/1/0 | 1.14e-15 | 1.26e-14 | 1.23e-15 |
| 1 | 6 | 3 | 4/3/2/1 | 3.42e-15 | 5.57e-14 | 1.15e-14 |
| 2 | 48 | 0 | 1/0/0/0 | 3.48e-16 | 3.97e-15 | 1.95e-15 |
| 2 | 48 | 1 | 2/1/0/0 | 4.42e-16 | 4.36e-15 | 2.34e-15 |
| 2 | 48 | 2 | 3/2/1/0 | 1.13e-15 | 2.31e-14 | 2.70e-15 |
| 2 | 48 | 3 | 4/3/2/1 | 3.11e-15 | 1.03e-13 | 2.53e-14 |

All defects clear the automated \(10^{-12}\) roundoff gate.  The complete
machine-readable record, including all four dimensions and the Euler characteristic,
is `tests/manufactured/de_rham_pairing.csv`.

NGSolve's Piola mappings preserve the magnetic part of the complex on curved geometry:
an exploratory order-3 OCC ball measured gradient/curl mapping and both successive-
derivative identities below \(6.95\times10^{-13}\).  Ordinary scalar NGSolve `L2`,
however, is not the density-mapped terminal space on a curved element, so projecting a
general `div(HDiv)` field into it is not a roundoff identity (measured relative defects
0.23--0.32).  Curved-mesh magnetic verification must therefore form
`B_h = ng.curl(A_h)` and measure its symbolic coordinate-derivative trace; the nested
call `ng.div(ng.curl(A_h))` is not supported in NGSolve 6.2.2606.  The curved-current
terminal-space decision is pending in ADR 0004; the API facts are recorded in
`docs/dev_notes.md` for milestones 4.2 and 4.4.

Mutation checks confirmed that replacing the offsets by equal `order` arguments fails
the order/dimension contract, and deleting the tetrahedral-family guard fails the
non-tetrahedral rejection test.

## Milestone 3.7 — constrained gradient-variant comparison

The transverse and full-gradient current-viscosity closures were compared with the
same bordered (M3)--(M3b) solver on shared frozen
$(\mathbf B,p,s,I_0,\mathrm{drive})$ states. Both reconstruct the physical (M2)
current

\[
\mathbf J=(G+\tilde u)\mathbf B
+\frac{\mathbf B\times\nabla p}{B_{\rm safe}^2}
-D_u\nabla_r\tilde u,
\]

and the cumulative current in every row is evaluated independently from that current.
Thus the study preserves DESIGN §5 invariant 4 rather than comparing only M3
residuals. It also records the sampled minimum field and floor activity (invariant 5),
and every four-shell partition spans at least 4.796 local cells/mollifier widths in the
resonant scan and 6.812 in the misaligned scan (invariant 6).

### Fixed-state regular limit and resonant layer

The resonant benchmark uses
$\mathbf B=(0.01,6(x-1/2),2)$ on a periodic-$y$ $24\times16$ structured triangular
mesh and the nonuniform toroidal-angle gradient
$\nabla\phi=(0.4,0.3\sin(2\pi y),1)$. This makes both the multiplier and
regularizing terms visible to the independent toroidal-current constraint. Its fixed
drive is $\sin(2\pi y)+0.05\sin(10\pi y)$; the smaller fifth harmonic measures
parallel grid-noise transfer without changing the current target. The small nonzero
$B_x$ makes both $\mathbf B\cdot\nabla s$ and $\mathbf B\cdot\nabla p$ nonzero, so
the bordered $P$ block couples the constrained multiplier back into $\tilde u$ rather
than reducing to the unconstrained kernel. The same analytic cumulative
$I_0(s)=0.04s+0.01s(1-s)$, pressure gradient, volume coordinate, drive, shell grid,
mesh, and edge value are held fixed throughout the $D_u$ scan. With reference length
one, the benchmark convention $\bar B=\min|\mathbf B|=\sqrt{4.0001}$ gives
$\epsilon_J=D_u/\sqrt{4.0001}$. The resulting order-unity normalized coefficient is
specific to that conservative $\bar B$ convention and is not assigned physical
significance.

| $D_u$ | $\epsilon_J$ | $\epsilon_\kappa/\epsilon_J$ | relative $u_\perp-u_{\rm full}$ L² | difference / $\epsilon_J$ |
| ---: | ---: | ---: | ---: | ---: |
| 0.04 | 0.0199998 | 0.25 | 2.0207e-2 | 1.0104 |
| 0.02 | 0.0099999 | 0.50 | 1.0489e-2 | 1.0489 |
| 0.01 | 0.0049999 | 1.00 | 5.1913e-3 | 1.0383 |

The adjacent decay rates are 0.946 and 1.015, so the cross-variant disagreement is
measured $O(\epsilon_J)$ on this genuinely fixed state. Each variant realizes the same
input current with independent M3b relative residual below $2.99\times10^{-17}$.
However, this fixed-$\epsilon_\kappa$ target is **not** a strict admissible
$D_u\to0$ sequence under note §5.5(vii): as
$\epsilon_\kappa/\epsilon_J$ rises from 0.25 to 1.00, the maximum shell
$|\langle\tilde u\rangle|$ rises from about 0.00874 to 0.03539. The checked test pins
$D_u\max_s|\langle\tilde u\rangle_s|$ to within a factor 1.02 (about
$3.49\times10^{-4}$ to $3.54\times10^{-4}$), demonstrating the observed $1/D_u$
growth instead of calling it bounded. The note's vanishing-mean condition requires
$\epsilon_\kappa\to0$ alongside $D_u$; milestone 3.6 records that admissible-family
behavior. The smallest row here is already poorly separated
($\epsilon_\kappa/\epsilon_J=1$), so this table supports only the measured
cross-variant $O(\epsilon_J)$ statement, not convergence to a regular common physical
limit. The multiplier-current norm nevertheless falls from about
$1.55\times10^{-2}$ to $4.06\times10^{-3}$, with adjacent rates 0.983 and 0.942.

The actual multiplier-to-$\tilde u$ forcing is nonzero. In the resonant scan the
coupling is carried by the advection term
$-G'\mathbf B\cdot\nabla s$, whose L² norm is
$3.86\times10^{-3}$ to $4.06\times10^{-3}$; the reaction share is only
$2.65\times10^{-12}$ to $3.04\times10^{-12}$ because this benchmark uses
$\mu_0=10^{-8}$. The misalignment table separately pins both contributions, with
advection norms 1.92--2.24 and reaction norms 0.0109--0.0132, so the reaction path is
not inferred from the numerically negligible resonant share. The smooth-floor
activity, evaluated directly as the L² norm of
$B_{\rm floor}^2/(B^2+B_{\rm floor}^2)$, is $1.735\times10^{-17}$ with sampled
$\min|\mathbf B|=2.000025$; a separate $B_{\rm floor}=2$ case exceeds 0.25 and
proves the diagnostic responds when the floor is active. The toroidal
regularizing-current norm is nonzero and variant-distinct: at $D_u=0.02$ it is
0.01109 (∇⊥) and 0.009955
(full ∇), so this benchmark does not hide the (M2) term from (M3b).

At $D_u=0.02$, the transverse/full fundamental-harmonic FWHM values are 0.41050 and
0.40702, a 0.85% difference; they span 9.85 and 9.77 normal elements. Each radial
amplitude has exactly one turning point, so neither closure adds a spurious layer
oscillation. The full-gradient closure reduces the injected fifth-to-fundamental
parallel-noise ratio from $6.8031\times10^{-3}$ to $6.4977\times10^{-3}$, a 4.49%
reduction. All three diffusivities clear six FWHM cells for both variants. These
FWHM/noise observables use the physical $J_\parallel/B$ evaluator. Replacing that
evaluator by auxiliary $u$ in the full-gradient rows changes the noise transfer by
3.37%, 0.501%, and 0.314% at $D_u=0.04$, 0.02, and 0.01 respectively; deleting the
correction therefore fails the largest-$D_u$ CSV pin. The exact $J_\parallel/B$
formula is independently guarded by the milestone-3.6 pointwise current test. The
complete scan is checked in as
`tests/manufactured/m3_gradient_du_limit.csv`.

### Field-misalignment and solver-cost measurements

The second benchmark uses a constant in-plane field at 22.5 degrees. The structured
triangle edges lie at 0, 45, and 90 degrees, so the field bisects the nearest two edge
directions and is deliberately maximally misaligned within that mesh family. A
zero-degree field on the same meshes is the aligned control. On the
$20\times20\to28\times28$ refinement, aligned physical-$u$ coarse-to-fine relative L²
changes are $1.4068\times10^{-2}$ (∇⊥) and $1.2687\times10^{-2}$ (full ∇); the
misaligned values are $1.9625\times10^{-2}$ and $1.6215\times10^{-2}$. Thus
misalignment amplifies h-sensitivity by 1.395 and 1.278 respectively, rather than the
comparison merely measuring ordinary refinement error. The fine-grid cross-variant
differences are $3.9467\times10^{-2}$ aligned and $2.9210\times10^{-2}$ misaligned.
The larger aligned cross-variant difference does not mean its mesh sensitivity is
worse: misalignment is measured by the within-variant coarse/fine amplification above,
not by the absolute distance between closures. Rotating the field for the aligned
control also changes $\mathbf B\cdot\nabla p$ and $\mathbf B\cdot\nabla s$ by 7.6%, so
the absolute amplification includes a small state change; the ratio between the two
variant amplifications, $1.395/1.278\simeq1.09$, is the comparative signal.
The multiplier-current norms are also variant-distinct in both controls, and all
independently evaluated M3b residuals remain below $7.01\times10^{-17}$. These rows
are in
`tests/manufactured/m3_gradient_misalignment.csv`.

One frozen solve is the available proxy for one future Picard linearization. Each row
records the counters at the actual call sites: one $A$ assembly, one UMFPACK
factorization, five response solves, and four subsequent uses of the same
factorization after the first response. Since the kernel is direct, Krylov iterations
are `not_applicable` and the preconditioner is `none`. No cross-call cache exists in
the frozen API; Phase 5 must measure cross-iteration reuse in the actual Picard driver.
The timing columns separate $A$ assembly, right-hand-side assembly, diagnostic-only
SUPG assembly, factorization/responses, the bordered production solve, post-solve
diagnostics, and the total call. Across the three resonant rows, full-∇ assembles $A$
about 2.3 times faster (within-row transverse/full ratios 2.21--2.46); the fine
misalignment ratio is 1.73. Bordered-solve differences are below the observed
run-to-run timing spread and are not attributed to either closure. These single-run
local wall times do not establish a nonlinear cache or preconditioner advantage.

The full-gradient closure provides small damping and smearing changes, lower absolute
h-sensitivity in both alignment controls, and modest local timing differences in these
tests. The evidence is not large or broad enough to override the note-derived
transverse physics, especially before a nonlinear driver can measure real
cache/preconditioner behavior. The default remains
`regularization_gradient="perpendicular"`; no ADR is warranted by this study.

Mutation checks verified that the comparison cannot pass with collapsed or mixed
operators. Forcing the full-gradient Galerkin/M2 path through the perpendicular
projection reduced the $D_u=0.04$ normalized cross-variant difference from 1.0104 to
0.0372 and failed the $O(\epsilon_J)$ gate. Reconstructing the full-gradient physical
(M2) regularizing current with the perpendicular operator while retaining the full
(M3)--(M3b) solve raised the independently evaluated resonant-case current residual
to $9.017\times10^{-5}$, far above the $10^{-10}$ gate. Finally, deleting the
full-gradient $J_\parallel/B$ correction changed the $D_u=0.04$ noise transfer from
the pinned 0.0054730 to 0.0052886 and failed its 0.5% relative gate.

## Milestone 3.6 — constrained unknown-G M3–M3b solve

For frozen $(\mathbf B,p,\chi)$, the production current-continuity kernel now solves
jointly for homogeneous $\tilde u$ and a piecewise-linear unknown $G(s)$,
$u=G(s)+\tilde u$, using the bordered system

\[
\begin{pmatrix}A&P\\ C_u&C_G\end{pmatrix}
\begin{pmatrix}\tilde{\mathbf u}\\\mathbf g\end{pmatrix}
=\begin{pmatrix}\mathbf f\\\Delta\mathbf I_0\end{pmatrix}.
\]

The $P$ columns contain both $\mathbf B\cdot\nabla G$ and
$(\mu_0G/B_{\rm safe}^2)\mathbf B\cdot\nabla p$, including their SUPG rows. The
physical current used by the shell constraints is reconstructed from (M2),

\[
\mathbf J=(G+\tilde u)\mathbf B
+\frac{\mathbf B\times\nabla p}{B_{\rm safe}^2}
-D_u\nabla_r\tilde u,
\]

so the regularizing flux never acts on full $u$. The same runtime-selected
$\nabla_r$ (perpendicular or full) is used in $A$, SUPG, $C_u$, the independent
current reconstruction, and the multiplier-current diagnostic. $C_G$ contains only
$G\mathbf B\cdot\nabla\phi$; adding a separate $D_uG'\nabla_rs$ term there would
double count the constrained closure.

The parallel-current diagnostic is taken from the reconstructed physical current. For
the perpendicular closure it is $J_\parallel/B=u$; for the full-gradient closure it is

\[
\frac{J_\parallel}{B}=u-\frac{D_u}{B_{\rm safe}}
\mathbf b_{\rm safe}\cdot\nabla\tilde u.
\]

The public point evaluator and its L² diagnostic therefore never report auxiliary $u$
as physical parallel current in the full-gradient variant. The manufactured suite
checks the point evaluator independently against $\mathbf J\cdot\mathbf B/B^2$.

The normalized-volume coefficient used by the $G$ basis is the exact monotone PCHIP
from `MollifiedVolumeMap`, transcribed interval by interval together with its analytic
gradient. Its mapped-quadrature samples agree with the shell evaluator's shared
$s=V_\chi/V_\Omega$ samples within $2\times10^{-12}$. Shell rows are built from the
same compact mollified layer-set functional as milestone 3.5. A single sparse UMFPACK
factorization of $A$ supplies the base solution and all $A^{-1}P$ response columns;
only the shell-sized Schur complement is dense. Checkpoint schema 1 now optionally
stores the normalized shell grid, piecewise-linear basis identifier, solved $G$
coefficients, $G(1)=u_b$, every independently reconstructed M3b row residual, and the
M3/M3b relative residuals alongside the normalized $p_0/I_0$ profile payload. No
legacy prescribed-$F$ state is accepted.

The coupled manufactured solution makes both $G$ couplings and all three M2 current
components nonzero. The checked-in h/p/N table is
`tests/manufactured/m3_constrained_rates.csv`. At polynomial order 2, the physical-$u$
L² errors converge as follows:

| Variant | Subdivisions (per axis) | Error (coarse → fine) | Measured h-rate |
| --- | ---: | ---: | ---: |
| perpendicular | 20 → 28 | 2.9070e-4 → 1.4811e-4 | 2.0041 |
| full | 20 → 28 | 2.9609e-4 → 1.5085e-4 | 2.0042 |

At 24 subdivisions, raising $p=1\to2$ reduces the error from
$3.3789\times10^{-4}$ to $2.0165\times10^{-4}$ (perpendicular) and from
$3.8692\times10^{-4}$ to $2.0538\times10^{-4}$ (full). The $p=3$ values,
$2.0163\times10^{-4}$ and $2.0537\times10^{-4}$, expose the second-order mollified-
shell ceiling rather than an algebraic-solve limit. Across the h/p rows, the largest
M3 relative residual is $1.131\times10^{-16}$ and the largest independently evaluated
M3b relative residual is $1.063\times10^{-16}$.

Doubling the shell count from 4 to 8 on a $32\times32$ mesh changes the physical field
by $2.145\times10^{-4}$ (perpendicular) and $2.176\times10^{-4}$ (full) in relative L²,
evaluated on one common order-20 mapped-quadrature rule. The former point sample moves
by $6.99\times10^{-6}$ and $7.33\times10^{-6}$, respectively, but is retained only as
a secondary reproducibility value rather than the convergence norm. The eight-shell
grid spans 3.991 local radial-cell widths and 3.991 mapped mollifier widths per shell,
at the lower edge of the required 3–4-cell resolution. Two distinct $I_0(s)$ profiles
are realized by independently reconstructed cumulative currents to the $10^{-10}$
solver gate for both variants on the coupled state with $\mathbf B\cdot\nabla s\ne0$,
$\mathbf B\cdot\nabla p\ne0$, and nonzero $\tilde u$. The base profile also reproduces
the analytic physical $u$ below $5\times10^{-4}$, so deleting the G-advection coupling
turns this positive control red. The historical negative control confirms that two
distinct old $F(p)$ shifts with the same boundary value reconstruct the same physical
$u$ below $10^{-10}$.

The fixed-$I_0$ regular-limit scan is recorded in
`tests/manufactured/m3_constrained_du_scan.csv`. It uses a nondegenerate manufactured
family with both G couplings nonzero, one analytic $I_0(s)$ shared by every $D_u$, a
bounded G profile that genuinely changes with $D_u$, and a shell-mean $\tilde u$
correction proportional to $D_u$. For $D_u=0.08\to0.04\to0.02$,
$\|D_uG'\nabla_rs\|_2$ decreases
$2.8498\times10^{-2}\to1.3611\times10^{-2}\to6.7504\times10^{-3}$ (perpendicular)
and $2.8744\times10^{-2}\to1.3721\times10^{-2}\to6.8044\times10^{-3}$ (full).
The maximum shell $|\langle\tilde u\rangle|$ simultaneously falls from
$3.9866\times10^{-2}$ to $9.3790\times10^{-3}$ (perpendicular) and from
$3.9875\times10^{-2}$ to $9.3877\times10^{-3}$ (full). The largest M3 and M3b
relative residuals in this scan are $1.096\times10^{-16}$ and $7.186\times10^{-17}$.
This is a manufactured-family realization check, not an emergent fixed-frozen-state
limit: `magnetic_magnitude_gradient` is re-derived at each $D_u$ from an exact family
whose $\tilde u$ and bounded, $D_u$-dependent $G'$ were chosen to vanish regularly.
Thus the scan demonstrates that the bordered solver realizes an admissible bounded-$G'$
family and that the mandatory controls fail when its physics is mutated; it does not
show that an arbitrary fixed target and frozen drive are admissible. The note's
vanishing diagnostic is qualified by “for an admissible target.” Milestone 3.7 must
perform the emergent scan with $(\mathbf B,p,s,I_0)$ and the frozen drive held fixed,
and must report or reject a target when $\|D_uG'\nabla_rs\|$ or
$\langle\tilde u\rangle_s$ fails to approach a common regular limit, in addition to
the broader resonant/misaligned cross-variant comparison.

The final M3b evaluation is independent in the specific §9.2 sense that it resamples
the reconstructed physical M2 current and never reuses assembled $C_u/C_G$ rows. Its
roundoff-level residual certifies consistency between solve rows and reconstruction and
therefore catches omitted or mismatched current components. Because both paths apply
the same linear mollified-shell functional to the same quadrature convention, it cannot
detect a common-mode error in that functional; the hand-derived analytic $I_0$ oracle
in the coupled manufactured case supplies that separate physics check.

Mutation checks on the first coupled h-row gave the following conspicuous failures:

- deleting $-G'\mathbf B\cdot\nabla s$ raised physical-$u$ L² error from
  $2.907\times10^{-4}$ to $5.777\times10^{-2}$;
- dropping the $-(\mu_0G/B^2)\mathbf B\cdot\nabla p$ coupling raised it to
  $1.293\times10^{-3}$;
- omitting the diamagnetic or regularizing M2 shell contribution raised the independent
  M3b residual to $1.160\times10^{-2}$ or $3.304\times10^{-4}$, respectively; and
- reconstructing M2 with diffusion on full $u$ instead of $\tilde u$ raised the
  independent M3b residual to $4.746\times10^{-4}$.

After replacing the formerly degenerate mandatory controls, the reviewer mutation that
deletes $\mathbf B\cdot\nabla G$ now also turns all four two-$I_0$/regular-limit test
instances red: the two-profile base-case errors rise to $5.783\times10^{-2}$
(perpendicular) and $5.855\times10^{-2}$ (full), while the $D_u=0.08$ multiplier norms
move from $2.850\times10^{-2}$ to $4.525\times10^{-2}$ and from $2.874\times10^{-2}$
to $4.634\times10^{-2}$, respectively.

## Milestone 3.5 — normalized profiles and shell-current moments

The public pressure contract is now $p_0(s)$, and the new cumulative-current
contract is $I_0(s)$, with the one shared coordinate

\[
s(\mathbf r)=\frac{V_\chi(\chi(\mathbf r))}{V_\Omega}\in[0,1].
\]

Analytic and piecewise-linear pressure/current variants reject evaluation outside
$[0,1]$. Pressure tables must be non-increasing and satisfy the requested
$p_0(1)=p_b$; current tables enforce $I_0(0)=0$ but deliberately allow reversed-
current segments. Checkpoint profile records require the literal
`coordinate_kind="normalized_volume"`; missing tags, dimensional-volume tags, and
legacy prescribed-F records are rejected rather than inferred from sample ranges.
The metadata-only schema remains version 1 because it previously persisted no profile
payload; its first profile-bearing configuration uses only the corrected contract.
The legacy `PrescribedCurrentProfile` is no longer exported from `remec.solvers`, and
its remaining F-shift solve emits a deprecation warning and exists only for milestone
3.6's two-F cancellation oracle.

`MollifiedVolumeMap.evaluate_volume_coordinate` is the sole evaluated-s path used by
the M4b transplant and the M3b moment diagnostic. The normalized layer-cake target is
now implemented as

\[
\int_\Omega\varphi(p)\,dV
=V_\Omega\int_0^1\varphi(p_0(s))\,ds.
\]

The independent current diagnostic keeps the three physical (M2) integrands separate,

\[
\mathbf J=u\mathbf B+\frac{\mathbf B\times\nabla p}{B_{\rm safe}^2}
-D_u\nabla_r\tilde u,
\qquad
I_{\rm tor}(s)=\frac1{2\pi}\int_{\Omega_s}\mathbf J\cdot\nabla\phi\,dV,
\]

then uses the same spatial mollifier mapped into s for cumulative rows. Shellwise rows
are adjacent cumulative differences. Thus the endpoint ($I_{\rm tor}(0)=0$), total-
current row ($I_{\rm tor}(1)$), component sum, and shell-partition identities are
imposed by construction; the tests independently compare every cumulative and
shellwise component with analytic circle/annulus integrals.

The canonical shell weight evaluates the shared compact Heaviside kernel in s-space
with local half-width
$\tfrac12|s(\chi-\epsilon)-s(\chi+\epsilon)|$. This preserves the exact shared s field
and its endpoints. It differs from evaluating the chi-space expression
$H_\epsilon(\chi-\chi(s_k))$ only at the mollifier consistency order, and the measured
moment convergence remains second order. Both volume-map construction and shell
moments call the same kernel helper so a future kernel change cannot make them drift.

Shell resolution is checked locally against both relevant scales. The chi half-width
is divided by `spatial_width_cells` to recover one radial-cell width, mapped into s,
and each shell must span at least three such local widths. Each shell must also span
at least two local mapped mollifier widths, preventing a wider smoothing kernel from
passing the cell-count check while smearing across a shell. At 96 radial cells, the
marginal 15-shell equal-volume partition has a maximum total-current error of
$2.085\times10^{-5}$. A graded 18-shell partition demonstrates that the local check,
unlike a global-maximum-width check, retains resolved inner shells with a
$2.084\times10^{-5}$ error. Tests with mollifier widths of 0.5, 1.5, and 2.0 radial
cells independently pin both guards.

The manufactured axisymmetric surrogate uses a circular poloidal section, analytic
radial M2 projections, and an integrated toroidal angle. At 96 radial cells and
quadrature order 6, maximum cumulative errors are $7.801\times10^{-5}$ (parallel),
$3.899\times10^{-5}$ (diamagnetic), $1.949\times10^{-5}$ (regularizing), and
$1.953\times10^{-5}$ (total). Rescaling the radius from 1 to 2.75 while scaling
current density by inverse area changes the total profile by only
$3.55\times10^{-15}$, and the shared sampled s field by $8.88\times10^{-16}$.
The checked-in h-refinement table is
`tests/manufactured/shell_current_moment_rates.csv`:

| Radial cells | Quadrature order | Maximum cumulative error | Adjacent rate |
| ---: | ---: | ---: | ---: |
| 24 | 6 | 2.6828e-4 | — |
| 48 | 6 | 6.6678e-5 | 2.0085 |
| 96 | 6 | 1.6649e-5 | 2.0018 |

At fixed 48-cell resolution, quadrature orders 1, 2, and 3 reduce the same error from
$5.599\times10^{-4}$ to $1.578\times10^{-4}$ to $6.173\times10^{-5}$, after
which the spatial mollification error dominates. Mutation checks removed the
regularizing M2 term from the total, producing a maximum discrepancy 0.393 and turning
the analytic component/total test red; removing division by $V_\Omega$ from the
shared s field made both pressure and current radius-rescaling tests fail.

## Milestone 3.4 — resonant M3 layer scaling

The note's local resonant reduction of (M3),

\[
i\,k_\parallel(x)B\,\hat u
-D_u\frac{d^2\hat u}{dx^2}=\hat h,
\qquad
\delta\sim\left(\frac{D_u\bar R}{m|\iota'|B}\right)^{1/3},
\]

is tested on a periodic-\(y\) slab with
\(\mathbf B=(0,20(x-1/2),10)\) and the fundamental resonant drive
\(\hat h\sin(2\pi y)\). This gives the retained local balance
\(i40\pi(x-1/2)\hat u-D_u\hat u''=\hat h\). The frozen verification interface
injects this fixed drive through its explicit M3 drive numerator. The nondimensional
\(\mu_0=10^{-8}\) isolates the reduced balance by making the first-derivative correction
asymptotically negligible, exactly as in the note's layer analysis. The physical field
is reconstructed through the preferred \(u=F+\tilde u\) path with \(F=0.2\), and the
measured observable is the full width at half maximum of the fundamental Fourier
amplitude of the reconstructed (M2) \(J_\parallel/B\), not the solved auxiliary field.
The note's same `layer_width` line also gives a peak-amplitude scaling, but DESIGN §25
defines milestone 3.4 by \(\delta\propto D_u^{1/3}\). This record therefore claims and
tests only the width half of that line; it makes no peak-amplitude convergence claim.

The layer-aligned triangular mesh has 64 elements normal to the resonant surface and 16
along its smooth periodic harmonic (2048 elements total), so \(h_\perp=1/64\).
`MakeStructured2DMesh(periodic_y=True)` supplies the top/bottom identification and the
M3 kernel consumes it with `ngsolve.Periodic(ngsolve.H1(...))`; the only Dirichlet
boundaries are `left|right`. An automated regression evaluates the solved direct-u and
utilde fields at both sides of the seam and requires agreement to \(10^{-11}\).
Degree-three H1 elements use the production SUPG path for both runtime gradient
variants. This milestone isolates the resonant advection/transverse-diffusion balance;
the milestone-3.2 and 3.3 tests, rather than this width observable, constrain the other
SUPG residual terms and their signs. The checked-in machine-readable scan is
`tests/manufactured/m3_layer_scaling.csv`:

| Gradient variant | \(D_u\) | FWHM | Unit-prefactor inner scale | FWHM / inner scale | FWHM elements |
| --- | ---: | ---: | ---: | ---: | ---: |
| ∇⊥ | 0.0025 | 0.129309 | 0.027096 | 4.772 | 8.276 |
| ∇⊥ | 0.005 | 0.163535 | 0.034139 | 4.790 | 10.466 |
| ∇⊥ | 0.010 | 0.207292 | 0.043013 | 4.819 | 13.267 |
| ∇⊥ | 0.020 | 0.263781 | 0.054193 | 4.867 | 16.882 |
| ∇⊥ | 0.040 | 0.337937 | 0.068278 | 4.949 | 21.628 |
| full ∇ | 0.0025 | 0.129302 | 0.027096 | 4.772 | 8.275 |
| full ∇ | 0.005 | 0.163511 | 0.034139 | 4.790 | 10.465 |
| full ∇ | 0.010 | 0.207215 | 0.043013 | 4.818 | 13.262 |
| full ∇ | 0.020 | 0.263537 | 0.054193 | 4.863 | 16.866 |
| full ∇ | 0.040 | 0.337151 | 0.068278 | 4.938 | 21.578 |

The note specifies a proportional inner scale, not an FWHM convention. Taking the
local balance's coefficient literally gives
\(\delta_0=(D_u/(40\pi))^{1/3}\), which spans only 1.734--4.370 base-mesh elements.
The reported operational layer width is explicitly the reconstructed-current FWHM,
4.772--4.949 times \(\delta_0\). Consequently, the DESIGN §5 resolution diagnostic is
called here with the measured FWHM: the statement that every row clears six elements
is an FWHM-based verdict, not a claim that the unit-prefactor estimate itself clears
six. Its mesh independence is checked at the thinnest case in
`tests/manufactured/m3_layer_mesh_refinement.csv`:

| \((n_x,n_y)\) | Elements | Measured FWHM | FWHM elements |
| --- | ---: | ---: | ---: |
| (64, 16) | 2048 | 0.1293094217 | 8.275803 |
| (96, 24) | 4608 | 0.1293169073 | 12.414423 |

The two widths differ by \(5.789\times10^{-5}\) relative, so the minimum-resolution
verdict does not depend on the base mesh. Least-squares fits of \(\log(\mathrm{FWHM})\)
against \(\log D_u\) give exponents 0.346160 (∇⊥) and 0.345392 (full ∇). These are not
presented as exact one-third laws: `fwhm_to_inner_scale` drifts monotonically over the
finite-\(D_u\) scan. The adjacent exponent moves from 0.338771 at the lowest pair to
0.357415 at the highest pair for ∇⊥ (0.338644 to 0.355391 for full ∇). Extending the
scan downward from \(D_u=0.005\) to 0.0025 therefore moves the fit toward \(1/3\),
evidence that the remaining positive residual is a finite-\(D_u\) effect. The automated
gate checks that the lowest pair is closer to \(1/3\) than the highest pair, requires
the global exponent to lie within 0.04 of \(1/3\), requires every layer width to
increase strictly with \(D_u\), and compares every measured row with the recorded table
within 5%. Free-DOF relative residuals remain below \(4.36\times10^{-17}\),
sampled \(\min|\mathbf B|=10\), and the smooth field-floor activity is below 1e-12. A
direct-u solve at \(D_u=0.01\) independently agrees with reconstructed utilde to 1e-10
in physical \(u\) and \(J_\parallel/B\), and to 1e-9 componentwise in the reconstructed
(M2) current, for both variants.
The profile is labeled `resonant-layer-constant-f-v1`; every transformed solve now
requires such a stable caller-owned `PrescribedCurrentProfile.identifier`, exposes it
on the result, and includes it in both the configuration digest and structured start/
completion records. Changing only the identifier changes the digest, so later coupled
checkpoints cannot silently substitute a different \(F(p)\) provenance record.

`CurrentContinuitySolver.assess_layer_resolution` implements the production-facing
resolution contract. It reports \(\delta/h_\perp\), warns with
`UnresolvedCurrentLayerWarning` below `RuntimeOptions.min_layer_cells` (default 6), and
raises `UnresolvedCurrentLayerError` in strict mode. The normal element width must be
supplied from the local mesh metric; polynomial degree does not inflate the element
count. Thus algebraic convergence and high-order degrees never substitute for a
resolved physical layer. This diagnostic is necessarily caller-invoked because the
frozen solver cannot infer a physical \(\delta\); a later coupled production driver
must call it after estimating both the layer width and its local normal mesh scale.

Mutation checks confirmed that, at \(D_u=0.005\), halving the Galerkin
\(D_u\nabla_rv\cdot\nabla_ru\) coefficient changes the perpendicular width from
0.163535 to 0.129504 and turns the recorded-width test red. Doubling the reported
normal element width changes an eight-element resolved diagnostic to four elements,
emits the unresolved warning, and turns the unit contract red. Together these
constrain both the M3 balance that produces the \(D_u^{1/3}\) layer and the independent
§5 resolution accounting.

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
