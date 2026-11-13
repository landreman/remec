# ADR 0014: Calibrate the 3D island gate before selecting its pressure threshold

**Status:** Proposed — blocks completion of milestone 6.3

**Date:** 2026-09-05

## Context and the decisions requiring sign-off

The investigation of `scratch/plot_frozen_field_island_solution.py` found that the
periodic field and scalar traces are correct, but the present milestone ladder
demonstrates weaker pressure suppression than its design and ADR 0013 imply.
This affects (M4a), (M4b), the interpretation of note equation `(wc)`, and the
pressure/level-set diagnostics. It does not propose changing the transport tensor,
source, pressure profile, geometry contract, pollution tolerance, or test budgets.

Three passages need to be distinguished:

1. Note §4.3 gives an order-of-magnitude balance:
   `w_c ~ epsilon_kappa^(1/4) (L_s/k_theta)^(1/2)`.
2. DESIGN §8.6 says the ε₁=10⁻³ crossing is between ε_κ=10⁻³ and 10⁻⁴, and
   “That is what makes the benchmark affordable in 3D”. It also requires that
   “the flattening width follows max(w_island, w_c) and scales as
   ε_κ^{1/4} below threshold”.
3. Note §8.4 says the inverse `chi_hat(s)`, “and with it the composed profile
   `p(s)`, is nearly flat”. Yet (M4b) prescribes `p(s)=p0(s)`; for the current
   input `p0(s)=1-s`, its derivative is exactly −1. Note §6.1 also describes the
   transplanted spatial profile as “flat across the flattened regions”. Its
   later paragraph acknowledges dependence of island-interior level spacing on
   the reference source. Preserved level sets alone do not fix gradient magnitude.

The quarter-power scaling in passage 1 is sound. Passage 2 treats a local balance
length as a full-island threshold and makes an unsupported quantitative prediction.
Passage 3 needs a mathematical clarification of the independent variable and the
strength of the claimed spatial flattening. Neither the note nor the acceptance
criteria are silently reinterpreted by this ADR. The proposed corrections below
require human sign-off under AGENTS.md's STOP conditions.

## Local derivation with the width convention explicit

Write `W = w_island = 4 sqrt(epsilon_1/(2 t1))` for the **exact full radial
width**, `r_s = sqrt((1/2-t0)/t1)`, `zeta=2 theta-z/R0`, and `x=r-r_s`.
On the unperturbed resonant surface,

    B_s = sqrt(1 + (r_s/(2 R0))^2),
    k_parallel = (2 iota - 1)/(R0 |B|),
    alpha = |d k_parallel/dr|_rs = 2 iota'(r_s)/(R0 B_s),
    delta = (epsilon_kappa/alpha^2)^(1/4).

To leading local order, with `ell=W/4`,

    b.grad ≈ alpha [x partial_zeta - ell^2 sin(zeta) partial_x].

Scaling `x=delta X` leaves island strength `(ell/delta)^2` in this directional
operator. The competition in (M4a) is therefore characterized by

    Lambda = (W/(4 delta))^4
           = 4 epsilon_1^2 r_s^2
             / [epsilon_kappa (R0^2 + r_s^2/4)].

The natural pendulum full-width scale `4 delta` and the local diffusion length `delta`
must have separate names. Neither `Lambda=1` nor `W=4 delta` is an exact threshold
for an operational pressure-gradient statistic. This is a local approximation;
finite island width, source, boundary, and the M4b composition still matter.

At `t0=.29, t1=.38, R0=1`, `B_s=1.06684483`, `alpha=1.05915661`, and
`delta=.97167250 epsilon_kappa^(1/4)`. Ignoring `B_s` accounts for only a 3.3%
length difference from the design's .94074 coefficient. The consequential mistake
is using `W>delta` as evidence of strong flattening. A factor four in a width
comparison corresponds to a factor 256 in epsilon_kappa; it is not harmless in
selecting a numerical ladder even though it is allowed by a scaling relation.

| ε₁ | exact W | ε_κ at Lambda=1 (local estimate) | Lambda at ε_κ=1e-5 |
| ---: | ---: | ---: | ---: |
| .001 | .14509525 | 1.94220e-6 | .19422 |
| .005 | .32444284 | 4.85549e-5 | 4.85549 |

This explains why extending the thin-island ladder only to 1e-5 does not establish
a strongly flattened interior. It does not mean there is no response above 1e-5:
flattening develops continuously, and the response is already measurable there.

## Evidence and its limits

The September 4–5 exploratory results are preserved in
[0014-calibration-snapshot.csv](0014-calibration-snapshot.csv), copied from the
diagnostic's machine-written records. The runs used NGSolve 6.2.2606 on the local
checkout at `a5c936e`, `S_ref=1`, and `p0(s)=1-s`. They are **not** acceptance
rate tables, a completed 3D resolution study, or a replacement for exhaustive CI.
The local reproduction scripts are `scratch/diagnose_frozen_field_island.py` and
`scratch/plot_island_diagnosis.py`; scratch is untracked. Before these results become
a verification gate, commit a maintained reproducer and automated reference tests.

The independent diagnostic is the exact helical-symmetry reduction for this
single-helicity field and circular wall. For the R0=1 diagnostic runs, at z=0, set

    v = (Y,-X)/2,
    q = (B_x+Y/2, B_y-X/2),
    K_2 = epsilon_kappa (I+v v^T) + (1-epsilon_kappa) q q^T/|B|^2.

It retains the axial derivative exactly; it does not normalize `q` to a unit
island tangent. Its ε₁=0 solution agrees with `(1-r^2)/(4 epsilon_kappa)` to
1.10e-10 relative L² error. Refining maxh=.07→.035 at order four for
ε₁=.005, ε_κ=1e-4 changes the local χ gradient ratio .53737284→.53738630.
These are reference-accuracy checks, not an asserted asymptotic rate.

The following ratios are radial derivatives at the unperturbed resonance on the
O-point ray, divided by the exact integrable gradients. The last column is the
pressure drop across the full magnetic island width divided by the integrable drop.

| ε₁ | ε_κ | calculation | χ gradient ratio | p gradient ratio | p drop ratio |
| ---: | ---: | --- | ---: | ---: | ---: |
| .001 | 1e-4 | helical, p=4, maxh=.035 | .9163 | .9186 | .9356 |
| .001 | 1e-5 | helical, p=4, maxh=.035 | .7025 | .7897 | .8466 |
| .001 | 1e-6 | helical, p=4, maxh=.035 | .1946 | .5043 | .7378 |
| .005 | 1e-4 | 3D, p=3, 24×4 | .5322 | .7096 | .8141 |
| .005 | 1e-4 | 3D, p=3, 24×8 | .5376 | .7163 | .8140 |
| .005 | 1e-5 | helical, p=4, maxh=.0175, 257 levels | .02844 | .2239 | .7028 |

For the .005/1e-4 3D pair, periodic B traces agree below 3e-16, χ traces below
1e-12, and the local helical reference gives χ ratio .5374. Axial under-resolution
does not account for the absence of a strong plateau at those parameters.

The .005/1e-5 production row at p=3 and 36×4 has 339,781 H¹ DOFs and took about
238 seconds through (M4a)–(M4b) on four threads. It shows strong χ suppression, but
its local O-ray derivative changes sign relative to the reference and its M4b
co-area consistency error is .216 versus a .2 warning threshold. It is not yet a
quantitatively resolved 3D benchmark. Local p derivatives are also more sensitive
than χ to level-map resolution. Do not claim a lower-resolution cost win from it.

## Required distinctions in the diagnostics

- A threshold of .97 measures a reduction exceeding 3%, not a near-flat plateau.
  A .95/.97/.99 sensitivity scan tests only the weak 1–5% suppression range.
  Width stability within that range cannot establish strong-flattening saturation.
- `max(W,delta)` can describe the scale of a transport response. It cannot be an
  identity for the width where a fixed substantial gradient reduction is attained:
  that width can be zero for a weak island while delta remains nonzero. Measure
  response extent, suppression amplitude, and pressure drop separately.
- From (M4b), `partial_r p = p0'(s) V_chi'(chi) partial_r chi / V_omega`.
  Level sets and the parallel-to-total gradient ratio are preserved; the magnitude
  of a spatial gradient reduction is not. Measure both χ and p, with p carrying the
  pressure acceptance claim. `dp/ds=-1` checks the selected transplant, not flattening.
- For p0(s)=1-s, exact realization of its volume distribution excludes an exactly
  constant pressure on a finite-volume region, which would create an atom in that
  distribution. This does not preclude local radial flattening or a finite reduction
  over a specified region. The note's stronger wording requires clarification.
- A co-area spike means a steep forward V_chi and a flat inverse chi_hat(V), as in
  ADR 0013 Option 2. A magnetic X-point is not automatically a critical point of χ
  at finite anisotropy. Verify scalar critical structure before claiming its
  singularity, and report a localized enhancement if a singularity is not established.
  Any absolute co-area scaling also needs an explicit χ normalization: multiplying
  χ by a positive constant c leaves M4b pressure unchanged but divides its co-area
  density by c.

## Options and tradeoffs

**Option 1 — Keep ε₁=.001 and extend the anisotropy ladder.** Retain the present
geometry and thin island, calibrating into the 1e-6–1e-7 range as needed. This preserves
the original example but requires thinner resolved transport layers and more costly
pollution, volume-map, and solver verification. No particular endpoint is guaranteed
to meet a not-yet-agreed pressure statistic.

**Option 2 — Use ε₁=.005 as the principal flattening case; retain the thin-island
rows as controls and reference cases.** Keep R0=a=1, the full normalized physical B,
S_ref=1, and p0(s)=1-s. Use the existing high-ε_κ rows as weak-response controls and
calibrate a ladder through 1e-4 and 1e-5, extending lower only if the agreed pressure
statistic requires it. At equal Lambda, this permits 25 times larger epsilon_kappa
and sqrt(5) times larger local delta than ε₁=.001. This is a plausible reduction
in radial resolution cost, not a measured total-DOF saving. Angular/axial resolution,
wall clearance, and volume-map accuracy remain independent gates. The .001 case at
1e-5 has a measurable response, so it must not be called a no-response control;
choose any required negligible-island control through its measured suppression.

The .005 island's outer separatrix is at .905613, leaving .094387 to the unit wall.
Increasing ε₁ to .01 brings it to .972808, leaving only .027192 and introducing a
new wall-resolution concern. Enlarging the island without checking the wall is not
an unlimited route to cheaper runs.

**Option 3 — Shorten R0 or change shear/domain parameters.** This changes connection
lengths and potentially the required epsilon_kappa. It also changes the helical
perpendicular metric, axial geometry, aspect ratios, and the comparison to milestone
6.2. At fixed ε₁, Lambda scales as `1/(R0^2+r_s^2/4)`, not indefinitely as R0^-2.
This remains an untested alternative if Option 2 cannot achieve the required budget.

## Recommendation and proposed document edits

Recommend **Option 2**, subject to human approval and calibration of a substantial
pressure-suppression criterion. It has direct diagnostic support and avoids changing
the underlying field normalization, source, or profile to manufacture a plateau.
Adopt ADR 0013's **Option 2 interpretation** of the inverse plateau, with its window
and strength criteria calibrated on a resolved scalar solution rather than chosen to
make the old weak-signal rows pass. The note's p(s) wording remains a separate
mathematical clarification requiring sign-off.

After approval, update DESIGN §8.6 and STATUS's reference paragraph to distinguish
W, delta, and Lambda; remove the claimed 1e-3→1e-4 strong-flattening crossing and
the literal max formula for thresholded width. State separately a pressure gradient
threshold, the physical interval over which it must hold, and its refinement
tolerance; candidates such as 50% suppression over a finite interval may be examined
but are **not accepted thresholds in this ADR**. A single zero derivative at a local
extremum is insufficient. Calibrate pressure-drop and localized co-area observations
without replacing the pressure gate by a χ-only test.

**Do not multiply the mesh's local resolution width by four while leaving
min_layer_cells unchanged.** That would silently allow four times larger cells.
Preserve the accepted local element-width gate and pollution tolerance during the
approved nomenclature correction. A change of resolution criterion needs its own
explicit justification and sign-off.

Then commit the reference reproducer/tests, select the 3D mesh using independent
radial/angular/axial/order and volume-map checks, regenerate the acceptance artifacts
through their sole script, retain all falsifiability controls and fast sentinels, and
complete a fresh full branch exhaustive run. Milestone 6.3 remains `[~]` until that
work passes. The annotations added with this ADR withdraw unsupported evidence claims;
they do not approve new solver inputs, relaxed tolerances, or completed acceptance.

**DECISION: pending human sign-off**
