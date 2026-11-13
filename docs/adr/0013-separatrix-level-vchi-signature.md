# ADR 0013: Separatrix-level Vχ signature for the 3D island gate

**Status:** Proposed

**Date:** 2026-08-23

**Review update:** 2026-09-05; related parameter and M4b questions are recorded in
[ADR 0014](0014-island-flattening-calibration.md). No option has been approved.

## Context

Milestone 6.3 must make its level-set diagnostic local and falsifiable. `docs/DESIGN.md`
§8.6 item 5 requires “the near-plateau of Vχ(χ̂) at the island level and the
corresponding spike in the co-area density −dV/dχ̂”. Section 12.3, however, says that
in flattened regions Vχ is nearly a **step** and its inverse χ̂(V) is tame. At a
differentiable point, a plateau of Vχ has small |dV/dχ̂|, whereas a co-area spike is
large |dV/dχ̂|. The two §8.6 requirements therefore cannot hold at the same point under
their literal reading.

The adversarial review correctly rejected the former global max/median diagnostics:
they were not localized to the island and the no-flattening control could score higher.
The corrected implementation evaluates χ at the analytic island X-point, maps it to
the normalized volume coordinate, and compares against the ε₁=0 solution on the same
mesh and normalized volume. Fresh-process regeneration gives:

| row | main/control co-area ratio | control/main inverse-level span ratio |
| --- | ---: | ---: |
| ε_κ=10⁻², 12×2 | 1.0571 | 1.000016 |
| ε_κ=10⁻⁴, 48×8 | 1.0167 | 0.999879 |
| ε_κ=10⁻⁵, 36×4 | 1.1065 | 1.000135 |

The ε_κ=10⁻⁵ pressure result is stable over the tested **weak-suppression** thresholds: widths at
0.95/0.97/0.99 of the integrable gradient are 0.1516/0.1594/0.1632 versus exact island
width 0.1451. These thresholds detect reductions exceeding only 5%/3%/1%. They do
not independently establish a near-flat pressure interior or saturation of a width
defined by substantial suppression. The earlier inference that the ambiguity affected
only Vχ, while the strong-flattening branch was settled, is withdrawn.

For ε₁=1e-3, the September diagnostic reference gives a local O-ray p-gradient
ratio .790 and a full-island pressure-drop ratio .847 at ε_κ=1e-5. Moreover, the
analytic magnetic X-point is not automatically a critical point of the finite-anisotropy
χ field. The table above remains historical evidence of its stated measurements;
the weak inverse-span signal alone cannot decide the appropriate physical gate.

## Option 1 — Literal Vχ plateau

Gate a small local |dV/dχ̂| relative to the integrable control and drop the co-area
spike requirement at that point.

This follows the word “plateau” literally but contradicts the explicit “corresponding
spike” and §12.3’s statement that flattened Vχ is nearly a step.

## Option 2 — Co-area spike plus inverse-map plateau

Interpret §8.6’s plateau as a plateau of the inverse χ̂(V), consistent with §12.3.
At the analytic separatrix volume coordinate, report and gate both the main/control
co-area-density ratio and the control/main local inverse-level-span ratio. Retain the
same-mesh integrable control and report the normal-metric mollifier width there.

This makes the derivative and inverse-derivative relationship mathematically
consistent, but the measured inverse-span signal is currently only O(10⁻⁴), so a
meaningful tolerance must be specified rather than inferred from roundoff-scale data.

## Option 3 — Local transition-window signature

Treat “plateau and corresponding spike” as two features in a mollifier-width-sized
window around the analytic separatrix, not at one point. Gate the window’s co-area peak
against the same-mesh integrable control and gate flatter shoulders on either side,
with the window and sampling rule fixed in normalized-volume coordinates.

This can represent a smoothed step faithfully and may yield a stronger discriminator,
but introduces a window definition that the current design does not specify.

## Recommendation

Recommend **Option 2's mathematical interpretation**: a co-area spike is a steep
forward Vχ and a flat inverse χ̂(V). The former recommendation of Option 3 was based
on preserving the erroneous word “plateau” for the forward map; that wording is not
a mathematical reason to introduce a new statistic. Option 3 remains available as
an explicitly chosen localized diagnostic, but is not required to resolve the
derivative/inverse-derivative relationship.

Before setting quantitative gates, resolve ADR 0014's pressure-strength and parameter
calibration, verify whether χ has the claimed scalar critical structure, and select
a reproducible localized statistic with independent refinement of its volume-map
resolution. Do not assume that its maximum must occur at χ evaluated at the magnetic
X-point. A pressure plateau cannot be inferred solely from a χ or co-area feature,
because (M4b) preserves level sets but can amplify the remaining spatial gradient.
No global max/median statistic should return, and no threshold should be selected
merely to make the old weak-response rows pass.

## Decision

**DECISION: pending human sign-off**
