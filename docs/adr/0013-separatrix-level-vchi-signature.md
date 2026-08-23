# ADR 0013: Separatrix-level Vχ signature for the 3D island gate

**Status:** Proposed

**Date:** 2026-08-23

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

The ε_κ=10⁻⁵ pressure result is independently threshold-robust: flattening widths at
0.95/0.97/0.99 of the integrable gradient are 0.1516/0.1594/0.1632 versus exact island
width 0.1451. Thus the ambiguity affects which Vχ claim is asserted, not whether the
pressure-flattening branch exists.

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

Choose **Option 3**. It is the only reading that preserves both nouns in §8.6 without
violating `−dV/dχ̂ = ρ(χ̂)`, and normalization by the local mollifier width makes the
window mesh-aware. The ADR should bind the exact window statistic and minimum
main/control margin after the current regenerated rows are scanned; no global
max/median statistic should return.

## Decision

**DECISION: pending human sign-off**
