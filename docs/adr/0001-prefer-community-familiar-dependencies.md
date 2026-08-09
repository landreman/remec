# ADR 0001: Prefer Community-Familiar Dependencies

**Status:** Accepted

## Context

remec is intended for researchers and developers in the fusion and stellarator
communities. Dependency choices affect how easily they can install, inspect, reproduce,
and contribute to the project. A newer or more feature-rich package can be attractive,
but may impose an unfamiliar workflow with little benefit for this project.

## Options

1. Prefer established tools that are widely used and likely familiar to the target
   community, when they meet the project's technical requirements.
2. Prefer the newest or most feature-rich tool for each task.

## Tradeoffs

Option 1 reduces onboarding and maintenance friction, and makes project commands and
dependency management easier to recognize. It can forgo conveniences offered by newer
tools. Option 2 may improve some workflows, but increases the risk of requiring users
to learn tools that are not otherwise common in their work.

## Recommendation

Adopt option 1. Prefer `pip` for package installation and `make` for project commands
over newer alternatives such as `uv` and `just`, unless the alternative provides a
clear, project-specific benefit that established tools cannot meet. Evaluate every
dependency independently for technical fit, maintenance, licensing, platform support,
and compatibility with the project's base-dependency constraints.

DECISION: Option 1 approved