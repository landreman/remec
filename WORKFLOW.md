# The workflow

Setup is roughly a half day, once. After that your entire involvement per milestone is:
type one line, answer ADR questions when they come up, click merge.

---

## One-time setup

### 1. Repository

```
remec/
├── AGENTS.md                            # provided
├── CLAUDE.md                            # symlink → AGENTS.md
├── WORKFLOW.md                          # this file, for you not the agents
├── Makefile
├── pyproject.toml
├── docs/
│   ├── DESIGN.md
│   ├── 20260815-01_Regularized_3D_MHD_equilibrium.tex
│   ├── STATUS.md                        # provided
│   ├── dev_notes.md                     # empty
│   ├── verification.md                  # empty
│   └── adr/
├── .agents/skills/milestone/SKILL.md    # provided — Codex
└── .claude/commands/review-milestone.md # provided — Claude Code
```

```bash
ln -s AGENTS.md CLAUDE.md
```

One contract, both agents. When you correct something, you correct it once.

### 2. Codex config

`~/.codex/config.toml`:

```toml
model = "gpt-5.5"                    # or whatever is top of your /model picker
model_reasoning_effort = "high"
plan_mode_reasoning_effort = "xhigh"

[profiles.grind]                     # boilerplate, packaging, I/O readers
model_reasoning_effort = "medium"
```

Set approvals to full-auto within the workspace. In the app: Settings → Permissions →
allow file edits and command execution inside the project, deny network. In the CLI:
`--sandbox workspace-write` with approvals never. The sandbox is what makes this safe to
leave unattended; `--yolo` disables it, so don't.

Verify Codex can't reach the network during a run except for the package index during
setup. Agents that can search the web will find a Stack Overflow answer about anisotropic
diffusion and quietly follow it instead of the note.

### 3. Bootstrap Phase 0 yourself, interactively

Do not run 0.1 unattended. Sit with it, because the environment it produces is the floor
everything else stands on: NGSolve wheel pinned, `make check` working, CI green on Linux
and macOS. An hour here saves every subsequent milestone from re-litigating the venv.

### 4. Claude Code PR review

```bash
claude
> /install-github-app
```

Pick automatic-on-open. It writes `.github/workflows/claude-code-review.yml` and stores
the auth secret. Then edit the workflow's prompt to point at your command:

```yaml
prompt: "/review-milestone"
```

Set `claude_args: --model opus` — this review is the one place you want the strongest
model, because it is the only thing standing between a wrong sign and a merged wrong
sign.

If you'd rather not spend on every push, change the trigger to `types: [opened]` only,
or drop the Action and run `claude` locally on the branch before merging.

---

## The daily loop

**You:** open Codex in the repo, type

```
/milestone next
```

or `/milestone 4.2` to pick one. That is the whole prompt. Everything else lives in
`AGENTS.md`, the skill, and `STATUS.md`.

**Codex:** branches, writes tests, implements, verifies mutations, updates the completed
milestone's `STATUS.md` row to `[x]` in that same PR, and opens a PR. `[x]` means the
milestone is complete in the submitted PR (or merged), so it is not changed again after
merge. Or stops and drafts an ADR.

**CI:** runs `make check` on Linux and macOS.

**Claude:** reviews the PR against `DESIGN.md` and the note, posts findings inline.

**You:** if there are findings, comment `@codex address the review` on the PR. If not,
merge. If an ADR is open, that's your decision to make — write the answer into the ADR,
commit it, comment `@codex the ADR is resolved, continue`.

**You never:** write a bespoke prompt for a sub-step, approve individual file writes, or
explain the design document to the agent again.

---

## Model and effort

| Work | Model | Effort |
|---|---|---|
| Phase 0, packaging, I/O readers (6.3, 6.4), diagnostics | top model | medium |
| Everything numerical — Phases 1–5, 7, 8 | top model | high |
| 1.3, 3.2, 4.2, 4.4, 8.3, 8.4 | top model | xhigh |
| Planning a phase, or debugging a wrong convergence rate | top model, `/plan` | xhigh |
| Review | Claude Opus, extended thinking on | — |

Codex model names churn every few months — gpt-5.5 as of the current docs, with a 5.6
family (sol / terra / luna) rolling out. Rather than pinning a name that will be stale,
take whatever sits at the top of `/model` and use `-mini`/`luna`-class models only for
subagents. Effort matters more than model choice here: the failure mode on this codebase
is a plausible-looking weak form with a wrong sign, and that is exactly what more
deliberation buys you.

The six xhigh milestones are the ones where a wrong answer looks right. 1.3 is the phase
gate the whole project rests on. 3.2's stabilization parameter has no obviously-wrong
value. 4.2 and 4.4 are where discrete exactness either holds at roundoff or silently
doesn't. 8.3 and 8.4 are Jacobians that converge to the wrong fixed point if a term is
missing.

---

## Surfaces

**Codex desktop app** for driving. It manages worktrees for you, runs two or three
milestones side by side, and gives you a diff view for the ones you want to skim. You
don't touch git.

**Codex CLI** for anything you want to script — a nightly batch, a re-run of the full
verification suite, `codex exec` in a loop.

**IDE extension:** skip it. It's built around watching the agent work, which is the
opposite of what you asked for. Install it if you want to hand-edit alongside, not as the
main surface.

**Codex cloud:** skip it for this project. NGSolve is a heavy native dependency and the
nightly benchmarks want your own CPU.

**Claude Code** only through the GitHub Action, plus the occasional local `claude` when a
review thread gets long enough that you want to talk to it.

---

## Branches or worktrees

Branch per milestone, always — `DESIGN.md` §25 sizes each one for a single reviewable PR
and that maps exactly onto a PR.

Worktrees only when you're running agents in parallel, which for this plan means:
Phase 7 alongside Phase 8, or 6.3 and 6.4 (the two independent readers) at once, or a
long nightly benchmark while you continue on something else. Two or three at a time.
More than that and you become the bottleneck at review, which defeats the purpose.

The app handles worktree creation. If you're doing it by hand:

```bash
git worktree add ../remec-7.1 -b milestone/7.1-ap-prototype
cd ../remec-7.1 && python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"
```

Use a Python 3.10-or-newer `python3` interpreter, matching the supported project range.

**The gotcha:** each worktree needs its own venv. `pip install -e .` records an absolute
path, so a shared venv will silently point every worktree at whichever directory
installed last, and you will spend an afternoon debugging an agent that keeps "fixing"
code it isn't running. NGSolve wheels are large; budget the disk.

Don't run two agents on milestones in the same phase. They'll collide in
`src/remec/fem/` and you'll spend more time on merge conflicts than you saved.

---

## Where you stay in the loop

The design document already defines this, which is why the workflow can be this thin: an
ADR is the human-decision primitive. The agent's instruction is to draft one and stop
rather than choose. So the ADRs *are* your queue, and `docs/adr/` is the only place you
need to look to know whether anything wants you.

You should expect ADRs at 7.1 (mandatory, the AP strategy), 10.4 (mandatory, physics
sign-off), 9 (the PETSc decision), and opportunistically wherever the note and the
design document disagree — Appendix A suggests there are seams there.

Two other places to insert yourself, both cheap:

**Phase gates.** 1.3 and 5.4 are the two milestones where a green CI could be green for
the wrong reason and everything downstream inherits the error. Read those PRs yourself,
slowly, alongside Claude's review. They're the only two where I'd spend your own hour.

**Weekly.** Skim `docs/dev_notes.md` and the merged PR list. If you find yourself giving
the same review note twice, put it in `AGENTS.md` and it stops recurring. That feedback
loop is what keeps the one-line prompt working as the codebase grows.

---

## Honest limits

Agents on numerical physics code fail in a specific way: they produce something that
runs, converges, and passes tests that don't constrain the physics. `make check` going
green is weak evidence. The design document's insistence on manufactured solutions with
predicted rates is what makes this tractable — a wrong operator won't hit second order —
which is why the skill puts tests first and requires mutation checks, and why the review
command spends most of its attention on whether the test could fail.

Take the mutation-verification step seriously. If you drop one thing from this workflow,
don't let it be that one.

Expect to hand-hold 0.1, 1.3, 4.2, and 4.4 more than the rest. Expect Phase 6's mesh work
(6.2) to be slower than its one line in §25 suggests. Expect at least one ADR where the
agent has genuinely found something the design document didn't settle — that's the system
working, not failing.
