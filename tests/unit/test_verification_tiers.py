"""Machine-checkable enforcement of the ADR 0007 verification-tier contract.

ADR 0007 permits an expensive parameter ladder to leave the developer-runnable tiers
only if the same family keeps a live sentinel in the fast suite. That rule is what makes
the remote tier safe: without it, `exhaustive` becomes a place where evidence goes to
stop being run on every change. Enforcing it by review alone erodes, so it is checked
here, in the fast tier, on every run.

What is checked, statically, from the test sources:

1. every `exhaustive` test names its family with `@pytest.mark.sentinel("<family>")`;
2. every such family has at least one test that is neither `slow` nor `exhaustive`
   carrying the same `sentinel` marker.

What is deliberately *not* checked: that the fast sentinel exercises the same production
path, applies the same scientific gates, and is mutation-sensitive to the physics the
exhaustive rows claim. No static check can establish that; it is the reviewer's job
(`.claude/commands/review-milestone.md` item 6). This module makes the structural half
of the rule impossible to forget, not the scientific half impossible to fake.

The parse is static rather than a pytest collection because collecting the exhaustive
tier would mean importing and parameterizing hours of 3D setup inside a fast test.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parents[1]

#: Markers that remove a test from the fast suite selected by ``make test``.
_TIER_MARKERS = frozenset({"slow", "exhaustive"})


@dataclass(frozen=True)
class TierRecord:
    """One test function with the marker names and sentinel families that apply to it."""

    location: str
    markers: frozenset[str]
    families: frozenset[str]

    @property
    def is_fast(self) -> bool:
        """Return whether ``make test`` selects this test."""
        return not (self.markers & _TIER_MARKERS)


def _parse_mark(node: ast.expr) -> tuple[str, tuple[str, ...]] | None:
    """Return ``(name, string_arguments)`` for a ``pytest.mark.<name>`` expression."""
    arguments: tuple[str, ...] = ()
    if isinstance(node, ast.Call):
        arguments = tuple(
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        )
        node = node.func
    if not isinstance(node, ast.Attribute):
        return None
    owner = node.value
    if (
        isinstance(owner, ast.Attribute)
        and owner.attr == "mark"
        and isinstance(owner.value, ast.Name)
        and owner.value.id == "pytest"
    ):
        return node.attr, arguments
    return None


def _module_level_marks(tree: ast.Module) -> list[tuple[str, tuple[str, ...]]]:
    """Return the marks applied to every test in the module through ``pytestmark``."""
    marks: list[tuple[str, tuple[str, ...]]] = []
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in statement.targets
        ):
            continue
        value = statement.value
        entries = list(value.elts) if isinstance(value, (ast.List, ast.Tuple)) else [value]
        marks.extend(mark for entry in entries if (mark := _parse_mark(entry)) is not None)
    return marks


def collect_records(source: str, label: str) -> list[TierRecord]:
    """Return one :class:`TierRecord` per test function defined in ``source``."""
    tree = ast.parse(source)
    inherited = _module_level_marks(tree)
    records: list[TierRecord] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test"):
            continue
        marks = list(inherited)
        marks.extend(
            mark
            for decorator in node.decorator_list
            if (mark := _parse_mark(decorator)) is not None
        )
        records.append(
            TierRecord(
                location=f"{label}:{node.lineno} {node.name}",
                markers=frozenset(name for name, _ in marks),
                families=frozenset(
                    family
                    for name, arguments in marks
                    if name == "sentinel"
                    for family in arguments
                ),
            )
        )
    return records


def tier_violations(records: list[TierRecord]) -> list[str]:
    """Return the ADR 0007 structural violations present in ``records``.

    An empty list means every exhaustive test names a family and every such family is
    still exercised by at least one fast test.
    """
    violations = [
        f"{record.location}: marked `exhaustive` without @pytest.mark.sentinel(<family>)"
        for record in records
        if "exhaustive" in record.markers and not record.families
    ]

    exhaustive_families = {
        family for record in records if "exhaustive" in record.markers for family in record.families
    }
    fast_families = {family for record in records if record.is_fast for family in record.families}
    violations.extend(
        f"exhaustive family {family!r} has no fast live sentinel"
        for family in sorted(exhaustive_families - fast_families)
    )
    return violations


def _repository_records() -> list[TierRecord]:
    """Return the records for every test module in the suite."""
    records: list[TierRecord] = []
    for path in sorted(_TESTS_ROOT.rglob("test_*.py")):
        records.extend(
            collect_records(path.read_text(encoding="utf-8"), str(path.relative_to(_TESTS_ROOT)))
        )
    return records


def test_repository_satisfies_the_adr_0007_tier_contract() -> None:
    """Every exhaustive test names a family, and every family keeps a fast sentinel."""
    violations = tier_violations(_repository_records())
    assert not violations, "ADR 0007 tier violations:\n" + "\n".join(violations)


def test_the_suite_is_actually_being_scanned() -> None:
    """Guard against a silently empty scan making the contract check vacuous."""
    records = _repository_records()
    assert len(records) > 100
    assert any("slow" in record.markers for record in records)


# --- the checker's own tests -------------------------------------------------------
# The contract check above passes vacuously until Phase 6 adds the first exhaustive
# ladder. These synthetic cases keep it from being a test that cannot fail.

_MISSING_SENTINEL = """
import pytest

@pytest.mark.exhaustive
def test_full_ladder():
    pass
"""

_ORPHANED_FAMILY = """
import pytest

@pytest.mark.exhaustive
@pytest.mark.sentinel("reiman")
def test_full_ladder():
    pass

@pytest.mark.slow
@pytest.mark.sentinel("reiman")
def test_intermediate_ladder():
    pass
"""

_WELL_FORMED = """
import pytest

@pytest.mark.exhaustive
@pytest.mark.sentinel("reiman")
def test_full_ladder():
    pass

@pytest.mark.sentinel("reiman")
def test_smallest_row_and_isotropic_control():
    pass
"""

_MODULE_LEVEL_MARK = """
import pytest

pytestmark = [pytest.mark.exhaustive, pytest.mark.sentinel("reiman")]

def test_row_one():
    pass
"""


def test_exhaustive_without_a_sentinel_marker_is_a_violation() -> None:
    """The family must be declared, or the fast-sentinel rule cannot be checked at all."""
    violations = tier_violations(collect_records(_MISSING_SENTINEL, "synthetic.py"))
    assert len(violations) == 1
    assert "without @pytest.mark.sentinel" in violations[0]


def test_a_developer_slow_sentinel_does_not_satisfy_the_fast_requirement() -> None:
    """This is the mutation that matters: `slow` is not run on every change."""
    violations = tier_violations(collect_records(_ORPHANED_FAMILY, "synthetic.py"))
    assert violations == ["exhaustive family 'reiman' has no fast live sentinel"]


def test_a_family_with_a_fast_sentinel_is_accepted() -> None:
    """The intended arrangement must pass, or the check would only block good work."""
    assert tier_violations(collect_records(_WELL_FORMED, "synthetic.py")) == []


def test_module_level_pytestmark_is_honoured() -> None:
    """Sharded exhaustive rows usually carry their markers at module scope."""
    records = collect_records(_MODULE_LEVEL_MARK, "synthetic.py")
    assert records[0].markers == frozenset({"exhaustive", "sentinel"})
    assert records[0].families == frozenset({"reiman"})
    assert tier_violations(records) == ["exhaustive family 'reiman' has no fast live sentinel"]
