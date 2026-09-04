"""Whether the live attribution rule matches the one the report is measured under.

The eval harness excludes ambiguous recoveries from the headline lift. If production
counted them, the number a merchant sees on the dashboard would be systematically
larger than the number in the report — not because the system got better, but because
the two are answering different questions. These tests pin them together.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.execution import attribution as A
from app.sim.environment import AMBIGUITY_WINDOW_HOURS as SIM_WINDOW

PAID_AT = datetime(2026, 3, 5, 14, 30)


def test_the_ambiguity_window_matches_the_simulator() -> None:
    """The one constant that has to agree across the two halves of the project.

    A wider window in production books ambiguous rupees the report excluded; a narrower
    one throws away wins the report counted. Either way the measured lift stops
    predicting the live figure, and the eval's whole claim is that it does."""
    assert A.AMBIGUITY_WINDOW_HOURS == SIM_WINDOW


def test_a_payment_on_our_link_is_ours() -> None:
    """The only verdict that needs no inference: Razorpay names the link."""
    verdict, why = A.attribute(
        paid_at=PAID_AT, via_our_link=True, last_contact_at=PAID_AT - timedelta(hours=1)
    )
    assert verdict == A.SYSTEM_RECOVERED
    assert "link we sent" in why


def test_a_payment_with_no_contact_is_not_ours() -> None:
    """Nothing was sent, so nothing can be claimed — however convenient the timing."""
    verdict, _ = A.attribute(paid_at=PAID_AT, via_our_link=False, last_contact_at=None)
    assert verdict == A.CUSTOMER_SELF_RECOVERED


def test_a_payment_just_inside_the_window_is_unprovable() -> None:
    """The bucket a dishonest system folds into its wins."""
    verdict, why = A.attribute(
        paid_at=PAID_AT,
        via_our_link=False,
        last_contact_at=PAID_AT - timedelta(hours=SIM_WINDOW - 0.1),
    )
    assert verdict == A.AMBIGUOUS
    assert "unprovable" in why


def test_a_payment_well_after_the_window_is_theirs() -> None:
    verdict, _ = A.attribute(
        paid_at=PAID_AT,
        via_our_link=False,
        last_contact_at=PAID_AT - timedelta(hours=SIM_WINDOW + 0.1),
    )
    assert verdict == A.CUSTOMER_SELF_RECOVERED


def test_the_window_boundary_is_inclusive_like_the_simulator() -> None:
    """`<=` in both places. A single-sided difference here is the kind of drift that
    survives review because it changes one payment in ten thousand and never crashes."""
    verdict, _ = A.attribute(
        paid_at=PAID_AT,
        via_our_link=False,
        last_contact_at=PAID_AT - timedelta(hours=SIM_WINDOW),
    )
    assert verdict == A.AMBIGUOUS


def test_only_a_provable_recovery_teaches_the_bandit_a_win() -> None:
    assert A.reward(A.SYSTEM_RECOVERED) == 1.0


def test_a_self_recovery_teaches_the_bandit_a_loss() -> None:
    """We acted and the customer paid elsewhere. The arm is what did not work."""
    assert A.reward(A.CUSTOMER_SELF_RECOVERED) == 0.0


def test_an_ambiguous_recovery_teaches_the_bandit_nothing() -> None:
    """`None`, not `0.0`, and the difference is the point.

    Scoring it a win would have the bandit learn that messaging people who were about to
    pay anyway is the best strategy available. Scoring it a loss tells a Beta posterior
    the arm failed, when the truth is that the experiment produced no readable result.
    """
    assert A.reward(A.AMBIGUOUS) is None


# ── There is one attribution rule, and this is it ──────────────────────────────────


def test_nothing_else_in_the_codebase_decides_attribution() -> None:
    """No second implementation of the verdict, anywhere.

    There used to be one. `app/audit/attribution.py` held an `AttributionEngine` that
    reached the verdict from the audit trail's last event type instead of from the clock:
    a `RETRY_ATTEMPTED` followed by any capture, from any source, at any later time, was
    booked `SYSTEM_RECOVERED` because the retry had "likely caused the recovery". It never
    referenced `AMBIGUITY_WINDOW_HOURS` at all, so the constant this file pins to the
    simulator did not constrain it.

    It had no callers — the webhook has always used the pure rule — which is exactly what
    made it dangerous. Dead code that implements the opposite of the project's central
    claim, in a file whose own docstring called itself "honest counting", is a paragraph
    of the README waiting to be falsified by anyone who opens it. The verdict is decided
    in one place, by a function whose every input is an argument, and this test is what
    keeps it that way.
    """
    import ast
    from pathlib import Path

    app = Path(A.__file__).parent.parent
    pure = Path(A.__file__).resolve()
    offenders = []

    for path in app.rglob("*.py"):
        if path.resolve() == pure or "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover — a broken file is another test's problem
            continue

        # Enum bodies declare the vocabulary; they do not decide which word applies.
        # `AttributionType` in the schemas and `AttributionTruth` in the simulator are both
        # legitimately lists of names, so their class bodies are lifted out before the scan.
        declarations = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(b, ast.Name) and b.id == "Enum" for b in node.bases
            ):
                declarations.update(
                    child.lineno for stmt in node.body for child in ast.walk(stmt)
                )

        for node in ast.walk(tree):
            # Assigning one of the verdict names is deciding attribution. Reading one
            # (`verdict == attribution.SYSTEM_RECOVERED`) is not, and has to stay legal:
            # the webhook branches on the result, and the dashboard groups by it.
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return)):
                continue
            if node.lineno in declarations:
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr in A.VERDICTS:
                    continue  # qualified — `attribution.SYSTEM_RECOVERED`, a read
                if isinstance(child, ast.Name) and child.id in A.VERDICTS:
                    offenders.append(f"{path.relative_to(app.parent)}:{node.lineno}")

    assert not offenders, (
        "attribution verdicts are produced outside app/execution/attribution.py at "
        + ", ".join(sorted(set(offenders)))
        + " — two rules for who gets the credit is how the dashboard and the report "
        "start disagreeing"
    )
