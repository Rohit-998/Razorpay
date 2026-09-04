"""Who gets the credit when a failed payment finally goes through.

The eval harness will not count a rupee the system cannot prove it caused, and this
is the same rule on live traffic. Without it every recovery looks like a win: a
customer who was always going to pay on Wednesday pays on Wednesday, we happen to
have messaged them on Monday, and a naive pipeline books the whole amount as
recovered revenue. Run that for a quarter and the reported number is mostly other
people's intentions.

Three verdicts, and only the first is money the system may claim:

  `SYSTEM_RECOVERED` — the customer paid *on our link*. Razorpay tells us this
  directly: the link we created carries `reference_id = <the failed payment id>`,
  and `payment_link.paid` names the link. There is nothing to infer.

  `AMBIGUOUS` — they paid through their own channel within `AMBIGUITY_WINDOW_HOURS`
  of us contacting them. Our message may have been the reason, or they may have
  been on their way to the checkout page anyway, and nothing in the data separates
  those. Declining to claim it is the whole point: this is the bucket a dishonest
  system quietly folds into its wins.

  `CUSTOMER_SELF_RECOVERED` — they paid with no recent contact from us. Real
  revenue, correctly excluded from ours.

Deliberately pure, for the same reason `compliance.evaluate` is: every input is an
argument, so a verdict can be re-derived from its own audit row months later. It also
means the production rule and the simulator's `_resolve_self_recovery` can be read
side by side and checked for drift by eye — they are the same rule, and if they ever
disagree the measured lift stops predicting the live number.
"""

from __future__ import annotations

from datetime import datetime

AMBIGUITY_WINDOW_HOURS = 6.0
"""Kept identical to `app.sim.environment.AMBIGUITY_WINDOW_HOURS`, and asserted
equal in the tests. The eval's headline lift is computed under this window; if
production used a wider one it would book ambiguous rupees the report excluded, and
the two numbers would no longer be measuring the same thing."""

SYSTEM_RECOVERED = "SYSTEM_RECOVERED"
AMBIGUOUS = "AMBIGUOUS"
CUSTOMER_SELF_RECOVERED = "CUSTOMER_SELF_RECOVERED"


def attribute(
    *,
    paid_at: datetime,
    via_our_link: bool,
    last_contact_at: datetime | None,
) -> tuple[str, str]:
    """Return the verdict and the sentence that justifies it, for the audit row.

    The reason string is not decoration. An attribution is a claim about causation
    made by software, and the person auditing it a quarter later needs to see the
    number of hours it turned on rather than take the label on faith.
    """
    if via_our_link:
        return (
            SYSTEM_RECOVERED,
            "customer completed the payment on the link we sent, so causation is "
            "recorded rather than inferred",
        )
    if last_contact_at is None:
        return (
            CUSTOMER_SELF_RECOVERED,
            "customer paid through their own channel with no contact from us — "
            "real revenue, but not ours to claim",
        )
    hours = (paid_at - last_contact_at).total_seconds() / 3600.0
    if hours <= AMBIGUITY_WINDOW_HOURS:
        return (
            AMBIGUOUS,
            f"customer paid through their own channel {hours:.1f}h after we "
            f"contacted them, inside the {AMBIGUITY_WINDOW_HOURS:.0f}h window where "
            "causation is unprovable — not counted as a win",
        )
    return (
        CUSTOMER_SELF_RECOVERED,
        f"customer paid through their own channel {hours:.1f}h after our last "
        "contact, too long after to be attributable to it",
    )


def reward(verdict: str) -> float | None:
    """What the bandit is allowed to learn from an outcome, or `None` to learn nothing.

    Only a provable recovery is a reward. Paying an ambiguous outcome as a win would
    teach the bandit that contacting people who were about to pay anyway is the best
    strategy available — which is true if you score it that way, and worthless.

    It is not a loss either, and that is why this returns `None` rather than `0.0` for
    the ambiguous case. The arm did not fail; the experiment produced no readable
    result, and a Beta posterior updated with a zero has been told something false
    about the arm rather than nothing about it. A self-recovery *is* a zero: we acted,
    the customer paid through a channel of their own, and our action is the thing that
    did not work.
    """
    if verdict == SYSTEM_RECOVERED:
        return 1.0
    if verdict == CUSTOMER_SELF_RECOVERED:
        return 0.0
    return None
