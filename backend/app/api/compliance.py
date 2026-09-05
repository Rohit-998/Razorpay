"""The limits in force, so the guardrails the dashboard claims can be checked.

The detail page has a panel headed "Safety checks applied". Before this endpoint existed it
listed three hand-written sentences — "One recovery action per customer in 24 hours",
"Respect consent and contact-time limits" — none of which was read from anything. Two of the
three were not even true: the limit is two contacts a day, not one, and there is no consent
field in the schema to respect. A compliance claim nobody can check against the running
service is the one kind of claim this project cannot afford to make, since compliant
escalation is half of what the brief asks to be shown.

So the panel reads from here, and here reads from `Settings`, which is what
`app.execution.compliance.evaluate` reads. Change a limit in the environment and the
sentence on the screen changes with it, because there is one source for both.

What this endpoint does *not* say is whether a limit was breached for a given payment. That
is per-decision and lives in the `COMPLIANCE_CHECKED` audit event, whose `blocked_by` holds
the refusal sentences with the numbers they were reached on. This is the rulebook; the audit
trail is the record of it being applied.
"""

from fastapi import APIRouter

from app.config import get_settings
from app.execution.compliance import (
    CHARGING_ACTIONS,
    CONTACTING_ACTIONS,
    RETRY_ACTIONS,
    method_ceiling_paise,
)
from app.models.schemas import PaymentMethod

router = APIRouter()


def _actions(actions) -> list[str]:
    """The action names a limit applies to, sorted so the response is stable."""
    return sorted(a.value for a in actions)


@router.get("/compliance/policy")
async def get_compliance_policy():
    """Every limit `evaluate()` enforces, with the value it is enforcing today.

    The `applies_to` list is the actual `frozenset` the rule is checked against in
    `compliance.py`, not a restatement of it — so an action added to `CONTACTING_ACTIONS`
    shows up here without anyone remembering to update a docstring.
    """
    s = get_settings()
    return {
        "note": (
            "These are the limits the compliance engine enforces on every action before "
            "it runs. A blocked action is not dropped: the engine returns a "
            "recommendation and the worker takes it, which is why the ladder's "
            "compliant policy still recovers money."
        ),
        "limits": [
            {
                "key": "recovery_window",
                "label": f"Stop trying {s.max_recovery_window_hours}h after the failure",
                "value": f"{s.max_recovery_window_hours}h",
                "applies_to": ["RETRY", "SEND_LINK", "ESCALATE"],
                "why": (
                    "A payment chased three days later is a customer who has moved on, "
                    "and the window is the stopping rule that ends the session."
                ),
            },
            {
                "key": "quiet_hours",
                "label": (
                    f"No contact between {s.quiet_hours_start}:00 and "
                    f"{s.quiet_hours_end}:00 IST"
                ),
                "value": f"{s.quiet_hours_start}:00–{s.quiet_hours_end}:00 IST",
                "applies_to": _actions(CONTACTING_ACTIONS),
                "why": (
                    "Judged in IST regardless of where the server runs. Escalation counts "
                    "as contact because an agent telephones the customer."
                ),
            },
            {
                "key": "contacts_per_day",
                "label": f"At most {s.max_contacts_per_day} contacts per customer per day",
                "value": s.max_contacts_per_day,
                "applies_to": _actions(CONTACTING_ACTIONS),
                "why": (
                    "Counted per customer per IST calendar day, so a batch cannot message "
                    "the same person once per failed payment."
                ),
            },
            {
                "key": "max_retries",
                "label": f"At most {s.max_retries_per_payment} auto-retries per payment",
                "value": s.max_retries_per_payment,
                "applies_to": _actions(RETRY_ACTIONS),
                "why": "The issuer sees repeated declines on a card as abuse.",
            },
            {
                "key": "retry_interval",
                "label": f"At least {s.min_retry_interval_minutes} minutes between retries",
                "value": f"{s.min_retry_interval_minutes}m",
                "applies_to": _actions(RETRY_ACTIONS),
                "why": (
                    "A retry inside the interval is a duplicate charge attempt, not a "
                    "second chance."
                ),
            },
            {
                "key": "mandate_ceiling",
                "label": (
                    f"No auto-retry above ₹{s.require_action_above_paise / 100:,.0f} "
                    "without a standing mandate"
                ),
                "value": s.require_action_above_paise,
                "applies_to": _actions(RETRY_ACTIONS),
                "why": (
                    "A large sum is not re-charged without the customer's say-so; a "
                    "mandate is that say-so, which is why autopay is exempt."
                ),
            },
            {
                "key": "rail_ceiling",
                "label": (
                    "Per-transaction ceiling of the rail — ₹"
                    f"{(method_ceiling_paise(PaymentMethod.UPI) or 0) / 100:,.0f} on UPI"
                ),
                "value": method_ceiling_paise(PaymentMethod.UPI),
                "applies_to": _actions(CHARGING_ACTIONS),
                "why": (
                    "NPCI's own limit. Charging above it fails at the rail, so attempting "
                    "it spends a retry to learn nothing."
                ),
            },
        ],
        "always_allowed": {
            "actions": ["WAIT", "GIVE_UP"],
            "why": (
                "Spending nothing cannot breach a limit on spending, and a stopping rule "
                "that could itself be blocked would be a policy with no way to stop."
            ),
        },
    }
