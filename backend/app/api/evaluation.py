"""The measurement, served over HTTP.

`python -m app.eval` writes `reports/report.json`, and until now nothing read it. The
strongest claim the project makes — ₹17.52 L of incremental lift per batch with a bootstrap
interval that excludes zero — lived in a file the product could not see, while
`/dashboard/stats` served a raw recovery rate that the report spends a page explaining is
misleading. A dashboard built on that would have contradicted its own evaluation.

These endpoints are read-only and deliberately dumb. They do no statistics: every number
here was computed by the harness, under fixed seeds, and is reproducible by anyone who
clones the repo. Recomputing anything at request time would mean the dashboard could
disagree with the report, and then neither would be evidence of anything.

The file is not committed as a fixture — it is regenerated, and the API surfaces its
`generated_at` so a stale report cannot masquerade as a live one.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

import structlog

logger = structlog.get_logger()
router = APIRouter()

REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "report.json"

_cache: dict[str, object] = {}


def _load() -> dict:
    """Read the report, re-reading only when the file has actually changed.

    Keyed on mtime rather than cached outright, because the demo flow regenerates the
    report while the API is up and a process-lifetime cache would serve the old numbers
    for as long as uvicorn stayed running.
    """
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No evaluation report yet. Generate it with `cd backend && "
                "python -m app.eval` — it needs no credentials and no network."
            ),
        )
    stamp = REPORT_PATH.stat().st_mtime
    if _cache.get("stamp") != stamp:
        with REPORT_PATH.open(encoding="utf-8") as fh:
            _cache["data"] = json.load(fh)
        _cache["stamp"] = stamp
        logger.info("eval.report_loaded", generated_at=_cache["data"].get("generated_at"))
    return _cache["data"]  # type: ignore[return-value]


POLICY_BLURBS = {
    "do_nothing": "The counterfactual. Every lift below is measured against this.",
    "naive_retry": "Retry until it works. What most recovery flows actually do.",
    "rules": "A good hand-written incumbent — the bar worth beating.",
    "payrevive": "This submission: value-priced actions under a compliance veto.",
    "oracle": "Reads hidden state. An upper bound, not a proposal.",
}

LADDER_ORDER = ["do_nothing", "naive_retry", "rules", "payrevive", "oracle"]

CONCERN_KEYS = [
    "quiet_hour_contacts",
    "invalid_actions",
    "engine_refused_actions",
    "episodes_at_step_cap",
    "self_inflicted_blocks",
]

CONCERN_LABELS = {
    "quiet_hour_contacts": "Messages sent in quiet hours",
    "invalid_actions": "Actions the gateway refused",
    "engine_refused_actions": "Actions compliance would refuse",
    "episodes_at_step_cap": "Payments it never stopped working",
    "self_inflicted_blocks": "Working instruments our retries broke",
}
"""Deliberately phrased as what happened to someone, not as a metric name. `invalid_actions`
is the gateway rejecting a charge we should have known was impossible; `engine_refused_actions`
is the compliance engine rejecting one we should have known was not allowed. Which of the two
a count lands in is the difference between a bug and a violation."""


def _won_of(value: object, seeds: int) -> dict[str, int] | None:
    """`seeds_beating_baseline` as `{count, of}`, whichever shape the report stored it in.

    The pooled block writes both halves. A per-scenario block writes the count alone, and its
    denominator is that scenario's seed list — so it is supplied here rather than left for the
    caller to guess, because a bare "20 seeds won" is unreadable without knowing whether 20 or
    40 were run. Reports written before either existed get `None`, which the reader can render
    as absent; serving `0 of 0` would read as "it won nothing".
    """
    if isinstance(value, dict):
        return {"count": int(value["count"]), "of": int(value["of"])}
    if isinstance(value, int):
        return {"count": value, "of": seeds}
    return None


@router.get("/eval/report")
async def full_report():
    """The whole report, exactly as the harness wrote it."""
    return _load()


@router.get("/eval/ladder")
async def ladder(scenario: str | None = None):
    """The policy ladder, chart-ready: lift with its interval, and whether it could ship.

    `scenario=None` returns the pooled figure over all five scenarios and twenty seeds,
    which is the headline. Naming a scenario returns that scenario alone, which is where
    the interesting disagreements live — `naive_retry` looks least bad on `outage_day`,
    where the failures are transient and retrying is nearly the right answer, and worst on
    `stress_dead_instruments`, where it is retrying cards that will never work again.
    """
    report = _load()
    per_scenario = report["by_scenario"]

    if scenario is not None:
        if scenario not in per_scenario:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown scenario. Available: {', '.join(per_scenario)}",
            )
        source = per_scenario[scenario]
    else:
        source = report["pooled"]

    rows = []
    for name in LADDER_ORDER:
        if name not in source:
            continue
        entry = source[name]
        lift = entry["incremental_lift_rupees"]
        row = {
            "policy": name,
            "blurb": POLICY_BLURBS.get(name, ""),
            "is_proposal": name == "payrevive",
            "is_ceiling": name == "oracle",
            "is_baseline": name == "do_nothing",
            "lift": lift,
            "share_of_achievable": entry["share_of_achievable_lift"],
            # Present on both shapes, because these three are the headline and a page
            # served the pooled ladder had nothing to put under it. `seeds_beating_baseline`
            # arrives as `{count, of}` pooled and as a bare count per scenario, where the
            # denominator is the seed list in `design`; normalised here so the reader does
            # not have to know which shape it asked for.
            "net_lift": entry.get("net_lift_rupees"),
            "regret_vs_ceiling": entry.get("regret_vs_oracle_rupees"),
            "seeds_beating_baseline": _won_of(
                entry.get("seeds_beating_baseline"), len(report["design"]["seeds"])
            ),
        }
        # Only the per-scenario rows carry the operational detail; the pooled block is
        # intervals alone, because summing action counts across scenarios of different
        # sizes produces a number that means nothing. `/eval/shippability` is where the
        # pooled counts live, and it sums them for a documented reason: a defect is a
        # defect wherever it happened.
        if scenario is not None:
            row.update(
                {
                    "totals": entry["totals"],
                    "concerns": entry["concerns"],
                    "hard_limits": entry["hard_limits"],
                    "self_inflicted_block_rate": entry["self_inflicted_block_rate"],
                    "shippable": entry["shippable"],
                }
            )
        rows.append(row)

    return {
        "scenario": scenario,
        "scenarios_available": list(per_scenario),
        "generated_at": report["generated_at"],
        "design": report["design"],
        "policies": rows,
    }


@router.get("/eval/shippability")
async def shippability():
    """The gates, pooled over every batch — the table that can invalidate the money.

    Four of these five columns are gates: zero is attainable for each, so any count above
    zero is a defect that no amount of lift buys back. The fifth, blocking a working
    instrument, is a cost — a failed retry can kill a live card whatever the reason for the
    failure, so the only policy that blocks nothing is one that retries nothing. It is
    therefore reported as a rate against the instruments that were alive to be broken, and
    judged against the incumbent rather than against zero.

    Counts are summed across scenarios here, unlike `/eval/ladder`, because a defect is a
    defect wherever it happened — a quiet-hour message on `outage_day` is not cancelled out
    by a clean run on `baseline`. Rupee lift is not summed this way, which is why the
    pooled ladder carries intervals and no action counts.
    """
    report = _load()
    per_scenario = report["by_scenario"]

    rows = []
    for name in LADDER_ORDER:
        totals = {key: 0 for key in CONCERN_KEYS}
        contacts = actions = live = 0
        hard_limits: list[str] = []
        for scenario in per_scenario.values():
            if name not in scenario:
                continue
            entry = scenario[name]
            hard_limits = entry["hard_limits"]
            contacts += entry["totals"]["contacts"]
            live += entry["totals"]["live_instrument_payments"]
            # Absent from reports generated before the denominator was exported. Serving 0
            # would read as "no actions were checked", which is the opposite of the truth.
            actions += entry["totals"].get("actions_needing_approval") or 0
            for key, value in entry["concerns"].items():
                totals[key] = totals.get(key, 0) + value
        if not hard_limits:
            continue

        blocks = totals["self_inflicted_blocks"]
        failed_gates = [key for key in hard_limits if totals.get(key)]
        rows.append(
            {
                "policy": name,
                "blurb": POLICY_BLURBS.get(name, ""),
                "is_proposal": name == "payrevive",
                "is_ceiling": name == "oracle",
                "gates": [
                    {
                        "key": key,
                        "label": CONCERN_LABELS[key],
                        "count": totals[key],
                        "of": {
                            "quiet_hour_contacts": contacts,
                            "engine_refused_actions": actions,
                        }.get(key),
                        "passed": totals[key] == 0,
                    }
                    for key in hard_limits
                ],
                "harm": {
                    "label": CONCERN_LABELS["self_inflicted_blocks"],
                    "count": blocks,
                    "of": live,
                    "rate": round(blocks / live, 6) if live else 0.0,
                },
                "failed_gates": failed_gates,
                # The ceiling is excluded from the verdict rather than failed by it. It reads
                # hidden state, so calling it unshippable would be measuring the wrong thing:
                # it was never a proposal.
                "verdict": (
                    "not a proposal" if name == "oracle"
                    else "fails a gate" if failed_gates
                    else "shippable"
                ),
            }
        )

    return {
        "generated_at": report["generated_at"],
        "batches_run": report["design"]["batches_run"],
        "gate_keys": [k for k in CONCERN_KEYS if k != "self_inflicted_blocks"],
        "policies": rows,
    }


CAUSE_NOTES = {
    "INSUFFICIENT_FUNDS": (
        "The account is short. A drained account is drained on every rail, so waiting for "
        "the salary credit is the strategy — and retrying meanwhile can get the card blocked."
    ),
    "BANK_DOWNTIME": (
        "The bank is down for a window. The one cause where retrying is nearly right, "
        "provided it waits for the window to close instead of hammering through it."
    ),
    "NETWORK_TRANSIENT": (
        "A gateway blip with nothing behind it. Cheap to retry and it usually works, "
        "which is why a policy that only retries still recovers something."
    ),
    "AUTH_TIMEOUT": (
        "Someone walked away from an OTP screen. Nothing we do server-side authenticates; "
        "a link the customer opens themselves does."
    ),
    "WRONG_CREDENTIALS": (
        "The details on file are stale. Retrying them fails identically every time — only "
        "outreach, where the customer types them in fresh, changes the answer."
    ),
    "PERMANENT_DECLINE": (
        "The instrument is dead at failure time. No retry on that rail ever succeeds, and "
        "every attempt is pure spend. This is what a stopping rule is for."
    ),
    "MERCHANT_ERROR": (
        "Our own configuration, broken in bursts. Not the customer's to fix, and messaging "
        "them about it is worse than useless until it is repaired."
    ),
}
"""Why each cause responds to what it responds to, in the simulator's own terms.

Written against `_charge_succeeds`, which is the only thing that decides an outcome: is the
instrument alive, are the stored details valid, is our configuration working, is the money
there, is the bank up. Each note names which of those five a cause turns off, because that
is what makes one action right and another expensive — not the label itself, which no policy
is allowed to see.
"""


@router.get("/eval/causes")
async def causes(policy: str = "payrevive"):
    """One policy against the incumbent and the ceiling, broken down by true root cause.

    This is the view that shows *where* the lift comes from, and it is the only place in the
    API where the hidden label is used. That is legitimate here and nowhere else: the
    breakdown is computed after the fact by the simulator, which knows why each payment
    failed, and no policy was allowed to read it while deciding. It answers the question a
    reviewer actually has — is the money coming from the causes where good judgement is
    possible, or from the ones where anything would have worked?

    Rupees are summed across scenarios rather than averaged, because a seed that produced
    three `MERCHANT_ERROR` payments should not weigh the same as one that produced forty.
    """
    report = _load()
    per_scenario = report["by_scenario"]
    reference = ["rules", policy, "oracle"]
    tracked = [p for p in dict.fromkeys(reference) if _has_policy(per_scenario, p)]
    if policy not in tracked:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown policy. Available: {', '.join(LADDER_ORDER)}",
        )

    pooled: dict[str, dict] = {}
    for scenario in per_scenario.values():
        for name in tracked:
            if name not in scenario:
                continue
            for cause, slice_ in scenario[name]["by_cause"].items():
                row = pooled.setdefault(
                    cause,
                    {
                        "cause": cause,
                        "note": CAUSE_NOTES.get(cause, ""),
                        "payments": 0,
                        "at_risk_rupees": 0.0,
                        "by_policy": {n: {"recovered_rupees": 0.0, "system_recovered_rupees": 0.0,
                                          "spend_rupees": 0.0, "escalations": 0} for n in tracked},
                    },
                )
                # Payment counts and rupees at risk are properties of the batch, not of the
                # policy — identical across the ladder by construction, since every policy
                # sees the same batch. Counted from the requested policy alone so they are
                # not tripled.
                if name == policy:
                    row["payments"] += slice_["payments"]
                    row["at_risk_rupees"] += slice_["at_risk_rupees"]
                bucket = row["by_policy"][name]
                bucket["recovered_rupees"] += slice_["recovered_rupees"]
                bucket["system_recovered_rupees"] += slice_["system_recovered_rupees"]
                bucket["spend_rupees"] += slice_["spend_rupees"]
                bucket["escalations"] += slice_["escalations"]

    rows = []
    for row in pooled.values():
        at_risk = row["at_risk_rupees"]
        for bucket in row["by_policy"].values():
            for key in ("recovered_rupees", "system_recovered_rupees", "spend_rupees"):
                bucket[key] = round(bucket[key], 2)
            # Share of the money at risk in this cause, which is comparable across causes.
            # A raw rupee total is not: `INSUFFICIENT_FUNDS` is a third of the batch and
            # `MERCHANT_ERROR` a few dozen payments.
            bucket["share_of_at_risk"] = (
                round(bucket["recovered_rupees"] / at_risk, 4) if at_risk else 0.0
            )
        row["at_risk_rupees"] = round(at_risk, 2)
        rows.append(row)

    rows.sort(key=lambda r: r["at_risk_rupees"], reverse=True)
    return {
        "generated_at": report["generated_at"],
        "policy": policy,
        "compared_with": [n for n in tracked if n != policy],
        "causes": rows,
    }


def _has_policy(per_scenario: dict, policy: str) -> bool:
    return any(policy in scenario for scenario in per_scenario.values())
