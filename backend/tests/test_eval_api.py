"""The API cannot be allowed to disagree with the report.

Two failure modes, both of which existed in this repo an hour ago and neither of which any
test would have caught.

The first: the measurement is unreachable. `python -m app.eval` wrote `reports/report.json`
and nothing read it, so the strongest claim the project makes — a bootstrap interval on
₹17.52 L of lift that excludes zero — was invisible to the product. A dashboard in that
state has to invent its own numbers, and it did.

The second, worse: the API reported a metric the report argues against. `/dashboard/stats`
served `recovered / total * 100`, a recovery rate, while `REPORT.md` spent a page explaining
that a recovery rate credits the system with every customer who would have paid anyway. Open
both at once and the submission contradicts itself.

So these tests assert two properties above all: that every number served is the number the
harness computed, and that no endpoint reports a bare recovery rate again.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException

from app.api import dashboard, evaluation, metrics
from app.db import database
from app.eval.harness import PolicyOnScenario
from app.execution import attribution

REPORT = json.loads(evaluation.REPORT_PATH.read_text(encoding="utf-8"))
"""Read directly, so every assertion below compares the endpoint against the file rather
than against a number I typed in. A test with the expected lift hardcoded would pass
whether or not the endpoint was reading the report at all."""


def _run(coro):
    return asyncio.run(coro)


# ── The measurement is served, and served unchanged ───────────────────────────


def test_the_full_report_is_passed_through_untouched() -> None:
    """`/eval/report` is the escape hatch: whatever the harness wrote, byte for byte.

    It exists so that no reviewer has to trust the shaped endpoints. If `/eval/ladder`
    ever disagreed with the report, this is the endpoint that would prove it.
    """
    assert _run(evaluation.full_report()) == REPORT


def test_the_ladder_reports_the_lift_the_harness_computed() -> None:
    """No arithmetic at request time — the property the whole module is built around.

    Recomputing a mean here would produce a dashboard that can drift from the report while
    both look authoritative, and then neither is evidence of anything.
    """
    served = {row["policy"]: row["lift"] for row in _run(evaluation.ladder())["policies"]}
    for policy, entry in REPORT["pooled"].items():
        assert served[policy] == entry["incremental_lift_rupees"]


def test_the_proposal_sits_between_the_incumbent_and_the_ceiling() -> None:
    """The shape of the claim, asserted so a regression in the policy shows up here too."""
    rows = {r["policy"]: r for r in _run(evaluation.ladder())["policies"]}
    assert rows["rules"]["lift"]["mean"] < rows["payrevive"]["lift"]["mean"]
    assert rows["payrevive"]["lift"]["mean"] < rows["oracle"]["lift"]["mean"]
    assert rows["payrevive"]["lift"]["excludes_zero"] is True
    assert rows["payrevive"]["is_proposal"] and rows["oracle"]["is_ceiling"]


def test_the_ceiling_is_marked_as_a_ceiling_everywhere_it_appears() -> None:
    """`oracle` reads hidden state. An endpoint that returned it as one policy among five,
    with no flag, would invite a reader to compare it as a competitor — and it would win."""
    ladder = {r["policy"]: r for r in _run(evaluation.ladder())["policies"]}
    ship = {r["policy"]: r for r in _run(evaluation.shippability())["policies"]}
    assert ladder["oracle"]["is_ceiling"] is True
    assert ship["oracle"]["verdict"] == "not a proposal"
    assert "not a proposal" in ship["oracle"]["verdict"]


def test_a_named_scenario_carries_the_operational_detail() -> None:
    """Per-scenario rows add action counts; the pooled row deliberately does not, because
    summing actions across scenarios of different sizes produces a meaningless number."""
    pooled = _run(evaluation.ladder())["policies"][0]
    scoped = _run(evaluation.ladder("stress_dead_instruments"))["policies"][0]
    assert "totals" not in pooled and "concerns" not in pooled
    assert "totals" in scoped and "concerns" in scoped and "shippable" in scoped


# ── The headline the page renders, on the response the page asks for ──────────


def test_the_pooled_row_carries_the_three_figures_the_headline_is_made_of() -> None:
    """The bug this locks down was invisible in the API and total in the product.

    `net_lift`, `regret_vs_ceiling` and `seeds_beating_baseline` existed only on
    scenario-scoped rows, and the analytics page reads the pooled response — so the
    measurement page's headline number, the one figure the whole submission rests on,
    rendered as an em dash on every load, with an empty section under it.

    Asserted on the pooled read specifically, because that is the request the page makes.
    """
    rows = {r["policy"]: r for r in _run(evaluation.ladder())["policies"]}
    proposal = rows["payrevive"]

    for key in ("net_lift", "regret_vs_ceiling", "seeds_beating_baseline"):
        assert proposal[key] is not None, f"{key} is what the headline renders"

    pooled = REPORT["pooled"]["payrevive"]
    assert proposal["net_lift"] == pooled["net_lift_rupees"]
    assert proposal["regret_vs_ceiling"] == pooled["regret_vs_oracle_rupees"]


def test_net_lift_is_the_gross_figure_minus_what_the_actions_cost() -> None:
    """Two numbers a page can put under one label. Net is the smaller one, and the gap is
    spend — so if they were ever equal, spend would have stopped being counted."""
    rows = {r["policy"]: r for r in _run(evaluation.ladder())["policies"]}
    proposal = rows["payrevive"]

    assert proposal["net_lift"]["mean"] < proposal["lift"]["mean"]
    assert proposal["net_lift"]["excludes_zero"] is True
    # Spend is small against the lift, which is the point of pricing actions — but it is
    # not zero, and a net figure that equalled the gross would mean it was being dropped.
    assert 0 < proposal["lift"]["mean"] - proposal["net_lift"]["mean"] < 50_000


def test_the_win_count_arrives_with_the_number_of_batches_behind_it() -> None:
    """`20` is excellent out of 20 and a coin flip out of 40.

    Pooled, the denominator is every batch the policy faced — five scenarios by twenty
    seeds — not the seed list, which is what a page would have to assume if the count
    travelled alone.
    """
    rows = {r["policy"]: r for r in _run(evaluation.ladder())["policies"]}
    won = rows["payrevive"]["seeds_beating_baseline"]
    design = REPORT["design"]

    assert won["of"] == len(design["scenarios"]) * len(design["seeds"])
    assert won["count"] == won["of"], "it beats doing nothing on every batch it ran"
    assert rows["do_nothing"]["seeds_beating_baseline"]["count"] == 0, "it is the baseline"


def test_a_scoped_win_count_is_given_the_seed_list_as_its_denominator() -> None:
    """Per-scenario the report stores a bare count, because the denominator there is that
    scenario's seeds. Normalising it in the API means the reader does not have to know which
    shape it asked for to render the same widget."""
    scoped = {r["policy"]: r for r in _run(evaluation.ladder("baseline"))["policies"]}
    won = scoped["payrevive"]["seeds_beating_baseline"]

    assert won["of"] == len(REPORT["design"]["seeds"])
    assert won["count"] == REPORT["by_scenario"]["baseline"]["payrevive"][
        "seeds_beating_baseline"
    ]


def test_a_report_without_the_pooled_figures_serves_absence_not_zero() -> None:
    """Reports written before this existed have a pooled block with three keys.

    Serving `0 of 0` for them would read as "it won nothing", which is a claim; `None` is
    the truth, and the page renders it as a dash. The alternative — a `KeyError` — would
    take the whole measurement page down over a stale artifact.
    """
    assert evaluation._won_of(None, 20) is None
    assert evaluation._won_of(17, 20) == {"count": 17, "of": 20}
    assert evaluation._won_of({"count": 100, "of": 100}, 20) == {"count": 100, "of": 100}


def test_an_unknown_scenario_names_the_ones_that_exist() -> None:
    with pytest.raises(HTTPException) as caught:
        _run(evaluation.ladder("no_such_scenario"))
    assert caught.value.status_code == 404
    assert "baseline" in caught.value.detail


def test_every_scenario_in_the_report_is_reachable() -> None:
    for scenario in REPORT["by_scenario"]:
        assert _run(evaluation.ladder(scenario))["scenario"] == scenario


# ── The gates, and the fact that a gate is not a rate ─────────────────────────


def test_the_proposal_passes_every_gate() -> None:
    """The claim that makes the money bankable, restated as a test.

    A gate is a limit where zero is attainable, so any count above zero is a defect that no
    amount of lift buys back. `payrevive` is the only policy on the ladder with a clean
    sheet, and if that ever stops being true the headline needs rewriting, not the test.
    """
    rows = {r["policy"]: r for r in _run(evaluation.shippability())["policies"]}
    assert rows["payrevive"]["failed_gates"] == []
    assert rows["payrevive"]["verdict"] == "shippable"
    assert all(gate["passed"] for gate in rows["payrevive"]["gates"])


def test_the_incumbent_and_the_naive_policy_both_fail_a_gate() -> None:
    """Not incidental colour. `rules` is a competent hand-written policy and it still takes
    2,445 actions the deployed engine would refuse — which is the argument for pricing
    actions under a veto rather than enumerating cases."""
    rows = {r["policy"]: r for r in _run(evaluation.shippability())["policies"]}
    assert "engine_refused_actions" in rows["rules"]["failed_gates"]
    assert rows["naive_retry"]["failed_gates"]
    assert rows["rules"]["verdict"] == "fails a gate"


def test_the_gates_served_are_the_gates_the_harness_enforces() -> None:
    """Read off `PolicyOnScenario.HARD_LIMITS` rather than listed here.

    The bug this prevents is quiet: someone adds a sixth hard limit to the eval, the API
    keeps rendering five, and a policy that fails the new one still shows as shippable in
    the product. The dashboard would be wrong in the safe-looking direction.
    """
    served = _run(evaluation.shippability())
    assert served["gate_keys"] == list(PolicyOnScenario.HARD_LIMITS)
    for row in served["policies"]:
        assert [g["key"] for g in row["gates"]] == list(PolicyOnScenario.HARD_LIMITS)


def test_a_refusal_count_arrives_with_its_denominator() -> None:
    """`8,800 refusals` and `8,800 of 46,174` are different claims, and only one of them can
    be checked. The count is meaningless without the number of actions it was drawn from."""
    rows = {r["policy"]: r for r in _run(evaluation.shippability())["policies"]}
    refused = next(
        g for g in rows["naive_retry"]["gates"] if g["key"] == "engine_refused_actions"
    )
    assert refused["count"] > 0
    assert refused["of"] and refused["of"] >= refused["count"]


def test_harm_is_a_rate_against_what_was_alive_to_be_broken() -> None:
    """Blocking a working card is the one entry here that is not a gate: a failed retry can
    kill a live instrument whatever the reason for the failure, so the only policy that
    blocks nothing is one that retries nothing. It is judged against the incumbent."""
    rows = {r["policy"]: r for r in _run(evaluation.shippability())["policies"]}
    proposal, incumbent = rows["payrevive"]["harm"], rows["rules"]["harm"]
    assert proposal["of"] == incumbent["of"], "same batch, so the same denominator"
    assert proposal["rate"] < incumbent["rate"], "more money and less harm, or it is a trade"
    assert 0 < proposal["rate"] < 0.01


def test_defects_are_summed_across_scenarios_rather_than_averaged() -> None:
    """A quiet-hour message on one scenario is not cancelled out by a clean run on another.

    Rupee lift is pooled with an interval instead, which is why the two endpoints aggregate
    differently — and this asserts the shippability side really is a sum.
    """
    served = {r["policy"]: r for r in _run(evaluation.shippability())["policies"]}
    expected = sum(
        REPORT["by_scenario"][s]["naive_retry"]["concerns"]["quiet_hour_contacts"]
        for s in REPORT["by_scenario"]
    )
    quiet = next(
        g for g in served["naive_retry"]["gates"] if g["key"] == "quiet_hour_contacts"
    )
    assert quiet["count"] == expected


# ── The by-cause view, which is the only place the hidden label is used ───────


def test_the_batch_is_counted_once_and_not_once_per_policy() -> None:
    """Three policies are pooled into each cause row. Payment counts and rupees at risk are
    properties of the batch — identical across the ladder by construction — so adding them
    up per policy would treble every figure and inflate the denominator of every share."""
    served = _run(evaluation.causes())
    scenarios = REPORT["by_scenario"]
    expected = {
        cause: sum(
            scenarios[s]["payrevive"]["by_cause"][cause]["payments"] for s in scenarios
        )
        for cause in scenarios["baseline"]["payrevive"]["by_cause"]
    }
    assert served["causes"], "the report has causes; the endpoint returned none"
    for row in served["causes"]:
        assert row["payments"] == expected[row["cause"]]
        assert len(row["by_policy"]) == 3, "rules, payrevive, oracle"


def test_the_proposal_beats_the_incumbent_on_the_causes_that_need_judgement() -> None:
    """Where the lift actually comes from.

    `NETWORK_TRANSIENT` is the cause where anything works — a bare retry recovers most of
    it, and the gap between policies there is small. `PERMANENT_DECLINE` is the opposite:
    the instrument is dead, no retry ever succeeds, and the only way to recover money is to
    reach the customer on something else. If the lift were coming from the easy causes it
    would not be evidence of judgement, so this asserts it is not.
    """
    rows = {r["cause"]: r for r in _run(evaluation.causes())["causes"]}
    easy = rows["NETWORK_TRANSIENT"]["by_policy"]
    hard = rows["PERMANENT_DECLINE"]["by_policy"]
    easy_gap = easy["payrevive"]["share_of_at_risk"] - easy["rules"]["share_of_at_risk"]
    hard_gap = hard["payrevive"]["share_of_at_risk"] - hard["rules"]["share_of_at_risk"]
    assert hard_gap > easy_gap * 3


def test_every_cause_the_simulator_emits_arrives_with_an_explanation() -> None:
    """A row labelled `WRONG_CREDENTIALS` with an empty note tells a reader nothing about why
    outreach beats retrying there. An eighth cause added to the simulator fails this."""
    unexplained = [
        row["cause"] for row in _run(evaluation.causes())["causes"] if not row["note"]
    ]
    assert not unexplained, f"causes with no note: {unexplained}"


def test_an_unknown_policy_is_refused_rather_than_returned_empty() -> None:
    with pytest.raises(HTTPException) as caught:
        _run(evaluation.causes("not_a_policy"))
    assert caught.value.status_code == 404


def test_the_shares_are_of_money_at_risk_not_of_payment_count() -> None:
    """`INSUFFICIENT_FUNDS` is the largest cause by count and mid-sized by rupees; comparing
    causes on counts would make the small-basket ones look more important than they are."""
    for row in _run(evaluation.causes())["causes"]:
        for bucket in row["by_policy"].values():
            assert 0.0 <= bucket["share_of_at_risk"] <= 1.0
            if row["at_risk_rupees"]:
                assert bucket["share_of_at_risk"] == pytest.approx(
                    bucket["recovered_rupees"] / row["at_risk_rupees"], abs=1e-4
                )


# ── No endpoint reports a bare recovery rate again ────────────────────────────


class _Chain:
    """Every Supabase call shape these two modules use, in one object.

    `select().eq().range().execute()` for the paged aggregate reads, and
    `select().eq().order().limit().execute()` for the exception queue.

    The filters really filter. A double that accepted `.eq("event_type", ...)` and ignored it
    would keep passing if someone deleted that call from the endpoint — and an unrelated audit
    event mistaken for an attribution observation is exactly the kind of quiet miscount these
    tests exist to catch. `order` is a no-op because the fixture is already in the order the
    endpoint asks for, which is the one thing here worth not pretending about.
    """

    def __init__(self, rows: list[dict]) -> None:
        self.data = rows

    def select(self, *_a, **_k) -> "_Chain":
        return self

    def eq(self, column: str, value) -> "_Chain":
        return _Chain([r for r in self.data if r.get(column) == value])

    def neq(self, column: str, value) -> "_Chain":
        return _Chain([r for r in self.data if r.get(column) != value])

    def order(self, *_a, **_k) -> "_Chain":
        return self

    def limit(self, count: int) -> "_Chain":
        return _Chain(self.data[:count])

    def range(self, start: int, end: int) -> "_Chain":
        """Inclusive at both ends, as PostgREST's is — `range(0, 999)` is a page of 1000.

        Honoured rather than stubbed out, because `select_all` pages until it sees a short
        page. A `range` that returned everything on every call would still terminate here,
        and would hide a helper that loops forever against the real database.
        """
        return _Chain(self.data[start : end + 1])

    def execute(self) -> "_Chain":
        return self


class _FakeDB:
    def __init__(self, tables: dict[str, list[dict]]) -> None:
        self._tables = tables

    def table(self, name: str) -> _Chain:
        return _Chain(self._tables.get(name, []))


SESSIONS = [
    # Two real wins: the customer paid on our link, and Razorpay named the link.
    {"id": "s_link_1", "status": "RECOVERED", "amount_recovered": 500_00,
     "root_cause": "BANK_DOWNTIME", "attribution": "SYSTEM_RECOVERED"},
    {"id": "s_link_2", "status": "RECOVERED", "amount_recovered": 1_500_00,
     "root_cause": "AUTH_TIMEOUT", "attribution": "SYSTEM_RECOVERED"},
    # Would have paid anyway. A recovery rate counts this; nothing here does.
    {"id": "s_self", "status": "RECOVERED", "amount_recovered": 9_000_00,
     "root_cause": "BANK_DOWNTIME", "attribution": "CUSTOMER_SELF_RECOVERED"},
    # Paid 20 minutes after our message. Unprovable, and therefore not a win.
    {"id": "s_ambiguous", "status": "RECOVERED", "amount_recovered": 4_000_00,
     "root_cause": "AUTH_TIMEOUT", "attribution": "AMBIGUOUS"},
    # Recovered, but the webhook that decides causation has not arrived yet.
    {"id": "s_pending", "status": "RECOVERED", "amount_recovered": 700_00,
     "root_cause": "BANK_DOWNTIME", "attribution": None},
    # The row that made this fix necessary. Its verdict claims the strongest thing the
    # system can say about a rupee, and no event anywhere justifies it — this is the shape
    # of all 217 recoveries the deleted coin-flip batch runner left in the real database.
    {"id": "s_legacy_coinflip", "status": "RECOVERED", "amount_recovered": 6_000_00,
     "root_cause": "BANK_DOWNTIME", "attribution": "SYSTEM_RECOVERED"},
    {"id": "s_failed", "status": "FAILED", "amount_recovered": 0,
     "root_cause": "PERMANENT_DECLINE", "attribution": None},
    {"id": "s_escalated", "status": "ESCALATED", "amount_recovered": 0,
     "root_cause": "MERCHANT_ERROR", "attribution": None},
    {"id": "s_open", "status": "OPEN", "amount_recovered": 0,
     "root_cause": None, "attribution": None},
]

OBSERVED = ["s_link_1", "s_link_2", "s_self", "s_ambiguous"]
"""The sessions a `payment.captured` or `payment_link.paid` callback actually resolved."""


@pytest.fixture()
def live(monkeypatch: pytest.MonkeyPatch) -> _FakeDB:
    db = _FakeDB({
        "recovery_sessions": SESSIONS,
        "payments": [{"amount": 100_000_00}],
        "audit_events": [
            {"event_type": "EXCEPTION_LOGGED", "payment_id": "pay_1",
             "recovery_session_id": "s1", "created_at": "2026-03-25T05:00:00Z",
             "event_data": {"reason": "no legal action remains", "category": "STOPPING_RULE"}},
            {"event_type": "EXCEPTION_LOGGED", "payment_id": "pay_2",
             "recovery_session_id": "s2", "created_at": "2026-03-25T04:00:00Z",
             "event_data": {"reason": "recovery window closed: window expired",
                            "category": "COMPLIANCE"}},
            # Not an exception. Present so the event_type filter has something to exclude —
            # the trail is one table and most of it is not the give-up queue.
            {"event_type": "COMPLIANCE_REMEDY", "payment_id": "pay_3",
             "recovery_session_id": "s3", "created_at": "2026-03-25T03:00:00Z",
             "event_data": {"recommendation": "DEFER_TO_MORNING"}},
        ] + [
            {"event_type": attribution.OBSERVATION_EVENT, "payment_id": f"pay_{sid}",
             "recovery_session_id": sid, "created_at": "2026-03-25T06:00:00Z",
             "event_data": {"attribution": "recorded"}}
            for sid in OBSERVED
        ],
    })
    # Patched in `app.db.database` because that is where `select_all` looks it up; the
    # per-module bindings below are the direct `get_supabase()` callers.
    monkeypatch.setattr(database, "get_supabase", lambda: db)
    monkeypatch.setattr(dashboard, "get_supabase", lambda: db)
    return db


def _keys(node) -> set[str]:
    """Every key anywhere in the response, however deeply nested."""
    if isinstance(node, dict):
        return set(node) | {k for v in node.values() for k in _keys(v)}
    if isinstance(node, list):
        return {k for item in node for k in _keys(item)}
    return set()


def test_the_dashboard_no_longer_serves_a_recovery_rate(live: _FakeDB) -> None:
    """The regression test for the defect that mattered most.

    `recovered / total * 100` was 62.5% on the fixture above. Four of those five recoveries
    are not ours: one customer came back on their own, one is inside the ambiguity window,
    one has no verdict yet. A number that reports 62.5% next to a report arguing at length
    that recovery rates are misleading makes the submission contradict itself.

    Asserted on the response's keys rather than on its text, because the endpoint explains in
    prose why the rate is absent and a substring scan cannot tell an explanation from a
    number.
    """
    keys = _keys(_run(dashboard.get_stats()))
    assert "recovery_rate" not in keys
    assert not {k for k in keys if k.endswith("_rate")}


def test_the_dashboard_credits_only_what_the_link_can_prove(live: _FakeDB) -> None:
    served = _run(dashboard.get_stats())["attributed"]
    assert served["SYSTEM_RECOVERED"]["amount_paise"] == 2_000_00
    assert served["CUSTOMER_SELF_RECOVERED"]["amount_paise"] == 9_000_00
    assert served["AMBIGUOUS"]["amount_paise"] == 4_000_00
    assert served["SYSTEM_RECOVERED"]["sessions"] == 2


def test_a_verdict_no_audit_event_backs_is_not_a_claim(live: _FakeDB) -> None:
    """The regression test for the finding that held up the demo.

    `s_legacy_coinflip` carries `attribution = SYSTEM_RECOVERED` on ₹6,000 and has no
    observation event, because the code that wrote it decided the outcome with
    `random.random()` rather than reading a callback. Trusting the column put every one of
    those rows in the numerator: the live dashboard reported 100% of recovered rupees as
    provably ours, which is both the strongest claim the system can make and, on that data,
    entirely invented.

    It has to land in `unattributed` rather than be dropped — the money did arrive, and a
    row that vanishes from every bucket makes the totals stop adding up.
    """
    served = _run(dashboard.get_stats())
    assert served["attributed"]["SYSTEM_RECOVERED"]["amount_paise"] == 2_000_00
    assert served["unattributed"]["sessions"] == 2, "the pending one and the coin flip"
    assert served["unattributed"]["amount_paise"] == 6_700_00
    counted = sum(b["sessions"] for b in served["attributed"].values())
    assert counted + served["unattributed"]["sessions"] == 6, "every recovery is in a bucket"


def test_a_recovery_with_no_verdict_yet_is_its_own_category(live: _FakeDB) -> None:
    """Not folded into `SYSTEM_RECOVERED`, which would claim money we cannot attribute, and
    not into `AMBIGUOUS`, which is a verdict rather than the absence of one."""
    served = _run(dashboard.get_stats())
    assert served["unattributed"]["sessions"] == 2
    assert "AMBIGUOUS" not in served["unattributed"]["label"]
    assert "RECOVERY_OBSERVED" in served["unattributed"]["why"]


def test_a_low_share_arrives_with_the_reason_it_is_low(live: _FakeDB) -> None:
    """0% of recovered rupees means two different things — nothing worked, or nothing was
    measured — and a progress bar cannot tell them apart. The caveat names which one, and is
    absent when there is nothing to caveat."""
    ours = _run(dashboard.get_stats())["provably_ours"]
    assert ours["share_of_recovered"] == round(2_000_00 / 21_700_00, 4)
    assert ours["unestablished_sessions"] == 2
    assert ours["unestablished_paise"] == 6_700_00
    assert "no attribution event" in ours["caveat"]
    assert "python -m app.eval" in ours["caveat"]


def test_the_established_share_excludes_the_rows_it_says_it_excludes(live: _FakeDB) -> None:
    """The rupee share divides a real numerator by a partly invented denominator.

    ₹67 L of the ₹217 L that came back here is `s_pending` and `s_legacy_coinflip` — one
    callback that has not arrived and one row a deleted `random.random()` wrote. Both belong in
    the totals, because the money did arrive. Neither belongs in a denominator used to judge
    whether the system caused anything, and on the live table those rows are 217 of 320
    recoveries, which drags a working split down to 0%.

    So this share is counted over the audit-backed cohort only: four established verdicts, two
    of them ours. Sessions rather than rupees, because every verdict except `SYSTEM_RECOVERED`
    books zero rupees on purpose — a rupee share of this cohort would divide ours by ours.
    """
    ours = _run(dashboard.get_stats())["provably_ours"]
    established = ours["established"]

    assert established["sessions"] == 4, "two links, one self-recovery, one ambiguous"
    assert established["ours_sessions"] == 2
    assert established["self_recovered_sessions"] == 1
    assert established["share_of_established_sessions"] == 0.5
    # The excluded rows are excluded from this denominator, not from the table.
    assert established["sessions"] + ours["unestablished_sessions"] == 6
    assert established["share_of_established_sessions"] > ours["share_of_recovered"]


def test_the_established_share_never_counts_an_ambiguous_recovery_as_ours(
    live: _FakeDB,
) -> None:
    """The ambiguous session is in the denominator and not in the numerator, which is the whole
    point of having a third bucket. Folding it in either direction would make the share either
    an overclaim or a silent write-off of a real recovery."""
    served = _run(dashboard.get_stats())
    established = served["provably_ours"]["established"]
    ambiguous = served["attributed"]["AMBIGUOUS"]["sessions"]

    assert ambiguous == 1
    assert established["ours_sessions"] + established["self_recovered_sessions"] + ambiguous == (
        established["sessions"]
    )
    assert established["share_of_established_sessions"] == round(2 / 4, 4)


def test_the_dashboard_points_at_where_the_counterfactual_lives(live: _FakeDB) -> None:
    """Live traffic has no `do_nothing` twin, so lift cannot be computed here. Saying so, and
    naming the endpoint that can, is the difference between a missing metric and a hidden
    one."""
    served = _run(dashboard.get_stats())
    assert served["counterfactual"]["available_at"] == "/api/v1/eval/ladder"
    assert "do_nothing" in served["counterfactual"]["note"]


def test_the_batch_metrics_no_longer_serve_a_recovery_rate(live: _FakeDB) -> None:
    served = _run(metrics.get_batch_report())
    assert "recovery_rate" not in _keys(served)
    assert served["overall"]["attributed"]["SYSTEM_RECOVERED"] == 2
    assert served["overall"]["closed_without_recovery"] == 2
    assert served["batch_size"] == 8, "the OPEN session is not a closed outcome"


def test_the_batch_metrics_apply_the_same_audit_check_as_the_dashboard(
    live: _FakeDB,
) -> None:
    """Two endpoints reading the same column had to agree, and they did not.

    `/metrics/batch` counted the legacy coin flip as a third `SYSTEM_RECOVERED` while
    `/dashboard/stats` counted two, so the same batch had two different numbers of provable
    wins depending on which panel you opened. Both now go through `attribution.verdict_of`.
    """
    batch = _run(metrics.get_batch_report())["overall"]
    stats = _run(dashboard.get_stats())
    assert batch["attributed"]["SYSTEM_RECOVERED"] == (
        stats["attributed"]["SYSTEM_RECOVERED"]["sessions"]
    )
    assert batch["unattributed"] == stats["unattributed"]["sessions"] == 2


def test_the_per_cause_breakdown_says_the_label_is_a_prediction(live: _FakeDB) -> None:
    """In production the predicted cause is the only label that exists, and the classifier is
    bounded at 65.78% on error fields alone. A row on a confusable cause is a mixture of two
    populations, and the endpoint has to say so or it reads as ground truth."""
    served = _run(metrics.get_batch_report())
    assert "by_predicted_cause" in served
    assert "65.78" in served["keyed_on"]
    assert served["by_predicted_cause"]["AUTH_TIMEOUT"]["attributed"]["AMBIGUOUS"] == 1


def test_the_exception_list_carries_the_reason_not_just_the_count(live: _FakeDB) -> None:
    """An abandoned payment and an unrecoverable one both close as FAILED. The sentence that
    justified the give-up is the only thing that tells them apart, and it is what the person
    working the queue needs."""
    served = _run(dashboard.get_exceptions())
    assert served["count"] == 2
    assert served["by_category"] == {"COMPLIANCE": 1, "STOPPING_RULE": 1}
    assert all(item["reason"] for item in served["exceptions"])


# ── The report is a file, and files can be absent or stale ────────────────────


def test_a_missing_report_explains_how_to_produce_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """503, not 500, and with the command in it.

    A fresh clone has no `report.json` until the harness runs. The failure a reviewer meets
    should tell them the one command that fixes it, and that it needs no credentials —
    otherwise the obvious guess is that the endpoint needs a database.
    """
    monkeypatch.setattr(evaluation, "REPORT_PATH", evaluation.REPORT_PATH.parent / "nope.json")
    monkeypatch.setattr(evaluation, "_cache", {})
    with pytest.raises(HTTPException) as caught:
        _run(evaluation.full_report())
    assert caught.value.status_code == 503
    assert "python -m app.eval" in caught.value.detail


def test_regenerating_the_report_is_picked_up_without_a_restart(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache is keyed on mtime rather than process lifetime.

    The demo flow regenerates the report while the API is up. A cache that never expired
    would serve the old numbers for as long as uvicorn stayed running, which is the exact
    way a dashboard ends up disagreeing with the file it is supposed to be reading.
    """
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"generated_at": "first"}), encoding="utf-8")
    monkeypatch.setattr(evaluation, "REPORT_PATH", path)
    monkeypatch.setattr(evaluation, "_cache", {})
    assert _run(evaluation.full_report())["generated_at"] == "first"

    stale = path.stat().st_mtime
    path.write_text(json.dumps({"generated_at": "second"}), encoding="utf-8")
    import os
    os.utime(path, (stale + 10, stale + 10))
    assert _run(evaluation.full_report())["generated_at"] == "second"

