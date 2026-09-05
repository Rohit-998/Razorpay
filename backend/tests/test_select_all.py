"""A total computed over one page is a wrong number wearing the right label.

PostgREST caps an unbounded `select()` at `db-max-rows` — 1000 on Supabase's defaults — and
returns the truncated set with no error and no flag. For a list endpoint that is a missing
page a reviewer would notice. For an aggregate it is worse than an error: `at_risk_paise`
keeps its name, keeps its units, and silently stops being a total.

This was live. `audit_events` in this project's database holds 1158 rows, and a census of
event types run against it summed to exactly 1000 — the cap, mistaken for the population.
The endpoint reading that table to decide which recoveries are provable would have counted
the sessions in the first page and reported the rest as unattributed, which is a *safe*
direction to be wrong in and still wrong.

So the tests below are about the boundary, because the boundary is where the bug lives: the
population being an exact multiple of the page size, and the filter being applied server side
rather than after truncation.
"""

from __future__ import annotations

import pytest

from app.db import database


class _Chain:
    """A PostgREST double that enforces the cap instead of pretending it does not exist.

    `range()` is inclusive at both ends, as the real one is, and every call is recorded so a
    test can assert how many round trips a read cost.

    It also models the half of `LIMIT/OFFSET` that is easy to forget: an unordered scan makes
    no promise about sequence. Ask for rows 0-999 and then 1000-1999 without an `ORDER BY` and
    Postgres is free to hand back a different arrangement each time, so a row can arrive twice
    or not at all. The double reproduces that by rotating its rows once per unordered read —
    deterministically, so a failure is reproducible rather than flaky. `order()` pins the
    sequence and the rotation stops.
    """

    def __init__(
        self,
        rows: list[dict],
        calls: list[tuple[int, int]],
        cap: int,
        ordered_by: list[str],
        ordered: bool = False,
    ) -> None:
        self.data = rows
        self.calls = calls
        self.cap = cap
        self.ordered_by = ordered_by
        self.ordered = ordered

    def _child(self, rows: list[dict], ordered: bool | None = None) -> "_Chain":
        return _Chain(
            rows,
            self.calls,
            self.cap,
            self.ordered_by,
            self.ordered if ordered is None else ordered,
        )

    def select(self, *_a, **_k) -> "_Chain":
        return self

    def eq(self, column: str, value) -> "_Chain":
        return self._child([r for r in self.data if r.get(column) == value])

    def order(self, column: str, **_k) -> "_Chain":
        self.ordered_by.append(column)
        # NULLs last, as Postgres orders them by default, so a fixture that does not carry the
        # column sorts rather than raising — the parameter is about sequence, not about
        # requiring every caller to select every column it sorts on.
        return self._child(
            sorted(self.data, key=lambda r: (r.get(column) is None, r.get(column))),
            ordered=True,
        )

    def range(self, start: int, end: int) -> "_Chain":
        self.calls.append((start, end))
        rows = self.data
        if not self.ordered:
            # A different arrangement on every scan, which is what the real thing is allowed
            # to do and what makes an unordered paged read return the wrong population.
            shift = len(self.calls) % max(len(rows), 1)
            rows = rows[shift:] + rows[:shift]
        window = rows[start : end + 1]
        # The cap applies to whatever the window asked for, which is what makes an
        # unbounded read silently partial rather than loud.
        return self._child(window[: self.cap])

    def execute(self) -> "_Chain":
        return self


class _FakeDB:
    def __init__(self, rows: list[dict], cap: int) -> None:
        self.rows = rows
        self.calls: list[tuple[int, int]] = []
        self.ordered_by: list[str] = []
        self.cap = cap

    def table(self, _name: str) -> _Chain:
        return _Chain(self.rows, self.calls, self.cap, self.ordered_by)


@pytest.fixture()
def fake(monkeypatch: pytest.MonkeyPatch):
    def build(rows: list[dict], cap: int = 1000) -> _FakeDB:
        db = _FakeDB(rows, cap)
        monkeypatch.setattr(database, "get_supabase", lambda: db)
        return db

    return build


def _rows(count: int, **extra) -> list[dict]:
    # `created_at` descends as the index rises, so a test that asks for the oldest first gets a
    # sequence it can tell apart from insertion order.
    return [
        {"id": f"r{i}", "amount": 100, "created_at": f"2026-09-{(count - i) % 28 + 1:02d}", **extra}
        for i in range(count)
    ]


def test_a_population_larger_than_the_cap_comes_back_whole(fake) -> None:
    """1158 rows, page size 1000. The number this file exists for."""
    db = fake(_rows(1158))
    got = database.select_all("audit_events", "id")
    assert len(got) == 1158
    assert sum(r["amount"] for r in got) == 115_800
    assert db.calls == [(0, 999), (1000, 1999)]


def test_a_population_that_is_an_exact_multiple_of_the_page_size_is_not_cut_short(
    fake,
) -> None:
    """The off-by-one that a `len(page) == 0` stop condition gets right and a
    `len(rows) < expected` guess does not.

    2000 rows arrive as two full pages, and a helper that stopped on the second would be
    correct here by luck. It has to ask a third time and get nothing to know it is done.
    """
    db = fake(_rows(2000))
    assert len(database.select_all("audit_events", "id")) == 2000
    assert db.calls == [(0, 999), (1000, 1999), (2000, 2999)]


def test_a_short_first_page_costs_exactly_one_round_trip(fake) -> None:
    """The common case. Paging must not turn a 726-row read into two requests."""
    db = fake(_rows(726))
    assert len(database.select_all("recovery_sessions", "id")) == 726
    assert db.calls == [(0, 999)]


def test_an_empty_table_is_not_an_error(fake) -> None:
    db = fake([])
    assert database.select_all("payments", "amount") == []
    assert db.calls == [(0, 999)]


def test_the_filter_runs_before_the_cap_not_after(fake) -> None:
    """Filtering locally would spend the page on rows that get thrown away.

    Here 2400 rows of the wrong event type sit in front of 30 of the right one. A read that
    fetched a page and then filtered would return zero and report, quite confidently, that no
    attribution event exists — every provable recovery reclassified as unattributed.
    """
    rows = _rows(2400, event_type="PIPELINE_COMPLETE") + _rows(30, event_type="RECOVERY_OBSERVED")
    db = fake(rows)
    got = database.select_all(
        "audit_events", "recovery_session_id", event_type="RECOVERY_OBSERVED"
    )
    assert len(got) == 30
    assert db.calls == [(0, 999)], "the filter narrowed it to a single short page"


def test_paging_is_not_hardcoded_to_the_supabase_default(fake) -> None:
    """`page_size` is a parameter because `db-max-rows` is configuration, not a constant.

    A deployment that lowered it to 500 would silently halve every aggregate in the product
    if the helper assumed 1000.
    """
    db = fake(_rows(120), cap=50)
    assert len(database.select_all("audit_events", "id", page_size=50)) == 120
    assert db.calls == [(0, 49), (50, 99), (100, 149)]


def test_every_page_is_asked_for_in_a_fixed_order(fake) -> None:
    """`range()` is `LIMIT/OFFSET`, and offsets into an unordered result mean nothing.

    Two pages, two scans, and nothing obliging the second scan to arrange rows the way the
    first one did — so a row lands on both pages or on neither, and the census that comes out
    is wrong in a way no assertion about *length* would catch if the pages happened to add up.
    The fix is one clause, so the test is here to keep it: every read names a sort column, and
    it is the primary key, which every table in `migrations/init.sql` has.
    """
    db = fake(_rows(1158))
    got = database.select_all("audit_events", "id")

    assert db.ordered_by == ["id", "id"], "one ORDER BY per page, not one per read"
    assert len({r["id"] for r in got}) == 1158, "no row fetched twice, none skipped"


def test_a_caller_can_say_which_rows_come_first(fake) -> None:
    """Ordering by the key is enough to make paging correct and is not always enough to make
    it *useful*: `/batch/run` works a slice of the open sessions and wants the oldest, not
    fifteen arbitrary UUIDs. The parameter is the difference between a defensible queue
    discipline and a coin flip over which payments get worked."""
    db = fake(_rows(20))
    got = database.select_all("recovery_sessions", "payment_id", order_by="created_at")

    assert db.ordered_by == ["created_at"]
    # `_rows` dates descend as the index rises, so the oldest row is the last one inserted —
    # a sequence that insertion order alone could not have produced.
    assert [r["id"] for r in got[:3]] == ["r19", "r18", "r17"]
