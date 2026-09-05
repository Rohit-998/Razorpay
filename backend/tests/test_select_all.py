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
    """

    def __init__(self, rows: list[dict], calls: list[tuple[int, int]], cap: int) -> None:
        self.data = rows
        self.calls = calls
        self.cap = cap

    def select(self, *_a, **_k) -> "_Chain":
        return self

    def eq(self, column: str, value) -> "_Chain":
        return _Chain(
            [r for r in self.data if r.get(column) == value], self.calls, self.cap
        )

    def range(self, start: int, end: int) -> "_Chain":
        self.calls.append((start, end))
        window = self.data[start : end + 1]
        # The cap applies to whatever the window asked for, which is what makes an
        # unbounded read silently partial rather than loud.
        return _Chain(window[: self.cap], self.calls, self.cap)

    def execute(self) -> "_Chain":
        return self


class _FakeDB:
    def __init__(self, rows: list[dict], cap: int) -> None:
        self.rows = rows
        self.calls: list[tuple[int, int]] = []
        self.cap = cap

    def table(self, _name: str) -> _Chain:
        return _Chain(self.rows, self.calls, self.cap)


@pytest.fixture()
def fake(monkeypatch: pytest.MonkeyPatch):
    def build(rows: list[dict], cap: int = 1000) -> _FakeDB:
        db = _FakeDB(rows, cap)
        monkeypatch.setattr(database, "get_supabase", lambda: db)
        return db

    return build


def _rows(count: int, **extra) -> list[dict]:
    return [{"id": f"r{i}", "amount": 100, **extra} for i in range(count)]


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
