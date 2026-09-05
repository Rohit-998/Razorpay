"""`/batch/run` works a slice, and has to say that it did.

The endpoint's predecessor returned in milliseconds because it never did the work: it skipped
compliance and the executor, drew `random.random()` against a table of recovery rates, and
wrote the coin flip to the database as a proven recovery. Doing the work instead costs a
classifier call, a compliance read, a bandit read, an action and several audit inserts per
payment — each a round trip to a hosted database — and a measured run over 137 open sessions
took 435 seconds.

So the endpoint takes a `limit`, and that is where a new dishonesty could enter. `processed: 15`
on a queue of 97 reads exactly like a finished batch, and a reviewer watching a demo has no way
to tell a bounded run from a complete one. The tests below are about the disclosure as much as
the arithmetic: what was open when the run started, and what it chose not to touch.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.api import batch


def _run(coro):
    return asyncio.run(coro)


class _Worker:
    """A stand-in for `worker.process_failed_payment`, recording who it was called for.

    Order matters here, not just count: a bounded run that works fifteen *arbitrary* sessions
    is a coin flip over which merchants get served, and the queue discipline is the reason the
    read asks for `created_at`.
    """

    def __init__(self, fails: set[str] | None = None) -> None:
        self.seen: list[str] = []
        self.fails = fails or set()

    async def __call__(self, _ctx: dict, payment_id: str) -> None:
        self.seen.append(payment_id)
        if payment_id in self.fails:
            raise RuntimeError("classifier unavailable")


@pytest.fixture()
def bench(monkeypatch: pytest.MonkeyPatch):
    """Wire up `select_all` and the worker, and hand back what the endpoint asked the DB for."""

    def build(open_ids: list[str], statuses: dict[str, int], fails: set[str] | None = None):
        worker = _Worker(fails)
        reads: list[dict[str, Any]] = []

        def fake_select_all(table: str, columns: str, **kwargs: Any) -> list[dict]:
            reads.append({"table": table, "columns": columns, **kwargs})
            if kwargs.get("status") == "OPEN":
                return [{"payment_id": pid} for pid in open_ids]
            # The post-run census, which the endpoint counts locally.
            return [{"status": s} for s, n in statuses.items() for _ in range(n)]

        monkeypatch.setattr(batch, "select_all", fake_select_all)
        # `run_batch_pipeline` imports this inside the function body, so the patch has to land
        # on the module it imports *from*, not on a name in `batch`.
        import app.worker

        monkeypatch.setattr(app.worker, "process_failed_payment", worker)
        return worker, reads

    return build


def test_an_unbounded_run_works_every_open_session(bench) -> None:
    """`limit=0` is the default and means the whole queue, so the endpoint keeps behaving the
    way the worker and the webhook path do when nobody asks for a slice."""
    worker, _ = bench([f"pay_{i}" for i in range(40)], {"OPEN": 12, "RECOVERED": 3})
    served = _run(batch.run_batch_pipeline())

    assert served["results"] == {"total": 40, "processed": 40, "errored": 0}
    assert served["open_at_start"] == 40
    assert served["not_worked_this_run"] == 0
    assert len(worker.seen) == 40


def test_a_bounded_run_says_how_much_it_left(bench) -> None:
    """The number that stops `processed: 15` from reading as a finished batch."""
    worker, _ = bench([f"pay_{i}" for i in range(97)], {"OPEN": 82})
    served = _run(batch.run_batch_pipeline(limit=15))

    assert served["results"] == {"total": 15, "processed": 15, "errored": 0}
    assert served["open_at_start"] == 97, "the queue as it was, not the slice"
    assert served["not_worked_this_run"] == 82
    assert worker.seen == [f"pay_{i}" for i in range(15)]


def test_the_slice_is_the_oldest_sessions_not_an_arbitrary_page(bench) -> None:
    """Which fifteen is a policy question, and the answer has to be in the query.

    Ordering by the primary key would page correctly and still pick fifteen arbitrary UUIDs —
    a payment that has been waiting an hour would sit behind one that arrived a second ago,
    forever, because nothing about a UUID moves it up the queue.
    """
    _, reads = bench(["pay_a", "pay_b"], {"OPEN": 0})
    _run(batch.run_batch_pipeline(limit=1))

    queue_read = next(r for r in reads if r.get("status") == "OPEN")
    assert queue_read["order_by"] == "created_at"
    assert queue_read["table"] == "recovery_sessions"


def test_one_payment_blowing_up_does_not_end_the_run(bench) -> None:
    """A batch is not a transaction. One classifier timeout must not strand the other
    fourteen, and it must not be counted as processed either."""
    worker, _ = bench(
        [f"pay_{i}" for i in range(10)], {"OPEN": 0}, fails={"pay_3", "pay_7"}
    )
    served = _run(batch.run_batch_pipeline())

    assert served["results"] == {"total": 10, "processed": 8, "errored": 2}
    assert len(worker.seen) == 10, "the run continued past both failures"


def test_a_limit_over_the_queue_length_is_the_queue(bench) -> None:
    """Asking for fifteen of four is four, with nothing left over and no claim that there is."""
    worker, _ = bench(["pay_a", "pay_b", "pay_c", "pay_d"], {"RECOVERED": 4})
    served = _run(batch.run_batch_pipeline(limit=15))

    assert served["results"]["total"] == 4
    assert served["not_worked_this_run"] == 0
    assert len(worker.seen) == 4


def test_an_empty_queue_is_not_a_completed_batch(bench) -> None:
    """`status: complete` on an empty queue would put a finished-looking response on the
    screen for a run that did nothing. It answers `no_data` and says so."""
    worker, _ = bench([], {"RECOVERED": 320})
    served = _run(batch.run_batch_pipeline(limit=15))

    assert served["status"] == "no_data"
    assert worker.seen == []


def test_the_census_is_read_through_the_paged_helper(bench) -> None:
    """`sessions_by_status` is the number a reviewer reads as the state of the whole batch, and
    the last unbounded version of it summed to exactly 1000 — PostgREST's row cap wearing the
    label of a population."""
    _, reads = bench(["pay_a"], {"OPEN": 1200, "RECOVERED": 900})
    served = _run(batch.run_batch_pipeline())

    assert served["sessions_by_status"] == {"OPEN": 1200, "RECOVERED": 900}
    assert sum(served["sessions_by_status"].values()) == 2100, "not capped at 1000"
    assert any(r["columns"] == "status" for r in reads)
