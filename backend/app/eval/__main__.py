"""`python -m app.eval` — run the ladder and write the report.

Deliberately a plain module rather than an API route. The evaluation has to be
runnable by anyone who clones the repository, with no database, no keys and no
running server, or the numbers in the report cannot be checked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.eval.harness import evaluate
from app.eval.report import write
from app.policies import LADDER
from app.sim.scenarios import SCENARIOS

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "reports"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score every recovery policy and write the report.")
    parser.add_argument(
        "--seeds", type=int, default=20,
        help="how many seeds per scenario; more narrows the intervals (default: 20)",
    )
    parser.add_argument(
        "--scenario", action="append", choices=sorted(SCENARIOS),
        help="restrict to one scenario, repeatable (default: all of them)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT,
        help=f"where to write report.json and REPORT.md (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    scenarios = tuple(args.scenario) if args.scenario else tuple(SCENARIOS)
    seeds = tuple(range(1, args.seeds + 1))
    total = len(LADDER) * len(scenarios) * len(seeds)
    print(f"running {total:,} batches: {len(LADDER)} policies × {len(scenarios)} scenarios "
          f"× {len(seeds)} seeds", file=sys.stderr)

    done = 0

    def progress(label: str) -> None:
        nonlocal done
        done += len(LADDER)
        print(f"  [{done:>5,}/{total:,}] {label}", file=sys.stderr)

    run = evaluate(LADDER, scenarios=scenarios, seeds=seeds, progress=progress)
    json_path, markdown_path = write(run, args.out)

    print(file=sys.stderr)
    for policy in run.policies:
        lift = run.pooled_lift(policy)
        share = run.pooled_share_of_achievable(policy)
        print(f"  {policy:<14} {lift}  {'' if share is None else f'{share:.1%} of ceiling'}",
              file=sys.stderr)
    print(f"\nwrote {markdown_path}\n      {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
