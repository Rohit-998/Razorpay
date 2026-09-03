"""Measurement, kept separate from the thing being measured.

`python -m app.eval` runs the whole ladder over every scenario at every seed and
writes both artefacts. Nothing in this package touches the database, the network or
the LLM: an evaluation that depends on a hosted service is not reproducible, and a
report nobody can regenerate is a screenshot.
"""

from app.eval.harness import EvalRun, Interval, PolicyOnScenario, evaluate, paired_interval
from app.eval.metrics import BatchMetrics, CauseBreakdown, Comparison, collect, compare
from app.eval.report import markdown, to_json, write

__all__ = [
    "BatchMetrics",
    "CauseBreakdown",
    "Comparison",
    "EvalRun",
    "Interval",
    "PolicyOnScenario",
    "collect",
    "compare",
    "evaluate",
    "markdown",
    "paired_interval",
    "to_json",
    "write",
]
