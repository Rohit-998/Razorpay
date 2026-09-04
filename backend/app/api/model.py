"""Training and metrics for the root-cause classifier.

`/model/train` used to read its labels out of `recovery_sessions.root_cause`. That column is
seeded with the generator's true cause and then overwritten by
`worker.process_failed_payment` with `classification.root_cause` — the classifier's own
prediction. So the first training run was honest and every run after a pipeline pass fitted
the model to its own output. Accuracy under that arrangement measures self-consistency: it
goes up as the model becomes more confidently wrong, and there is no reading of the number
that tells you anything about diagnosis.

Both endpoints now go through `app.ml.train`, which labels from the simulator's latent
`true_cause`, builds features with the same `FeatureExtractor` the worker serves with, and
splits by seed so whole independently generated batches are held out. There is one training
path and one metrics file, because two of either is how the model that answers requests
stops being the model the report describes.
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException, Query
import structlog

from app.ml import train as trainer
from app.ml.classifier import classifier

logger = structlog.get_logger()
router = APIRouter()


def _fit(train_seeds: tuple[int, ...], test_seeds: tuple[int, ...]) -> dict:
    """Run the trainer to completion on its own event loop, off the request thread.

    `trainer.main` is a coroutine only because building features goes through the async
    extractor; the work itself is CPU-bound and holds the GIL for the length of the fit.
    Awaiting it inline would stall every other request on the process for about a minute.
    """
    return asyncio.run(trainer.main(train_seeds=train_seeds, test_seeds=test_seeds))


@router.post("/model/train")
async def train_model(
    train_seeds: int = Query(
        default=len(trainer.TRAIN_SEEDS), ge=2, le=60,
        description="Batches to fit on. Fewer is faster and worse; the count is recorded "
                    "in the metrics so a short run cannot be read as the full one.",
    ),
    test_seeds: int = Query(
        default=len(trainer.TEST_SEEDS), ge=2, le=40,
        description="Held-out batches. Disjoint from the training seeds by construction.",
    ),
):
    """Fit the classifier on simulated batches and return what it scored.

    Runs to completion rather than returning a job id: it takes on the order of a minute at
    the default seed counts, and a fire-and-forget version would let the caller read
    `metrics.json` from the previous run and believe it described this one.
    """
    first = trainer.TRAIN_SEEDS[0]
    held = trainer.TEST_SEEDS[0]
    try:
        metrics = await asyncio.to_thread(
            _fit,
            tuple(range(first, first + train_seeds)),
            tuple(range(held, held + test_seeds)),
        )
    except Exception as exc:  # noqa: BLE001 — surfaced to the caller, not swallowed
        logger.error("model.train_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"training failed: {exc}") from exc

    logger.info("model.trained", accuracy=metrics["accuracy"], f1=metrics["macro_f1"])
    return {"status": "success", "metrics": metrics}


@router.get("/model/metrics")
async def get_model_metrics():
    """The metrics written by the last training run, read from disk unchanged.

    Nothing is recomputed here. A number the API derives at request time is a number that can
    disagree with the artifact the model was actually saved with, and the disagreement would
    be invisible.
    """
    if not trainer.METRICS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "the classifier has not been trained",
                "consequence": (
                    "worker.process_failed_payment returns early without a loaded model, so "
                    "no payment gets classified and no strategy gets chosen"
                ),
                "fix": "cd backend && python -m app.ml.train",
            },
        )

    metrics = json.loads(trainer.METRICS_PATH.read_text(encoding="utf-8"))
    ref = metrics["reference_points"]
    return {
        "loaded": classifier.is_loaded(),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        # The four numbers together are the claim; the accuracy alone is not one. Predicting
        # the most common cause every time scores the first. Any model reading only the error
        # fields is capped at the second. The third says the fitting is sound rather than
        # leaky — it should sit at the bound, not above it. Everything the fourth has over
        # the second is what bank health, customer history and timing bought.
        "reference_points": ref,
        "reads_above_the_bound": round(metrics["accuracy"] - ref["bayes_optimal_error_fields_only"], 4),
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_order": metrics["class_order"],
        "under_distribution_shift": metrics["under_distribution_shift"],
        "data": metrics["data"],
        "hardest_class": min(
            metrics["per_class"], key=lambda c: metrics["per_class"][c]["f1"]
        ),
    }
