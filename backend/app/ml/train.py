"""Train the root-cause classifier and score it against the bound it has to beat.

Run with `python -m app.ml.train`. Writes the model, the two encoders, and `metrics.json`
into `app/ml/model_artifacts/`, and prints a summary.

An accuracy figure on its own is not a claim about anything, so this reports four numbers
next to each other:

  * the majority-class rate — what predicting the most common cause every time would score;
  * the Bayes-optimal accuracy from the error fields alone, computed analytically from the
    emission distributions in `app.sim.emission`. This is the ceiling for any model that
    looks only at `error_source`, `error_step` and `error_reason`;
  * an XGBoost trained on those three columns only. It should land at the analytic bound.
    Materially above it means something is leaking — a value in one of those fields that
    identifies its cause, which is the failure mode the whole emission model exists to
    avoid;
  * the full model. Everything it scores above the bound comes from bank health, customer
    history and timing, which is the entire argument for extracting those features.

The split is by seed, not by row. Payments inside one batch share a generated world: the
same outages, the same bank-hours, the same customers. Cutting a batch in half and calling
one part held-out scores the model on the circumstances it was fitted on, and inflates
everything above. Whole batches go on one side or the other.

Training seeds are 101–140 and test seeds 201–215, deliberately disjoint from the 1–20 the
eval harness runs on. Nothing in the eval uses this model — `payrevive` decides from
`app.policies.calibration` — but keeping the seed ranges apart means that stays true by
construction rather than by inspection.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from app.ml.classifier import ALL_FEATURE_NAMES, MODEL_DIR, classifier
from app.ml.dataset import ERROR_FIELD_COLUMNS, build_rows, fit_encoder, to_matrix
from app.sim.emission import label_ambiguity
from app.sim.scenarios import SCENARIOS

TRAIN_SCENARIO = "baseline"
TRAIN_SEEDS = tuple(range(101, 141))
TEST_SEEDS = tuple(range(201, 216))
SHIFT_SCENARIOS = ("outage_day", "salary_week", "festival_spike", "stress_dead_instruments")

METRICS_PATH = MODEL_DIR / "metrics.json"


def _error_field_indices() -> list[int]:
    """Column positions of the three error fields inside the full feature matrix."""
    return [ALL_FEATURE_NAMES.index(name) for name in ERROR_FIELD_COLUMNS]


def _ablation_accuracy(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Accuracy of the same model class given only the error fields.

    The comparison the Bayes bound is for. Trained on the identical rows and split so the
    only difference from the full model is which columns it can see.
    """
    cols = _error_field_indices()
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", eval_metric="mlogloss", random_state=42,
    )
    classes = sorted(set(y_train))
    index = {c: i for i, c in enumerate(classes)}
    model.fit(X_train[:, cols], np.array([index[c] for c in y_train]), verbose=False)
    predicted = model.predict(X_test[:, cols])
    return float(accuracy_score([index[c] for c in y_test], predicted))


async def main(
    train_seeds: tuple[int, ...] = TRAIN_SEEDS,
    test_seeds: tuple[int, ...] = TEST_SEEDS,
) -> dict:
    """Build the dataset, fit, score, write the artifacts, and return the metrics.

    The seed ranges are arguments so `/model/train` can run a shorter fit for a demo without
    a second training path existing. Whatever it runs is recorded in `metrics["data"]`, so a
    reduced run cannot be mistaken for the full one.
    """
    encoder = fit_encoder()

    train_rows = await build_rows([TRAIN_SCENARIO], list(train_seeds))
    test_rows = await build_rows([TRAIN_SCENARIO], list(test_seeds))
    X_train, y_train = to_matrix(train_rows, encoder)
    X_test, y_test = to_matrix(test_rows, encoder)

    # Saved before training because `classifier.train` dumps whatever is on the singleton,
    # and a model served with a different encoder than it was fitted with is silently wrong.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    classifier.ordinal_encoder = encoder

    metrics = classifier.train(X_train, y_train, X_test, y_test)

    bound = label_ambiguity()
    scenario_bound = label_ambiguity(SCENARIOS[TRAIN_SCENARIO].normalised_weights())
    ablation = _ablation_accuracy(X_train, y_train, X_test, y_test)
    values, counts = np.unique(y_test, return_counts=True)
    metrics["reference_points"] = {
        "majority_class": str(values[int(counts.argmax())]),
        "majority_class_rate": round(float(counts.max() / counts.sum()), 4),
        # Two bounds, because they answer different questions. The uniform figure is what
        # the eval report quotes: the score available from the error fields to something
        # that does not know the cause mix. The scenario figure adds that knowledge, and is
        # the one a model trained on this scenario has to be held against — the model
        # learned the mix from the training set, so clearing the uniform bound proves
        # nothing about leakage. On the first run this mattered: the error-fields-only
        # ablation scored 69.2% against a uniform bound of 65.8%, which reads as leakage
        # until the prior is accounted for.
        "bayes_optimal_error_fields_only": scenario_bound[
            "bayes_optimal_accuracy_error_fields_only"
        ],
        "bayes_optimal_error_fields_only_uniform_prior": bound[
            "bayes_optimal_accuracy_error_fields_only"
        ],
        "distinct_error_signatures": bound["distinct_signatures"],
        "error_fields_only_model": round(ablation, 4),
        # So the comparison against the bound is checkable rather than asserted. The
        # ablation is a proportion measured on a finite test set, and a figure a little
        # above an exact bound is what sampling looks like — the residual has to be read
        # against this, not against zero. A real leak moves the ablation percentage points
        # clear of the bound, many standard errors out.
        "error_fields_only_standard_error": round(
            float(np.sqrt(ablation * (1.0 - ablation) / len(y_test))), 4
        ),
    }

    # How it holds up on batches whose cause mix it was never trained on. A model fitted on
    # a normal week and deployed through an outage day is the ordinary case, not the
    # exception, and a figure quoted only in-distribution does not cover it.
    shift: dict[str, dict] = {}
    for name in SHIFT_SCENARIOS:
        rows = await build_rows([name], list(test_seeds))
        X_shift, y_shift = to_matrix(rows, encoder)
        predicted = classifier.label_encoder.inverse_transform(
            classifier.model.predict(X_shift)
        )
        shift[name] = {
            "accuracy": round(float(accuracy_score(y_shift, predicted)), 4),
            "payments": int(len(rows)),
        }
    metrics["under_distribution_shift"] = shift

    metrics["data"] = {
        "train_scenario": TRAIN_SCENARIO,
        "train_seeds": list(train_seeds),
        "test_seeds": list(test_seeds),
        "split": "by seed, so whole independently generated batches are held out",
        "labels": "Episode.true_cause from the simulator, never a stored prediction",
        "features_built_by": "app.cache.feature_store.FeatureExtractor, the serving path",
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _print(metrics: dict) -> None:
    ref = metrics["reference_points"]
    shift = metrics["under_distribution_shift"]
    print(f"trained on {metrics['train_size']:,} payments, tested on {metrics['test_size']:,}")
    print(f"  majority class ({ref['majority_class']}): {ref['majority_class_rate']:.1%}")
    print(f"  error fields only, Bayes optimal:  "
          f"{ref['bayes_optimal_error_fields_only']:.2%} under this scenario's cause mix, "
          f"{ref['bayes_optimal_error_fields_only_uniform_prior']:.2%} under a uniform prior"
          f"  ({ref['distinct_error_signatures']} signatures)")
    print(f"  error fields only, fitted:         {ref['error_fields_only_model']:.2%}"
          f"  (± {ref['error_fields_only_standard_error']:.2%})")
    print(f"  all {len(ALL_FEATURE_NAMES)} features:                 {metrics['accuracy']:.2%}"
          f"   macro F1 {metrics['macro_f1']:.3f}")
    print("  under distribution shift:")
    for name, row in shift.items():
        print(f"    {name:<24} {row['accuracy']:.2%}  ({row['payments']:,} payments)")
    print(f"artifacts written to {MODEL_DIR}")


if __name__ == "__main__":
    _print(asyncio.run(main()))

