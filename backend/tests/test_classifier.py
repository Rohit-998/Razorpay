"""The classifier, and the three ways its accuracy figure used to be meaningless.

The labels were its own output. `/model/train` read them from
`recovery_sessions.root_cause`, which `worker.process_failed_payment` overwrites with
`classification.root_cause` on every pass — so after one pipeline run the model was being
fitted to what it had already predicted.

The encoder did not know the vocabulary. Its category lists were missing `payment_failed`,
`limit_exceeded` and `payment_authorization`, which are respectively the most common error
reason and the most common error step in the data. `OrdinalEncoder` maps anything unlisted to
`-1` without complaint, so the three highest-frequency values in the dataset shared one code
that meant "unrecognised".

The split was inside a batch. Rows from one generated batch share a world — the same outage
windows, the same bank-hours, the same customers — so cutting the last 20% off scored the
model on circumstances it had been fitted on.

None of the three raises anything. Each produces a number that looks like accuracy.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from app.ml import dataset as ds
from app.ml import train as trainer
from app.ml.classifier import (
    ALL_FEATURE_NAMES,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from app.models.schemas import PaymentMethod, RootCause
from app.sim import emission as em
from app.sim.scenarios import SCENARIOS


def _run(coro):
    return asyncio.run(coro)


METRICS = trainer.METRICS_PATH


def _code_without_prose(module) -> str:
    """A module's source with every docstring and comment removed.

    For the assertions that a name does *not* appear. `app.ml.dataset` explains at length why
    it must not read `world.downtime_at()`, and that explanation is the reason the call is
    unlikely to come back — a scan that cannot tell the warning from the mistake would force
    the warning to be deleted. `ast.unparse` drops comments too, which is the same argument.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node) is not None:
                node.body = node.body[1:]
    return ast.unparse(tree)


# ── The encoder knows every value the gateway can send ─────────────────────────


@pytest.mark.parametrize(
    "column,vocabulary",
    [
        ("error_source", em.ERROR_SOURCES),
        ("error_step", em.ERROR_STEPS),
        ("error_reason", em.ERROR_REASONS),
    ],
)
def test_every_error_field_value_has_a_code(column: str, vocabulary) -> None:
    """The defect this replaces was silent, and so a helpful trim would be.

    `app.sim.emission` is where the observable vocabulary is defined. A value it can emit
    that the encoder has no category for does not error — it encodes as `-1`, the same code
    every other unrecognised value gets, and the model classifies without it.
    """
    missing = set(vocabulary) - set(CATEGORICAL_FEATURES[column])
    assert not missing, f"{column} has no code for {sorted(missing)}"


def test_the_method_and_bucket_vocabularies_match_their_sources() -> None:
    """Same rule for the two categoricals that come from elsewhere in the app."""
    assert set(CATEGORICAL_FEATURES["payment_method"]) == {m.value for m in PaymentMethod}
    assert set(CATEGORICAL_FEATURES["amount_bucket"]) == {
        "micro", "small", "medium", "large", "premium"
    }


def test_no_feature_is_counted_twice_or_missing_from_the_order() -> None:
    """`ALL_FEATURE_NAMES` is the column order the serving path and SHAP both index by.

    A duplicate or an omission here does not raise: predictions keep coming, from the wrong
    columns, with SHAP labelling them confidently.
    """
    assert len(ALL_FEATURE_NAMES) == len(set(ALL_FEATURE_NAMES))
    assert set(ALL_FEATURE_NAMES) == (
        set(CATEGORICAL_FEATURES) | set(NUMERICAL_FEATURES) | set(BOOLEAN_FEATURES)
    )
    assert len(ALL_FEATURE_NAMES) == 17


# ── The label is never a feature, and never a stored prediction ────────────────


@pytest.fixture(scope="module")
def rows() -> list[ds.Row]:
    """One batch, built through the production extractor. Deterministic in (scenario, seed)."""
    return _run(ds.build_rows(["baseline"], [7]))


def test_the_row_carries_only_what_a_webhook_would_have_delivered(rows) -> None:
    """`true_cause` is the label. It reaches the dataset as `Row.cause` and never as a field
    of the feature vector, which is the whole reason the features go through `FailedPayment`
    rather than being read off the episode."""
    keys = set(rows[0].features.to_dict())
    assert keys == set(ALL_FEATURE_NAMES)
    assert not {k for k in keys if "cause" in k}
    assert {r.cause for r in rows} <= {c.value for c in RootCause}


def test_features_are_built_by_the_serving_extractor(rows) -> None:
    """Not by a copy of it. The row the model trains on and the row the worker classifies
    come off `FeatureExtractor.extract`; a second implementation would diverge silently and
    the metrics would keep looking fine."""
    import inspect

    source = inspect.getsource(ds.build_rows)
    assert "extractor.extract" in source
    assert "FeatureExtractor" in inspect.getsource(ds)


def test_the_downtime_flag_is_the_observable_proxy_not_the_latent_state(rows) -> None:
    """The one feature where reading the simulator's own variable would be leakage.

    Downtime-caused failures are generated *inside* real outage windows, so
    `world.downtime_at()` is very nearly the BANK_DOWNTIME label. What a merchant actually
    has is its own failure count against that bank's baseline, which is noisy — it fires on
    payments that are not downtime and misses ones that are. If the flag were latent it
    would be true for essentially every downtime episode and almost nothing else.

    Both halves are asserted, because either alone is weak. The source scan catches the
    call coming back; the recall figure catches a different route to the same latent state.
    """
    assert "downtime_at" not in _code_without_prose(ds)

    flagged = [r for r in rows if r.features.bank_is_in_downtime]
    assert flagged, "the proxy fires on some payments, or it carries no signal at all"
    downtime = {r.cause for r in flagged}
    assert len(downtime) > 1, "a flag true only for BANK_DOWNTIME would be the label itself"
    recall = sum(
        r.features.bank_is_in_downtime for r in rows if r.cause == "BANK_DOWNTIME"
    ) / max(1, sum(r.cause == "BANK_DOWNTIME" for r in rows))
    assert recall < 0.9, f"proxy caught {recall:.0%} of outages, which is not a proxy"


def test_the_dataset_is_reproducible(rows) -> None:
    """Anyone who clones the repo gets these exact rows, or the reported accuracy is not
    something they can check."""
    again = _run(ds.build_rows(["baseline"], [7]))
    X1, y1 = ds.to_matrix(rows, ds.fit_encoder())
    X2, y2 = ds.to_matrix(again, ds.fit_encoder())
    assert np.array_equal(X1, X2)
    assert list(y1) == list(y2)


def test_the_encoder_maps_every_value_the_dataset_contains(rows) -> None:
    """The end-to-end version of the vocabulary tests: no `-1` in the categorical block."""
    X, _ = ds.to_matrix(rows, ds.fit_encoder())
    categorical = X[:, : len(CATEGORICAL_FEATURES)]
    assert (categorical >= 0).all(), "some value encoded as unrecognised"


def test_the_encoder_is_fitted_on_the_vocabulary_not_on_the_sample() -> None:
    """Otherwise the code space shifts whenever the dataset changes, and a category absent
    from one training run is `-1` at serve time despite the model having a column for it."""
    from_small = ds.fit_encoder()
    codes = from_small.transform([["gateway", "payment_capture", "limit_exceeded",
                                  "wallet", "premium"]])
    assert (codes >= 0).all()


# ── The split holds out batches, not rows ──────────────────────────────────────


def test_training_and_test_seeds_do_not_overlap() -> None:
    assert not set(trainer.TRAIN_SEEDS) & set(trainer.TEST_SEEDS)


def test_the_eval_harness_seeds_are_not_trained_on() -> None:
    """The eval runs seeds 1-20. Nothing in it uses this model — `payrevive` decides from
    `app.policies.calibration` — and keeping the ranges disjoint means that stays true by
    construction rather than by someone checking."""
    assert not set(trainer.TRAIN_SEEDS) & set(range(1, 21))
    assert not set(trainer.TEST_SEEDS) & set(range(1, 21))


def test_a_batch_is_never_split_across_the_boundary() -> None:
    """The defect: `classifier.train` used to cut the last 20% of rows off the matrix. Within
    one batch those rows share the outage windows and the customers of the rows above them,
    so the held-out set was not held out from anything that mattered."""
    import inspect

    from app.ml import classifier as classifier_module
    from app.ml.classifier import RootCauseClassifier

    assert "split_idx" not in _code_without_prose(classifier_module)
    assert "X_test" in inspect.signature(RootCauseClassifier.train).parameters


# ── The accuracy figure is a claim, and the claim is checkable ─────────────────


@pytest.fixture(scope="module")
def metrics() -> dict:
    """What the last training run wrote. Committed, so these run on a fresh clone.

    Every assertion below is a relationship between numbers in this file rather than a
    number typed into the test. A hardcoded 82.8% would pass whether or not the figure
    still came from anywhere.
    """
    if not METRICS.exists():
        pytest.skip(f"no metrics.json — run `python -m app.ml.train` ({METRICS})")
    return json.loads(METRICS.read_text(encoding="utf-8"))


def test_the_metrics_say_which_seeds_produced_them(metrics: dict) -> None:
    """A figure with no record of the data behind it cannot be reproduced or disputed."""
    data = metrics["data"]
    assert data["train_seeds"] and data["test_seeds"]
    assert not set(data["train_seeds"]) & set(data["test_seeds"])
    assert data["train_scenario"] in SCENARIOS
    assert "true_cause" in data["labels"]
    assert "FeatureExtractor" in data["features_built_by"]


def test_the_error_fields_only_model_lands_at_the_bayes_bound(metrics: dict) -> None:
    """The leakage detector, and the reason the bound is computed analytically.

    An XGBoost given only `error_source`, `error_step` and `error_reason` cannot beat the
    Bayes-optimal accuracy of those three columns. If it does, by more than sampling
    explains, then some value in one of them identifies its cause — the failure mode
    `app.sim.emission` exists to prevent, and the one that would make the headline accuracy
    a lookup-table inversion.

    Judged against the scenario's own cause mix, not the uniform prior: the model learned
    the mix from the training set, so clearing the uniform figure proves nothing. And judged
    in standard errors, because the ablation is a proportion on a finite test set and a
    figure slightly above an exact bound is what sampling looks like. On the run in the
    committed artifact it sits 1.4 SE above.
    """
    ref = metrics["reference_points"]
    residual = ref["error_fields_only_model"] - ref["bayes_optimal_error_fields_only"]
    se = ref["error_fields_only_standard_error"]
    assert se > 0
    assert residual < 3 * se, (
        f"error-fields-only model is {residual / se:.1f} standard errors above its own "
        f"ceiling ({ref['error_fields_only_model']:.2%} vs "
        f"{ref['bayes_optimal_error_fields_only']:.2%}) — something in those three columns "
        f"identifies the cause"
    )
    assert ref["error_fields_only_model"] > ref["bayes_optimal_error_fields_only"] - 0.05, (
        "the ablation is far below its ceiling, so the comparison is measuring a bad fit "
        "rather than the information in the fields"
    )


def test_what_the_full_model_adds_is_what_the_other_features_bought(metrics: dict) -> None:
    """The whole argument for extracting bank health, customer history and timing.

    Everything above the error-fields bound comes from those three families and nothing
    else — there is no other input. If the margin collapsed, the feature store would be
    doing no work and the 17 columns would be 3 columns with overhead.
    """
    ref = metrics["reference_points"]
    margin = metrics["accuracy"] - ref["bayes_optimal_error_fields_only"]
    assert margin > 0.05, f"only {margin:.2%} above what the error fields alone can give"
    assert metrics["accuracy"] > ref["error_fields_only_model"]
    assert metrics["accuracy"] > ref["majority_class_rate"] * 2


def test_no_cause_is_predicted_perfectly(metrics: dict) -> None:
    """The other shape leakage takes, and the one an aggregate figure hides.

    Overall accuracy in the eighties is consistent with six honest classes and one that is
    being read off a giveaway value. Every cause here shares its observable vocabulary with
    at least one other, so a per-class F1 at the ceiling means a feature is carrying the
    label.
    """
    perfect = {
        cause: row["f1"] for cause, row in metrics["per_class"].items() if row["f1"] > 0.98
    }
    assert not perfect, f"suspiciously separable: {perfect}"
    assert all(row["support"] > 0 for row in metrics["per_class"].values())


def test_the_confusions_are_the_ones_the_emission_model_designed_in(metrics: dict) -> None:
    """Where the model is wrong matters more than how often.

    `emission.py` gives BANK_DOWNTIME and NETWORK_TRANSIENT nearly identical error fields
    on purpose: they are separable only by whether many payments on the same bank failed at
    once. So the largest off-diagonal mass in each of those two rows should be the other
    one. Errors landing somewhere else would mean the model is confused about something the
    simulator did not make confusable, and the accuracy would be measuring the wrong thing.
    """
    order = metrics["class_order"]
    matrix = metrics["confusion_matrix"]
    assert len(matrix) == len(order)
    assert all(len(row) == len(order) for row in matrix)
    assert sum(sum(row) for row in matrix) == metrics["test_size"]

    for cause, confusable in (
        ("BANK_DOWNTIME", "NETWORK_TRANSIENT"),
        ("NETWORK_TRANSIENT", "BANK_DOWNTIME"),
    ):
        row = dict(zip(order, matrix[order.index(cause)]))
        del row[cause]
        worst = max(row, key=row.get)
        assert worst == confusable, f"{cause} is confused mostly with {worst}, not {confusable}"

    hardest = min(metrics["per_class"], key=lambda c: metrics["per_class"][c]["f1"])
    assert hardest in {"NETWORK_TRANSIENT", "BANK_DOWNTIME"}


def test_the_model_is_scored_on_scenarios_it_was_never_trained_on(metrics: dict) -> None:
    """A number quoted only in-distribution does not cover the case it will meet.

    The model is fitted on a normal week. An outage day has a different cause mix and is
    exactly when a merchant needs the diagnosis. Accuracy should fall — it does, most on
    `outage_day` — and it has to stay clear of what the error fields alone would give.
    """
    shift = metrics["under_distribution_shift"]
    assert set(shift) == set(trainer.SHIFT_SCENARIOS)
    bound = metrics["reference_points"]["bayes_optimal_error_fields_only"]
    for name, row in shift.items():
        assert row["payments"] > 0
        assert row["accuracy"] > bound, f"{name} at {row['accuracy']:.2%}, below the bound"
        assert row["accuracy"] <= metrics["accuracy"], (
            f"{name} scores above the in-distribution figure, which means the test seeds "
            f"are harder than the shifted ones and the headline is the wrong way round"
        )


def test_the_cause_mix_the_bound_assumes_is_the_mix_the_data_has(metrics: dict) -> None:
    """The bug that read as leakage for an hour.

    `label_ambiguity()` defaults to a uniform prior; the generator draws from
    `BASE_CAUSE_WEIGHTS`. Comparing the ablation against the uniform figure made a sound fit
    look like a leak. The two bounds are both reported now, and this pins which one the
    comparison uses.
    """
    from app.sim.emission import label_ambiguity

    ref = metrics["reference_points"]
    scenario = SCENARIOS[metrics["data"]["train_scenario"]]
    assert ref["bayes_optimal_error_fields_only"] == pytest.approx(
        label_ambiguity(scenario.normalised_weights())[
            "bayes_optimal_accuracy_error_fields_only"
        ]
    )
    assert ref["bayes_optimal_error_fields_only_uniform_prior"] == pytest.approx(
        label_ambiguity()["bayes_optimal_accuracy_error_fields_only"]
    )
    assert (
        ref["bayes_optimal_error_fields_only"]
        > ref["bayes_optimal_error_fields_only_uniform_prior"]
    ), "the scenario mix is lopsided, so knowing it can only help"


def test_the_bound_rises_as_the_prior_concentrates() -> None:
    """Why the prior is an argument rather than a constant, stated as a property.

    A cause holding all the traffic is predictable from no evidence whatsoever. Any bound
    quoted without its prior is therefore not a bound on anything.
    """
    from app.sim.emission import EMISSIONS, label_ambiguity

    uniform = label_ambiguity()["bayes_optimal_accuracy_error_fields_only"]
    lopsided = label_ambiguity(
        {cause: (1000.0 if cause == "BANK_DOWNTIME" else 1.0) for cause in EMISSIONS}
    )["bayes_optimal_accuracy_error_fields_only"]
    assert lopsided > uniform
    assert label_ambiguity({"BANK_DOWNTIME": 1.0})[
        "bayes_optimal_accuracy_error_fields_only"
    ] == pytest.approx(1.0)
    assert uniform == pytest.approx(
        label_ambiguity({c: 5.0 for c in EMISSIONS})[
            "bayes_optimal_accuracy_error_fields_only"
        ]
    ), "the prior is normalised, so its scale cannot matter"


# ── The endpoint serves the artifact, and says so when there isn't one ─────────


def test_the_metrics_endpoint_reports_the_file_without_recomputing_it(
    metrics: dict,
) -> None:
    """Same property as the eval endpoints: no arithmetic at request time that could drift
    from the artifact the model was saved with. The one derived field is a subtraction, and
    it is asserted against the two numbers it comes from."""
    from app.api import model as endpoint

    served = _run(endpoint.get_model_metrics())
    assert served["accuracy"] == metrics["accuracy"]
    assert served["macro_f1"] == metrics["macro_f1"]
    assert served["confusion_matrix"] == metrics["confusion_matrix"]
    assert served["reference_points"] == metrics["reference_points"]
    assert served["reads_above_the_bound"] == pytest.approx(
        metrics["accuracy"]
        - metrics["reference_points"]["bayes_optimal_error_fields_only"]
    )
    assert served["hardest_class"] == min(
        metrics["per_class"], key=lambda c: metrics["per_class"][c]["f1"]
    )


def test_an_untrained_model_explains_what_it_costs_and_how_to_fix_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """503 with the command, because the consequence of skipping this step is silent.

    `worker.process_failed_payment` returns early when no model is loaded, so a fresh clone
    that never trains does not error — it just never classifies, never chooses a strategy,
    and never recovers anything, while the API stays up.
    """
    from fastapi import HTTPException

    from app.api import model as endpoint

    monkeypatch.setattr(trainer, "METRICS_PATH", METRICS.parent / "absent.json")
    with pytest.raises(HTTPException) as caught:
        _run(endpoint.get_model_metrics())
    assert caught.value.status_code == 503
    assert "python -m app.ml.train" in caught.value.detail["fix"]
    assert "classified" in caught.value.detail["consequence"]


def test_the_api_and_the_command_line_share_one_training_path() -> None:
    """Two ways to fit is how the model answering requests stops being the model the report
    describes. `/model/train` takes seed counts so a demo can run short, and records them,
    but the code it runs is `app.ml.train.main`."""
    import inspect

    from app.api import model as endpoint

    source = inspect.getsource(endpoint)
    assert "trainer.main" in source
    assert "XGBClassifier" not in source, "the endpoint is fitting its own model"
    assert endpoint.trainer is trainer



