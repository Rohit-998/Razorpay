"""Observation model: latent root cause → the error fields a webhook actually carries.

This module is where the original build had a fatal flaw worth naming, because
avoiding it is most of what makes the classifier meaningful.

If root cause deterministically sets `error_reason`, `error_step` and
`error_source`, and those three fields are then handed to a classifier as
features, the classifier is not diagnosing anything — it is inverting a lookup
table that the author wrote. It will report 95%+ accuracy, SHAP will faithfully
report that `error_reason` explained everything, and neither number will mean
anything at all.

So the emission distributions below overlap heavily, on purpose:

  * `payment_failed` and `other` — genuinely the most common reasons in
    production — appear under almost every cause and carry near-zero signal.
  * BANK_DOWNTIME and NETWORK_TRANSIENT emit nearly identical fields. They are
    separable only by whether many payments on the same bank failed at the same
    time, so the correlated-failure feature has to do real work.
  * AUTH_TIMEOUT and WRONG_CREDENTIALS both surface as
    `customer / payment_authentication / authentication_failed`. Separating them
    requires customer history: a reliable payer who times out abandoned an OTP,
    a repeat failer has stale details on file.
  * `payment_cancelled` reads like an abandoned checkout, and mostly is — but it
    also appears when someone gives up at the OTP screen, closes a hanging UPI
    app, or sees a decline and backs out. No observable value in this module is
    unique to one cause; `tests/test_sim.py` asserts it.
  * INSUFFICIENT_FUNDS names itself only about half the time. The rest hides in
    the generic bucket and is recoverable mainly from timing relative to salary
    and month-end.

The result is a well-posed inference problem in which all four feature families
carry signal and none of them dominate. Accuracy lands in the seventies, which is
a number worth reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ERROR_REASONS: tuple[str, ...] = (
    "payment_failed",
    "gateway_technical_error",
    "network_error",
    "timeout",
    "authentication_failed",
    "payment_cancelled",
    "insufficient_funds",
    "card_blocked",
    "invalid_card",
    "mandate_expired",
    "upi_psp_error",
    "bank_not_enabled",
    "limit_exceeded",
    "other",
)

ERROR_STEPS: tuple[str, ...] = (
    "payment_initiation",
    "payment_authentication",
    "payment_authorization",
    "payment_processing",
    "payment_capture",
)

ERROR_SOURCES: tuple[str, ...] = ("customer", "gateway", "business", "razorpay")


@dataclass(frozen=True)
class EmissionProfile:
    """Distributions over observable error fields for one latent cause."""

    reason: dict[str, float]
    step: dict[str, float]
    source: dict[str, float]


EMISSIONS: dict[str, EmissionProfile] = {
    "BANK_DOWNTIME": EmissionProfile(
        reason={
            "gateway_technical_error": 0.30, "timeout": 0.20, "payment_failed": 0.18,
            "upi_psp_error": 0.15, "network_error": 0.12, "bank_not_enabled": 0.05,
        },
        step={
            "payment_initiation": 0.41, "payment_authorization": 0.29,
            "payment_processing": 0.21, "payment_authentication": 0.09,
        },
        source={"gateway": 0.73, "razorpay": 0.17, "customer": 0.10},
    ),
    "NETWORK_TRANSIENT": EmissionProfile(
        reason={
            "network_error": 0.27, "timeout": 0.23, "gateway_technical_error": 0.19,
            "payment_failed": 0.15, "payment_cancelled": 0.09, "upi_psp_error": 0.07,
        },
        step={
            "payment_initiation": 0.39, "payment_authentication": 0.25,
            "payment_authorization": 0.22, "payment_processing": 0.14,
        },
        source={"gateway": 0.59, "razorpay": 0.23, "customer": 0.18},
    ),
    "AUTH_TIMEOUT": EmissionProfile(
        reason={
            "timeout": 0.27, "authentication_failed": 0.25, "payment_cancelled": 0.24,
            "payment_failed": 0.16, "other": 0.08,
        },
        step={
            "payment_authentication": 0.61, "payment_authorization": 0.23,
            "payment_initiation": 0.16,
        },
        source={"customer": 0.65, "gateway": 0.29, "razorpay": 0.06},
    ),
    "INSUFFICIENT_FUNDS": EmissionProfile(
        reason={
            "insufficient_funds": 0.47, "payment_failed": 0.21,
            "limit_exceeded": 0.13, "other": 0.11, "payment_cancelled": 0.08,
        },
        step={
            "payment_authorization": 0.43, "payment_processing": 0.35,
            "payment_authentication": 0.22,
        },
        source={"customer": 0.79, "gateway": 0.17, "business": 0.04},
    ),
    "WRONG_CREDENTIALS": EmissionProfile(
        reason={
            "authentication_failed": 0.36, "payment_failed": 0.18,
            "invalid_card": 0.16, "other": 0.12, "payment_cancelled": 0.12,
            "card_blocked": 0.06,
        },
        step={
            "payment_authentication": 0.65, "payment_authorization": 0.25,
            "payment_initiation": 0.10,
        },
        source={"customer": 0.81, "gateway": 0.15, "razorpay": 0.04},
    ),
    "PERMANENT_DECLINE": EmissionProfile(
        reason={
            "card_blocked": 0.23, "payment_failed": 0.23, "invalid_card": 0.18,
            "mandate_expired": 0.15, "limit_exceeded": 0.13,
            "insufficient_funds": 0.08,
        },
        step={
            "payment_authorization": 0.45, "payment_processing": 0.31,
            "payment_authentication": 0.24,
        },
        source={"gateway": 0.47, "customer": 0.42, "business": 0.11},
    ),
    "MERCHANT_ERROR": EmissionProfile(
        reason={
            "bank_not_enabled": 0.36, "other": 0.26, "payment_failed": 0.21,
            "gateway_technical_error": 0.10, "mandate_expired": 0.07,
        },
        step={
            "payment_initiation": 0.65, "payment_processing": 0.21,
            "payment_authorization": 0.14,
        },
        source={"business": 0.61, "gateway": 0.25, "customer": 0.14},
    ),
}

DESCRIPTIONS: dict[str, str] = {
    "payment_failed": "Payment failed. Please try again.",
    "gateway_technical_error": "Payment processing failed due to a technical error at the gateway.",
    "network_error": "Payment could not be completed due to a network connectivity issue.",
    "timeout": "The payment request timed out before completing.",
    "authentication_failed": "Payment authentication failed.",
    "payment_cancelled": "The payment was cancelled before completion.",
    "insufficient_funds": "The payment was declined due to insufficient balance.",
    "card_blocked": "The card has been blocked by the issuing bank.",
    "invalid_card": "The card details could not be verified.",
    "mandate_expired": "The autopay mandate for this payment has expired.",
    "upi_psp_error": "The UPI app reported an error while processing this payment.",
    "bank_not_enabled": "This bank is not enabled for the selected payment method.",
    "limit_exceeded": "The transaction exceeds a limit set on the account.",
    "other": "Payment could not be completed.",
}


def _draw(rng: np.random.Generator, distribution: dict[str, float]) -> str:
    keys = list(distribution)
    weights = np.array([distribution[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=weights / weights.sum()))]


@dataclass(frozen=True)
class Emission:
    """One sampled observable failure signature."""

    error_code: str
    error_source: str
    error_step: str
    error_reason: str
    error_description: str


def emit(rng: np.random.Generator, cause: str) -> Emission:
    """Sample the observable error fields a webhook would carry for this cause."""
    profile = EMISSIONS[cause]
    source = _draw(rng, profile.source)
    step = _draw(rng, profile.step)
    reason = _draw(rng, profile.reason)

    # Razorpay-shaped top-level code. Mostly a function of source, with the same
    # sloppiness real gateways exhibit, so it adds little beyond `error_source`.
    if source == "customer":
        code = "BAD_REQUEST_ERROR" if rng.random() < 0.88 else "GATEWAY_ERROR"
    elif source == "business":
        code = "BAD_REQUEST_ERROR" if rng.random() < 0.72 else "SERVER_ERROR"
    elif source == "razorpay":
        code = "SERVER_ERROR" if rng.random() < 0.66 else "GATEWAY_ERROR"
    else:
        code = "GATEWAY_ERROR" if rng.random() < 0.83 else "BAD_REQUEST_ERROR"

    return Emission(
        error_code=code,
        error_source=source,
        error_step=step,
        error_reason=reason,
        error_description=DESCRIPTIONS.get(reason, "Payment could not be completed."),
    )


def label_ambiguity(prior: dict[str, float] | None = None) -> dict[str, float]:
    """Bayes-optimal accuracy achievable from the error triple alone.

    Reported in the eval output as a sanity bound: it is the score a model gets
    from the error fields with no bank, customer or timing context. A classifier
    that does not clear it is not adding anything, and one that reports far above
    it is leaking.

    `prior` is the share of each cause in the traffic being scored, and it is not a
    detail. The bound rises as the prior gets more lopsided, because a Bayes
    classifier facing an ambiguous signature can fall back on which cause is more
    common — at the limit, one cause with all the mass is predictable at 100% from
    no evidence whatsoever. So a bound computed under a uniform prior is the right
    reference for "no knowledge of the mix" and the wrong one to hold a model
    trained on a specific scenario against; that model has learned the mix, and
    will clear the uniform figure without leaking anything. Defaults to uniform,
    which is what the eval report quotes.
    """
    from collections import defaultdict

    if prior is None:
        prior = {cause: 1.0 / len(EMISSIONS) for cause in EMISSIONS}
    total_mass = sum(prior.values())
    prior = {cause: prior.get(cause, 0.0) / total_mass for cause in EMISSIONS}

    joint: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for cause, profile in EMISSIONS.items():
        for source, p_source in profile.source.items():
            for step, p_step in profile.step.items():
                for reason, p_reason in profile.reason.items():
                    mass = prior[cause] * p_source * p_step * p_reason
                    joint[(source, step, reason)][cause] = mass

    total = 0.0
    correct = 0.0
    for cell, masses in joint.items():
        cell_mass = sum(masses.values())
        total += cell_mass
        correct += max(masses.values())
    return {
        "bayes_optimal_accuracy_error_fields_only": round(correct / total, 4),
        "distinct_signatures": len(joint),
    }
