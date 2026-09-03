"""
Recovery simulation environment.

This package is the ground truth of the project. Everything a policy is
measured on happens here, and nothing in here can see a policy's internals.

The design rule that matters: an outcome is never a function of a label.
Whether a retry succeeds depends on whether the bank is actually up at that
moment and whether the customer's money has actually arrived. Whether a
payment link converts depends on how many times we already pinged that person.
Root cause is a *latent* variable that policies must infer from noisy,
deliberately ambiguous observations.

The environment also knows, for every customer, when they would have paid on
their own with no help from us. That single fact is what makes honest
attribution and incremental-lift measurement possible.
"""

from app.sim.types import (
    Action,
    ActionType,
    AttributionTruth,
    Channel,
    Observation,
    StepResult,
    Terminal,
    Tone,
)

__all__ = [
    "Action",
    "ActionType",
    "AttributionTruth",
    "Channel",
    "Observation",
    "StepResult",
    "Terminal",
    "Tone",
]
