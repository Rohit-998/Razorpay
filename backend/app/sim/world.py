"""Bank reliability and downtime — the environment's other latent process.

Downtime is sampled per scenario before any policy runs, so every policy faces
the identical outage schedule. A policy cannot query it. What it *can* observe is
the merchant's own failure stream: when six SBI payments fail in ten minutes,
something is wrong with SBI. That correlated-failure signal is genuinely
informative and genuinely noisy, which is what makes the inference problem real
rather than a lookup.

Timing is the whole game for downtime. A retry fired during an outage is wasted
spend; the same retry fired twenty minutes after recovery succeeds. Nothing in
the observation tells a policy when recovery happened, so it has to reason about
it from the failure stream and from how long the outage has already run.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class BankProfile:
    """Per-bank reliability. Shares and rates are plausible, not scraped."""

    code: str
    name: str
    share: float
    """Share of this merchant's payment volume."""
    base_success_rate: float
    """Steady-state authorisation rate outside any outage."""
    downtime_episodes_per_week: float
    mean_downtime_minutes: float


BANKS: tuple[BankProfile, ...] = (
    BankProfile("SBIN", "State Bank of India", 0.21, 0.938, 3.1, 52.0),
    BankProfile("HDFC", "HDFC Bank", 0.18, 0.973, 1.1, 26.0),
    BankProfile("ICIC", "ICICI Bank", 0.15, 0.968, 1.4, 31.0),
    BankProfile("UTIB", "Axis Bank", 0.12, 0.961, 1.7, 34.0),
    BankProfile("KKBK", "Kotak Mahindra Bank", 0.08, 0.965, 1.3, 28.0),
    BankProfile("BARB", "Bank of Baroda", 0.07, 0.929, 3.4, 58.0),
    BankProfile("PUNB", "Punjab National Bank", 0.06, 0.921, 3.8, 64.0),
    BankProfile("YESB", "Yes Bank", 0.05, 0.944, 2.6, 41.0),
    BankProfile("IOBA", "Indian Overseas Bank", 0.04, 0.907, 4.2, 71.0),
    BankProfile("CNRB", "Canara Bank", 0.04, 0.925, 3.5, 55.0),
)

BANK_BY_CODE: dict[str, BankProfile] = {b.code: b for b in BANKS}


@dataclass(frozen=True)
class DowntimeEpisode:
    """One outage window. `success_rate` applies to every charge inside it."""

    bank: str
    start: datetime
    end: datetime
    severity: str
    success_rate: float

    def covers(self, t: datetime) -> bool:
        return self.start <= t < self.end


SEVERITIES: tuple[tuple[str, float, float], ...] = (
    # (label, probability, success rate during the outage)
    ("high", 0.30, 0.03),
    ("medium", 0.45, 0.22),
    ("low", 0.25, 0.58),
)


class World:
    """Bank state over the simulated window. Deterministic given a seed."""

    def __init__(
        self,
        rng: np.random.Generator,
        start: datetime,
        duration_days: int,
        downtime_multiplier: float = 1.0,
    ) -> None:
        self.start = start
        self.end = start + timedelta(days=duration_days)
        self.duration_days = duration_days
        self.episodes: list[DowntimeEpisode] = self._sample_downtime(
            rng, duration_days, downtime_multiplier
        )
        self._by_bank: dict[str, list[DowntimeEpisode]] = {}
        for episode in self.episodes:
            self._by_bank.setdefault(episode.bank, []).append(episode)
        for windows in self._by_bank.values():
            windows.sort(key=lambda e: e.start)

        # Failure timeline, populated by the environment once episodes exist. This
        # is what a merchant can actually see, and the only downtime evidence a
        # policy gets.
        self._failure_times: dict[str, list[datetime]] = {}

    def _sample_downtime(
        self, rng: np.random.Generator, duration_days: int, multiplier: float
    ) -> list[DowntimeEpisode]:
        episodes: list[DowntimeEpisode] = []
        labels = [s[0] for s in SEVERITIES]
        probs = np.array([s[1] for s in SEVERITIES], dtype=float)
        rates = {s[0]: s[2] for s in SEVERITIES}

        for bank in BANKS:
            expected = bank.downtime_episodes_per_week * (duration_days / 7.0) * multiplier
            for _ in range(int(rng.poisson(max(0.0, expected)))):
                offset_minutes = float(rng.uniform(0, duration_days * 1440))
                start = self.start + timedelta(minutes=offset_minutes)
                minutes = float(np.clip(
                    rng.lognormal(mean=np.log(bank.mean_downtime_minutes), sigma=0.65),
                    5.0, 600.0,
                ))
                severity = labels[int(rng.choice(len(labels), p=probs / probs.sum()))]
                episodes.append(DowntimeEpisode(
                    bank=bank.code,
                    start=start,
                    end=start + timedelta(minutes=minutes),
                    severity=severity,
                    success_rate=rates[severity],
                ))
        return episodes

    # ── Ground truth (policies cannot call these) ────────────────────────

    def downtime_at(self, bank: str, t: datetime) -> DowntimeEpisode | None:
        """The outage covering `t` for this bank, if any."""
        for episode in self._by_bank.get(bank, ()):
            if episode.covers(t):
                return episode
            if episode.start > t:
                break
        return None

    def success_probability(self, bank: str, t: datetime) -> float:
        """Probability a charge on this bank authorises at `t`, ignoring the customer."""
        profile = BANK_BY_CODE.get(bank)
        base = profile.base_success_rate if profile else 0.95
        episode = self.downtime_at(bank, t)
        return episode.success_rate if episode else base

    # ── Observable signal ────────────────────────────────────────────────

    def register_failures(self, failures: list[tuple[str, datetime]]) -> None:
        """Record the merchant's own failure stream: (bank, timestamp) pairs."""
        timeline: dict[str, list[datetime]] = {}
        for bank, t in failures:
            timeline.setdefault(bank, []).append(t)
        for times in timeline.values():
            times.sort()
        self._failure_times = timeline

    def observed_failures(self, bank: str, t: datetime, window_minutes: int = 60) -> int:
        """Failures this merchant saw on `bank` in the hour before `t`."""
        times = self._failure_times.get(bank)
        if not times:
            return 0
        lo = bisect.bisect_left(times, t - timedelta(minutes=window_minutes))
        hi = bisect.bisect_right(times, t)
        return max(0, hi - lo)

    def baseline_failures_per_hour(self, bank: str) -> float:
        """Expected hourly failure count for this bank across the whole window."""
        times = self._failure_times.get(bank, ())
        hours = max(1.0, self.duration_days * 24.0)
        return len(times) / hours

    def failure_spike(self, bank: str, t: datetime) -> bool:
        """Noisy downtime proxy: is this bank failing far above its own baseline?"""
        observed = self.observed_failures(bank, t)
        baseline = self.baseline_failures_per_hour(bank)
        return observed >= 3 and observed > max(2.0, baseline * 2.5)

    def observed_success_rate(self, bank: str, t: datetime) -> float:
        """What the merchant would compute as this bank's recent success rate.

        Successes are not simulated per-bank outside recovery, so this is derived
        from the failure count against expected volume — the same rough estimate a
        real dashboard shows.
        """
        profile = BANK_BY_CODE.get(bank)
        base = profile.base_success_rate if profile else 0.95
        observed = self.observed_failures(bank, t)
        baseline = max(0.5, self.baseline_failures_per_hour(bank))
        if observed <= baseline:
            return round(base, 4)
        excess = min(1.0, (observed - baseline) / (baseline * 6.0))
        return round(float(max(0.02, base * (1.0 - excess))), 4)
