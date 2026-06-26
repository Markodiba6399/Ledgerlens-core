"""Benford's Law digit-distribution analysis for transaction amounts.

Computes the chi-square statistic, per-digit Z-scores, and Mean Absolute
Deviation (MAD) of the leading-digit distribution of a set of amounts,
relative to the theoretical Benford distribution.

For wallets with fewer than BENFORD_BOOTSTRAP_THRESHOLD transactions the
asymptotic chi-square p-value is unreliable (the approximation requires
N * p_i >= 5 for every digit class). In that regime, bootstrap_chi_square_pvalue
generates 10,000 multinomial samples from the true Benford distribution and
derives an empirical p-value directly, eliminating false positives caused by
approximation breakdown at small N.
"""

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
from scipy.stats import chi2

DIGITS = list(range(1, 10))

# P(d) = log10(1 + 1/d) for d in 1..9  (dict form, kept for backward compatibility)
BENFORD_EXPECTED: dict[int, float] = {d: math.log10(1 + 1 / d) for d in DIGITS}

# Numpy array form of Benford probabilities for vectorised bootstrap operations.
# Index 0 → digit 1, index 8 → digit 9.
BENFORD_PROBS: np.ndarray = np.array([math.log10(1 + 1 / d) for d in DIGITS])
BENFORD_PROBS = BENFORD_PROBS / BENFORD_PROBS.sum()  # normalise; sum is ~1.0 by construction

# Bootstrap configuration — overridable via environment variables.
# Asymptotic chi-square p-values are valid only when N * p_min >= 5, where
# p_min = P(digit=9) ≈ 0.046. That requires N >= 5/0.046 ≈ 109. We use 100
# as a conservative threshold so the bootstrap kicks in well before the
# asymptotic approximation degrades noticeably.
BENFORD_BOOTSTRAP_THRESHOLD: int = int(os.getenv("BENFORD_BOOTSTRAP_THRESHOLD", "100"))
BENFORD_BOOTSTRAP_SAMPLES: int = int(os.getenv("BENFORD_BOOTSTRAP_SAMPLES", "10000"))


@dataclass
class BenfordWindowFeatures:
    """Typed Benford features for a single rolling time-window."""

    window_hours: int
    n_transactions: int
    chi_square_stat: float
    chi_square_pvalue: float
    chi_square_pvalue_method: Literal["asymptotic", "bootstrap"]
    mad: float
    z_scores: list[float]  # 9 values, index 0 = digit 1 … index 8 = digit 9
    benford_flag: bool


# --------------------------------------------------------------------------- #
# Core helpers (dict-based, backward-compatible with feature_engineering.py)
# --------------------------------------------------------------------------- #

def first_digit(value: float) -> int | None:
    """Return the leading (most significant) decimal digit of `value`.

    Returns None for zero, negative, or non-finite values, which are
    excluded from Benford analysis.
    """
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    while value < 1:
        value *= 10
    while value >= 10:
        value /= 10
    return int(value)


def digit_distribution(amounts: list[float]) -> dict[int, float]:
    """Return the observed proportion of each leading digit 1-9 in `amounts`."""
    digits = [d for d in (first_digit(a) for a in amounts) if d is not None]
    n = len(digits)
    if n == 0:
        return {d: 0.0 for d in DIGITS}
    counts = {d: 0 for d in DIGITS}
    for d in digits:
        counts[d] += 1
    return {d: counts[d] / n for d in DIGITS}


def chi_square_statistic(observed: dict[int, float], n: int) -> float:
    """Chi-square goodness-of-fit statistic vs. the Benford distribution.

    `observed` is a digit -> proportion mapping (e.g. from `digit_distribution`).
    `n` is the number of observations the proportions were computed from.
    """
    if n == 0:
        return 0.0
    chi_sq = 0.0
    for d in DIGITS:
        expected_count = BENFORD_EXPECTED[d] * n
        observed_count = observed.get(d, 0.0) * n
        if expected_count > 0:
            chi_sq += (observed_count - expected_count) ** 2 / expected_count
    return chi_sq


def z_scores(observed: dict[int, float], n: int) -> dict[int, float]:
    """Per-digit Z-score of the observed proportion vs. Benford's expectation."""
    if n == 0:
        return {d: 0.0 for d in DIGITS}
    scores = {}
    for d in DIGITS:
        p = BENFORD_EXPECTED[d]
        observed_p = observed.get(d, 0.0)
        # continuity correction as commonly used in Benford forensic analysis
        numerator = abs(observed_p - p) - (1 / (2 * n))
        denominator = math.sqrt(p * (1 - p) / n)
        scores[d] = max(numerator, 0.0) / denominator if denominator > 0 else 0.0
    return scores


def mean_absolute_deviation(observed: dict[int, float]) -> float:
    """MAD between observed and expected digit distributions.

    Values above ~0.015 (for first-digit tests) are commonly treated as
    indicating non-conformity with Benford's Law.
    """
    deviations = [abs(observed.get(d, 0.0) - BENFORD_EXPECTED[d]) for d in DIGITS]
    return float(np.mean(deviations))


# --------------------------------------------------------------------------- #
# Bootstrap infrastructure
# --------------------------------------------------------------------------- #

def _chi_sq_from_counts(observed: np.ndarray, expected: np.ndarray) -> float:
    """Chi-square statistic for raw count arrays (not proportions).

    Uses a small epsilon in the denominator to avoid division by zero when
    expected counts are negligibly small.
    """
    return float(np.sum((observed - expected) ** 2 / (expected + 1e-9)))


def bootstrap_chi_square_pvalue(
    observed_counts: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: Optional[int] = None,
) -> float:
    """Monte Carlo bootstrap p-value for the Benford chi-square test.

    Generates `n_bootstrap` multinomial samples of size N drawn from the
    theoretical Benford distribution, computes the chi-square statistic for
    each, and returns the fraction that equals or exceeds the observed
    statistic — this fraction is the empirical p-value.

    The result is valid for any sample size, unlike the asymptotic chi-square
    approximation which requires N * p_i >= 5 for every digit class i.

    Args:
        observed_counts: array of shape (9,) with leading-digit counts for
            digits 1–9 (index 0 = digit 1, index 8 = digit 9).
        n_bootstrap: number of bootstrap replicates (default 10,000).
        seed: RNG seed for reproducibility. Use an integer in tests; leave as
            None in production so each call draws fresh randomness.

    Returns:
        Empirical p-value in (0, 1]. Never returns exactly 0.0; the floor is
        1 / n_bootstrap so that zero p-values are not reported.
    """
    N = int(observed_counts.sum())
    if N == 0:
        return 1.0

    expected = BENFORD_PROBS * N
    observed_stat = _chi_sq_from_counts(observed_counts, expected)

    rng = np.random.default_rng(seed)
    # Vectorised: all n_bootstrap samples in one call → shape (n_bootstrap, 9)
    bootstrap_samples = rng.multinomial(N, BENFORD_PROBS, size=n_bootstrap)

    bootstrap_stats = np.sum(
        (bootstrap_samples - expected) ** 2 / (expected + 1e-9),
        axis=1,
    )

    p_value = max(float((bootstrap_stats >= observed_stat).mean()), 1.0 / n_bootstrap)
    return p_value


@lru_cache(maxsize=512)
def _cached_bootstrap_pvalue(
    counts_tuple: tuple,
    n_bootstrap: int,
    seed: Optional[int],
) -> float:
    """LRU-cached wrapper around bootstrap_chi_square_pvalue.

    Cache key is (counts_tuple, n_bootstrap, seed). In production seed=None,
    so all calls with the same counts and n_bootstrap share one cache entry
    within a process lifetime — intentional, since repeated scoring of the
    same wallet window should return the same p-value.
    """
    counts = np.array(counts_tuple)
    return bootstrap_chi_square_pvalue(counts, n_bootstrap, seed)


def compute_chi_square_pvalue(counts: np.ndarray, N: int) -> tuple[float, str]:
    """Return (p_value, method) for the Benford chi-square test.

    Selects bootstrap when N < BENFORD_BOOTSTRAP_THRESHOLD (asymptotic
    approximation is unreliable for small samples), otherwise uses the
    asymptotic chi-square survival function with 8 degrees of freedom.

    Args:
        counts: array of shape (9,) with raw digit counts (not proportions).
        N: total number of observations (typically counts.sum(); passed
           explicitly to avoid recomputation and handle edge cases cleanly).

    Returns:
        Tuple of (p_value, method) where method is "bootstrap" or "asymptotic".
    """
    if N < BENFORD_BOOTSTRAP_THRESHOLD:
        p = _cached_bootstrap_pvalue(
            tuple(int(c) for c in counts),
            BENFORD_BOOTSTRAP_SAMPLES,
            None,
        )
        return p, "bootstrap"

    expected = BENFORD_PROBS * N
    stat = _chi_sq_from_counts(counts, expected)
    p = float(chi2.sf(stat, df=8))
    return p, "asymptotic"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def compute_benford_metrics(amounts: list[float]) -> dict:
    """Compute the full set of Benford metrics for a list of transaction amounts.

    Returns a dict with:
      - ``chi_square``: raw chi-square statistic
      - ``chi_square_pvalue``: p-value (bootstrap or asymptotic, see below)
      - ``pvalue_method``: ``"bootstrap"`` when N < BENFORD_BOOTSTRAP_THRESHOLD,
        ``"asymptotic"`` otherwise
      - ``mad``: Mean Absolute Deviation
      - ``z_scores``: per-digit Z-scores (dict[int, float])
      - ``observed_distribution``: digit -> proportion mapping
      - ``sample_size``: number of valid (positive, finite) amounts
    """
    observed = digit_distribution(amounts)
    n = sum(1 for a in amounts if first_digit(a) is not None)

    counts = np.array([observed[d] * n for d in DIGITS])
    p_value, p_method = compute_chi_square_pvalue(counts, n)

    return {
        "chi_square": chi_square_statistic(observed, n),
        "chi_square_pvalue": p_value,
        "pvalue_method": p_method,
        "mad": mean_absolute_deviation(observed),
        "z_scores": z_scores(observed, n),
        "observed_distribution": observed,
        "sample_size": n,
    }


def is_anomalous(metrics: dict, mad_threshold: float = 0.015) -> bool:
    """Whether a `compute_benford_metrics` result exceeds the MAD threshold."""
    return metrics["mad"] > mad_threshold
