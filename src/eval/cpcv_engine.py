"""Combinatorial Purged Cross-Validation (CPCV) & Deflated Sharpe Ratio Engine.

Implements REQ-EVAL-001: Calculates the Deflated Sharpe Ratio (DSR) applying a
multiple-testing statistical haircut proportional to `trial_count`, and constructs
CPCV folds that strictly excise overlapping timestamps plus a continuous 24-hour
embargo window.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator

import numpy as np
from scipy.stats import kurtosis, norm, skew


def calculate_dsr(
    returns_array: np.ndarray | list[float],
    trial_count: int,
    return_variance: float | None = None,
) -> float:
    """Compute the Bailey & López de Prado (2014) Deflated Sharpe Ratio (REQ-EVAL-001).

    Applies a multiple-testing statistical haircut proportional to `trial_count` ($K$)
    and adjusts for non-Normal skewness and kurtosis:

    .. math::
        \\widehat{\\text{SR}}_0 = \\sqrt{V[\\widehat{\\text{SR}}]} \\left[
            (1 - \\gamma) \\Phi^{-1}\\left(1 - \\frac{1}{K}\\right)
            + \\gamma \\Phi^{-1}\\left(1 - \\frac{1}{K e}\\right)
        \\right]

    Parameters
    ----------
    returns_array : np.ndarray | list[float]
        1D array of out-of-sample periodic returns or episode PnLs.
    trial_count : int
        Total count of evaluated hyperparameter configurations ($K \\ge 2$).
    return_variance : float | None, optional
        Sample variance of returns or cross-trial Sharpe variance scale.

    Returns
    -------
    float
        Deflated Sharpe Ratio probability in $[0.0, 1.0]$.
    """
    if int(trial_count) <= 1:
        raise ValueError(
            f"trial_count must be >= 2 for multiple-testing DSR adjustment (got {trial_count}, REQ-EVAL-001)."
        )

    ret = np.asarray(returns_array, dtype=float).reshape(-1)
    ret = ret[~np.isnan(ret)]
    n = len(ret)
    if n < 3:
        return 0.0

    var_r = float(return_variance) if return_variance is not None and return_variance > 0 else float(np.var(ret, ddof=1))
    std_r = np.sqrt(max(var_r, 1e-12))
    sr_hat = float(np.mean(ret) / std_r)

    g3 = float(skew(ret, bias=False))
    g4 = float(kurtosis(ret, fisher=False, bias=False))

    euler_gamma = 0.5772156649015329
    k = float(trial_count)
    expected_max_z = (1.0 - euler_gamma) * float(norm.ppf(1.0 - 1.0 / k)) + euler_gamma * float(
        norm.ppf(1.0 - 1.0 / (k * np.e))
    )
    sr_trial_std = max(1.0 / np.sqrt(max(n - 1, 1)), 0.05)
    sr_0 = sr_trial_std * expected_max_z

    denom = np.sqrt(max(1.0 - g3 * sr_hat + 0.25 * (g4 - 1.0) * (sr_hat**2), 1e-8))
    z_stat = ((sr_hat - sr_0) * np.sqrt(n - 1.0)) / denom
    return float(np.clip(norm.cdf(z_stat), 0.0, 1.0))


class CPCVEngine:
    """Combinatorial Purged Cross-Validation Engine with 24-Hour Embargo (REQ-EVAL-001).

    Ensures zero temporal overlap between any testing fold and the corresponding
    training fold plus its contiguous 24-hour embargo window.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_hours: float = 24.0,
        purge_steps: int = 10,
    ) -> None:
        if n_groups < 3 or n_test_groups < 1 or n_test_groups >= n_groups:
            raise ValueError("Require n_groups >= 3 and 1 <= n_test_groups < n_groups.")
        if embargo_hours < 24.0:
            raise ValueError(
                f"REQ-EVAL-001 requires at least a 24.0-hour embargo period (got {embargo_hours})."
            )
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_hours = float(embargo_hours)
        self.purge_steps = int(purge_steps)

    def split(
        self,
        timestamps_hours: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield `(train_indices, test_indices, embargo_indices)` with zero overlap.

        Parameters
        ----------
        timestamps_hours : np.ndarray
            1D monotonically increasing array of observation timestamps in hours.

        Yields
        ------
        train_idx : np.ndarray
            Training fold indices with all test, purge, and 24-hour embargo points excised.
        test_idx : np.ndarray
            Out-of-sample testing fold indices.
        embargo_idx : np.ndarray
            Excised embargo window indices immediately following each test segment.
        """
        ts = np.asarray(timestamps_hours, dtype=float).reshape(-1)
        n = len(ts)
        all_idx = np.arange(n, dtype=int)
        groups = np.array_split(all_idx, self.n_groups)

        for test_gids in combinations(range(self.n_groups), self.n_test_groups):
            test_mask = np.zeros(n, dtype=bool)
            embargo_mask = np.zeros(n, dtype=bool)
            purge_mask = np.zeros(n, dtype=bool)

            for gid in test_gids:
                g = groups[gid]
                if len(g) == 0:
                    continue
                start_i, end_i = int(g[0]), int(g[-1])
                test_mask[start_i : end_i + 1] = True

                # Pre-test purge window
                p_start = max(0, start_i - self.purge_steps)
                purge_mask[p_start:start_i] = True

                # Post-test continuous 24-hour embargo window (or pre-test 24-hour buffer if test block is at terminal boundary)
                t_end_hr = ts[end_i]
                emb_cond = (ts > t_end_hr) & (ts <= t_end_hr + self.embargo_hours)
                if not np.any(emb_cond) and start_i > 0:
                    t_start_hr = ts[start_i]
                    emb_cond = (ts >= t_start_hr - self.embargo_hours) & (ts < t_start_hr)
                embargo_mask |= emb_cond

            embargo_mask &= ~test_mask
            if not np.any(embargo_mask):
                # When test groups occupy the very end of the timeline, excise the 24-hour buffer preceding the first test group
                first_test_i = int(np.flatnonzero(test_mask)[0])
                t_start_hr = ts[first_test_i]
                embargo_mask = (ts >= t_start_hr - self.embargo_hours) & (ts < t_start_hr) & (~test_mask)
            excluded = test_mask | embargo_mask | purge_mask
            train_idx = all_idx[~excluded]
            test_idx = all_idx[test_mask]
            embargo_idx = all_idx[embargo_mask]

            # Strict guardrail check: zero intersection between train and (test U embargo)
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            assert len(np.intersect1d(train_idx, embargo_idx)) == 0
            yield train_idx, test_idx, embargo_idx


def compute_cvar_95(pnl_array: np.ndarray) -> float:
    """Compute 95% Conditional Value at Risk (Expected Shortfall in the left 5% tail)."""
    arr = np.asarray(pnl_array, dtype=float).reshape(-1)
    if len(arr) == 0:
        return 0.0
    cutoff = float(np.percentile(arr, 5.0))
    tail = arr[arr <= cutoff]
    return float(np.mean(tail)) if len(tail) > 0 else cutoff


def compute_sortino(pnl_array: np.ndarray, mar: float | None = None) -> float:
    """Compute Sortino ratio penalized strictly by downside semi-deviation below MAR."""
    arr = np.asarray(pnl_array, dtype=float).reshape(-1)
    if len(arr) < 2:
        return 0.0
    target = float(np.median(arr)) if mar is None else float(mar)
    downside = np.minimum(arr - target, 0.0)
    down_std = float(np.sqrt(np.mean(downside**2)))
    return float(np.mean(arr) / max(down_std, 1e-6))
