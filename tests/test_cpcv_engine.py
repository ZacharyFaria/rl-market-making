"""Unit tests for REQ-EVAL-001: CPCV 24-hour embargo isolation and Deflated Sharpe Ratio."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.cpcv_engine import CPCVEngine, calculate_dsr


def test_req_eval_001_cpcv_zero_temporal_overlap_and_24h_embargo() -> None:
    """REQ-EVAL-001: Testing folds must have zero intersection with training folds + 24h embargo."""
    # 30 days of hourly observations (720 hours)
    timestamps_hours = np.arange(720, dtype=float)
    engine = CPCVEngine(n_groups=6, n_test_groups=2, embargo_hours=24.0, purge_steps=5)

    folds = list(engine.split(timestamps_hours))
    assert len(folds) == 15  # C(6, 2) = 15

    for train_idx, test_idx, embargo_idx in folds:
        assert len(embargo_idx) > 0
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(np.intersect1d(train_idx, embargo_idx)) == 0

        # Verify no training timestamp lies within (t_test_end, t_test_end + 24.0]
        train_ts = timestamps_hours[train_idx]
        test_ts = timestamps_hours[test_idx]
        for t_te in test_ts:
            in_test = np.any(np.isclose(train_ts, t_te))
            assert not in_test

    # Embargo less than 24 hours must raise ValueError per REQ-EVAL-001 guardrail
    with pytest.raises(ValueError, match="24.0-hour embargo"):
        CPCVEngine(n_groups=6, n_test_groups=2, embargo_hours=12.0)


def test_req_eval_001_deflated_sharpe_ratio_haircut() -> None:
    """REQ-EVAL-001: DSR applies multiple-testing haircut proportional to trial_count."""
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.18, scale=1.0, size=200)
    var_r = float(np.var(returns, ddof=1))

    dsr_5 = calculate_dsr(returns, trial_count=5, return_variance=var_r)
    dsr_100 = calculate_dsr(returns, trial_count=100, return_variance=var_r)

    assert 0.0 <= dsr_100 < dsr_5 <= 1.0
    with pytest.raises(ValueError, match="trial_count must be >= 2"):
        calculate_dsr(returns, trial_count=1, return_variance=var_r)

