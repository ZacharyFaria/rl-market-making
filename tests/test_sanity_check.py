"""Unit tests for REQ-AGT-001: Avellaneda-Stoikov closed-form convergence sanity check."""

from __future__ import annotations

import pytest

from src.envs.market_making_env import MarketMakingEnv
from src.envs.sanity_check import (
    avellaneda_stoikov_optimal_spread,
    avellaneda_stoikov_reservation_price,
    run_poisson_sanity_check,
    validate_as_convergence,
)


def test_req_agt_001_validate_as_convergence_2_percent_threshold() -> None:
    """REQ-AGT-001: Deviation <= 2% passes; deviation > 2% returns False."""
    as_spread = avellaneda_stoikov_optimal_spread(gamma=0.1, sigma=0.2, k=1.5, time_remaining=0.5)
    assert validate_as_convergence(as_spread * 1.015, as_spread) is True
    assert validate_as_convergence(as_spread * 0.985, as_spread) is True
    assert validate_as_convergence(as_spread * 1.025, as_spread) is False
    assert validate_as_convergence(as_spread * 0.975, as_spread) is False


def test_req_agt_001_poisson_baseline_recovery_and_hawkes_gate() -> None:
    """REQ-AGT-001: Baseline Poisson training converges within 2%, and failed check blocks Hawkes."""
    res = run_poisson_sanity_check(gamma=0.10, sigma=0.20, k=1.50, episodes=150, seed=42)
    assert res["passed"] is True
    assert res["relative_error_pct"] < 2.0

    # Reservation price skews downward when long inventory (q > 0) and upward when short (q < 0)
    r_long = avellaneda_stoikov_reservation_price(100.0, inventory=50.0, gamma=0.1, sigma=0.2, time_remaining=1.0)
    r_short = avellaneda_stoikov_reservation_price(100.0, inventory=-50.0, gamma=0.1, sigma=0.2, time_remaining=1.0)
    assert r_long < 100.0 < r_short

    # Guardrail: Hawkes environment initialization MUST raise RuntimeError if sanity check fails
    with pytest.raises(RuntimeError, match="REQ-AGT-001 Violation"):
        MarketMakingEnv(regime="hawkes", agent_as_spread_pair=(1.50, 1.00))
    with pytest.raises(RuntimeError, match="REQ-AGT-001 Violation"):
        MarketMakingEnv(regime="hawkes", sanity_check_validated=False)

