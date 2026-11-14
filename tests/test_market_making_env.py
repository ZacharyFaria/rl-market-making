"""Unit tests for REQ-AGT-002: Hard 1000-share inventory limit and -1000.0 breach penalty."""

from __future__ import annotations

import numpy as np

from src.envs.market_making_env import MarketMakingEnv


def test_req_agt_002_inventory_950_plus_100_rejected_with_minus_1000_penalty() -> None:
    """REQ-AGT-002 BDD: Given inventory=950, When bid size=100, discard order, retain 950, reward=-1000.0."""
    env = MarketMakingEnv(regime="hawkes", max_inventory=1000.0, seed=42)
    env.reset(seed=42)

    # Set current inventory to 950 shares
    env.inventory = 950.0

    # Submit a bid execution action of 100 shares (which would breach 950 + 100 = 1050 > 1000)
    state, reward, done, info = env.step((0.05, 100.0))

    assert info["inventory_breach_rejected"] is True
    assert float(state[0]) == 950.0
    assert env.inventory == 950.0
    assert np.isclose(reward, -1000.0)
    assert 5 <= info["latency_ms"] <= 50


def test_req_agt_002_negative_inventory_breach_rejected() -> None:
    """REQ-AGT-002: Short inventory breach (-950 - 100 = -1050 < -1000) is also rejected with -1000.0."""
    env = MarketMakingEnv(regime="hawkes", max_inventory=1000.0, seed=42)
    env.reset(seed=42)
    env.inventory = -950.0

    state, reward, _, info = env.step((0.05, 0.05, 0.0, 100.0))
    assert info["inventory_breach_rejected"] is True
    assert float(state[0]) == -950.0
    assert np.isclose(reward, -1000.0)

