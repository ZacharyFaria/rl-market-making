"""Unit tests for REQ-SIM-001: Uniform 5-50 ms stochastic latency in matching engine."""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.matching_engine import Order, apply_stochastic_latency


def test_req_sim_001_uniform_latency_between_5_and_50ms() -> None:
    """REQ-SIM-001: Every limit order submission receives an integer delay in [5, 50] ms."""
    rng = np.random.default_rng(42)
    delays: list[int] = []
    for i in range(500):
        order = Order(order_id=f"O-{i}", side="BUY", price=100.0, size=10.0, submission_time_ms=1000)
        delay = apply_stochastic_latency(order, rng)
        assert isinstance(delay, int)
        assert 5 <= delay <= 50
        assert order.insertion_time_ms == 1000 + delay
        delays.append(delay)

    assert min(delays) == 5
    assert max(delays) == 50
    assert 25.0 <= float(np.mean(delays)) <= 30.0


def test_req_sim_001_prohibits_zero_or_sub_5ms_latency() -> None:
    """REQ-SIM-001: Zero or sub-5ms latency is strictly prohibited in Hawkes mode."""
    rng = np.random.default_rng(42)
    order = Order(submission_time_ms=500)
    with pytest.raises(ValueError, match="prohibited"):
        apply_stochastic_latency(order, rng, min_delay_ms=0, max_delay_ms=50)
    with pytest.raises(ValueError, match="prohibited"):
        apply_stochastic_latency(order, rng, min_delay_ms=4, max_delay_ms=50)

