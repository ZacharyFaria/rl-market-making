"""Unit tests for REQ-SIM-002: Probabilistic L2 queue reconstruction with k in (-1, 0]."""

from __future__ import annotations

import pytest

from src.data.queue_reconstructor import estimate_queue_position


def test_req_sim_002_domain_validation_on_k() -> None:
    """REQ-SIM-002: Parameter k outside (-1, 0] must raise ValueError."""
    # Valid boundary k = 0.0 and interior k = -0.5
    assert estimate_queue_position(100.0, 0.0) > 0.0
    assert estimate_queue_position(100.0, -0.5) > 0.0
    assert estimate_queue_position(100.0, -0.999) > 0.0

    # Invalid k <= -1.0 or k > 0.0 must trigger ValueError
    with pytest.raises(ValueError, match="outside the valid mathematical domain"):
        estimate_queue_position(100.0, -1.0)
    with pytest.raises(ValueError, match="outside the valid mathematical domain"):
        estimate_queue_position(100.0, -1.5)
    with pytest.raises(ValueError, match="outside the valid mathematical domain"):
        estimate_queue_position(100.0, 0.05)


def test_req_sim_002_back_of_queue_cancellation_bias_monotonicity() -> None:
    """REQ-SIM-002: More negative k assigns higher cancellation likelihood to the back of queue."""
    # When more cancellations happen at the back of the queue ( behind our order ),
    # fewer cancellations deplete the volume *ahead* of our order.
    vol_ahead_uniform = estimate_queue_position(
        volume_ahead=80.0, k=0.0, cancelled_volume=20.0, total_level_volume=120.0
    )
    vol_ahead_moderate_bias = estimate_queue_position(
        volume_ahead=80.0, k=-0.4, cancelled_volume=20.0, total_level_volume=120.0
    )
    vol_ahead_strong_bias = estimate_queue_position(
        volume_ahead=80.0, k=-0.85, cancelled_volume=20.0, total_level_volume=120.0
    )

    assert vol_ahead_uniform < vol_ahead_moderate_bias < vol_ahead_strong_bias < 80.0

