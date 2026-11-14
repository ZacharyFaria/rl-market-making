"""Stochastic Latency Matching Engine for Limit Order Book Simulation.

Implements REQ-SIM-001: Injects a uniformly distributed execution delay strictly
between 5 ms and 50 ms to every limit order submission during Hawkes simulation,
eliminating deterministic zero-latency reward hacking loopholes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Order:
    """Limit or child order submitted by the RL agent to the matching engine.

    Attributes
    ----------
    order_id : str
        Unique order identifier.
    side : str
        `"BUY"` / `"BID"` or `"SELL"` / `"ASK"`.
    price : float
        Limit quote price.
    size : float
        Order quantity in shares.
    submission_time_ms : int
        Discrete simulation timestamp (in ms) when the agent emitted the order.
    insertion_time_ms : int
        Timestamp (in ms) when the order enters the LOB queue after latency delay.
    latency_ms : int
        Applied stochastic execution delay in milliseconds (`5 <= latency_ms <= 50`).
    """

    order_id: str = "ORD-001"
    side: str = "BUY"
    price: float = 100.0
    size: float = 10.0
    submission_time_ms: int = 0
    insertion_time_ms: int = 0
    latency_ms: int = 0


def apply_stochastic_latency(
    order: Order,
    rng: np.random.Generator,
    min_delay_ms: int = 5,
    max_delay_ms: int = 50,
) -> int:
    """Inject a uniformly distributed execution delay in $[5, 50]\\text{ ms}$ (REQ-SIM-001).

    Parameters
    ----------
    order : Order
        Limit order submitted by the agent.
    rng : np.random.Generator
        Seeded NumPy random number generator.
    min_delay_ms : int, default=5
        Minimum allowed execution delay in milliseconds (must be `>= 5`).
    max_delay_ms : int, default=50
        Maximum allowed execution delay in milliseconds (must be `<= 50` and `>= min_delay_ms`).

    Returns
    -------
    int
        Sampled execution delay in milliseconds (`5 <= delay <= 50`).
    """
    if min_delay_ms < 5:
        raise ValueError(
            f"Zero/sub-5ms latency is prohibited in Hawkes simulation (got min_delay_ms={min_delay_ms}, REQ-SIM-001)."
        )
    if max_delay_ms < min_delay_ms or max_delay_ms > 50:
        raise ValueError(
            f"max_delay_ms must satisfy {min_delay_ms} <= max_delay_ms <= 50 (got {max_delay_ms})."
        )

    delay_ms = int(rng.integers(min_delay_ms, max_delay_ms + 1))
    order.latency_ms = delay_ms
    order.insertion_time_ms = int(order.submission_time_ms) + delay_ms
    return delay_ms

