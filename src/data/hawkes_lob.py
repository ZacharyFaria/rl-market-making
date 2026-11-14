"""Mutually Exciting Bivariate Hawkes Process & L2 Limit Order Book Generator.

Simulates clustered buy/sell market order arrivals, adverse selection jumps, and
Level-2 queue volume dynamics for reinforcement learning training and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BivariateHawkesOrderFlow:
    """Mutually exciting Hawkes process driving bid and ask market order arrivals.

    .. math::
        \\lambda_t^+ = \\mu + \\alpha_{\\text{self}} I_t^+ + \\alpha_{\\text{cross}} I_t^-, \\quad
        \\lambda_t^- = \\mu + \\alpha_{\\text{self}} I_t^- + \\alpha_{\\text{cross}} I_t^+

    Attributes
    ----------
    mu : float
        Baseline exogenous arrival intensity (events per second).
    alpha_self : float
        Self-excitation jump magnitude for same-side market orders.
    alpha_cross : float
        Cross-excitation jump magnitude for opposite-side market orders.
    beta : float
        Exponential memory decay rate ($\\beta > \\alpha_{\\text{self}} + \\alpha_{\\text{cross}}$).
    """

    mu: float = 120.0
    alpha_self: float = 35.0
    alpha_cross: float = 10.0
    beta: float = 110.0

    def __post_init__(self) -> None:
        if self.alpha_self + self.alpha_cross >= self.beta:
            raise ValueError(
                "Hawkes stationarity requires alpha_self + alpha_cross < beta."
            )

    def step_intensities(
        self,
        exc_buy: float,
        exc_sell: float,
        dt: float,
        n_buy_events: int,
        n_sell_events: int,
    ) -> tuple[float, float, float, float]:
        """Advance exponential Hawkes excitation states over timestep `dt`."""
        decay = float(np.exp(-self.beta * dt))
        new_exc_buy = exc_buy * decay + self.alpha_self * n_buy_events + self.alpha_cross * n_sell_events
        new_exc_sell = exc_sell * decay + self.alpha_self * n_sell_events + self.alpha_cross * n_buy_events
        lam_buy = self.mu + new_exc_buy
        lam_sell = self.mu + new_exc_sell
        return lam_buy, lam_sell, new_exc_buy, new_exc_sell


def simulate_l2_episode(
    n_steps: int = 100,
    horizon_sec: float = 1.0,
    initial_mid: float = 100.0,
    sigma: float = 0.25,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a synthetic Level-2 LOB episode driven by mutually exciting Hawkes flow."""
    rng = np.random.default_rng(seed)
    dt = horizon_sec / max(n_steps, 1)
    hawkes = BivariateHawkesOrderFlow()

    mid_prices = np.zeros(n_steps + 1, dtype=float)
    imbalances = np.zeros(n_steps + 1, dtype=float)
    lam_buy_arr = np.zeros(n_steps + 1, dtype=float)
    lam_sell_arr = np.zeros(n_steps + 1, dtype=float)

    mid_prices[0] = initial_mid
    exc_b, exc_s = 0.0, 0.0
    lam_buy_arr[0] = hawkes.mu
    lam_sell_arr[0] = hawkes.mu

    for t in range(n_steps):
        lam_b = hawkes.mu + exc_b
        lam_s = hawkes.mu + exc_s
        n_b = int(rng.poisson(lam_b * dt))
        n_s = int(rng.poisson(lam_s * dt))
        lam_b_next, lam_s_next, exc_b, exc_s = hawkes.step_intensities(exc_b, exc_s, dt, n_b, n_s)

        flow_imb = (lam_b_next - lam_s_next) / max(lam_b_next + lam_s_next, 1e-6)
        imbalances[t + 1] = np.clip(flow_imb, -1.0, 1.0)
        lam_buy_arr[t + 1] = lam_b_next
        lam_sell_arr[t + 1] = lam_s_next

        # Adverse selection drift proportional to net Hawkes excitation imbalance
        drift = 0.15 * sigma * flow_imb * np.sqrt(dt)
        diffusion = sigma * np.sqrt(dt) * rng.standard_normal()
        mid_prices[t + 1] = mid_prices[t] + drift + diffusion

    return {
        "mid_prices": mid_prices,
        "imbalances": imbalances,
        "lam_buy": lam_buy_arr,
        "lam_sell": lam_sell_arr,
    }

