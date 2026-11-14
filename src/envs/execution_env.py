"""Gymnasium-compatible Optimal Execution Environment (Almgren-Chriss + Hawkes LOB).

Simulates liquidating a parent institutional block order of $Q_0$ shares over $N$ steps
under temporary market impact $\\eta v_t$, permanent market impact $\\kappa v_t$,
Hawkes order flow toxicity, and 5-50 ms stochastic latency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.envs.matching_engine import Order, apply_stochastic_latency


class OptimalExecutionEnv:
    """Single-Asset Optimal Execution Environment minimizing Implementation Shortfall (IS).

    State vector (`shape=(4,)`, `dtype=float32`):
      - `state[0]`: Remaining inventory fraction $q_t / Q_0 \\in [0, 1]$
      - `state[1]`: Remaining time fraction $(N - t) / N \\in [0, 1]$
      - `state[2]`: Current mid-price relative to arrival price $(S_t - S_0) / S_0$
      - `state[3]`: Local order book flow imbalance $I_t \\in [-1, 1]$
    """

    def __init__(
        self,
        parent_order_shares: float = 1000.0,
        horizon_steps: int = 25,
        initial_mid: float = 100.0,
        sigma: float = 0.30,
        temp_impact_eta: float = 0.00025,
        perm_impact_kappa: float = 0.00008,
        risk_aversion_lambda: float = 0.0015,
        seed: int = 42,
    ) -> None:
        self.parent_order_shares = float(parent_order_shares)
        self.horizon_steps = int(horizon_steps)
        self.initial_mid = float(initial_mid)
        self.sigma = float(sigma)
        self.temp_impact_eta = float(temp_impact_eta)
        self.perm_impact_kappa = float(perm_impact_kappa)
        self.risk_aversion_lambda = float(risk_aversion_lambda)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.remaining_shares: float = self.parent_order_shares
        self.mid_price: float = self.initial_mid
        self.step_idx: int = 0
        self.cash_proceeds: float = 0.0
        self.imbalance: float = 0.0
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset execution episode to $t = 0$ with $q_0 = Q_0$."""
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
        self.remaining_shares = self.parent_order_shares
        self.mid_price = self.initial_mid
        self.step_idx = 0
        self.cash_proceeds = 0.0
        self.imbalance = 0.0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.remaining_shares / max(self.parent_order_shares, 1.0),
                (self.horizon_steps - self.step_idx) / max(self.horizon_steps, 1),
                (self.mid_price - self.initial_mid) / max(self.initial_mid, 1.0),
                self.imbalance,
            ],
            dtype=np.float32,
        )

    def step(self, child_shares: float) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute child slice `child_shares` and compute step Implementation Shortfall reward."""
        self.step_idx += 1
        done = bool(self.step_idx >= self.horizon_steps or self.remaining_shares <= 1e-6)

        if self.step_idx >= self.horizon_steps:
            exec_shares = max(0.0, self.remaining_shares)
        else:
            exec_shares = float(np.clip(child_shares, 0.0, self.remaining_shares))

        order = Order(
            order_id=f"EXEC-{self.step_idx}",
            side="SELL",
            price=self.mid_price,
            size=exec_shares,
            submission_time_ms=self.step_idx * 100,
        )
        delay_ms = apply_stochastic_latency(order, self.rng)

        # Temporary impact widens when selling into negative order book imbalance
        liquidity_multiplier = 1.0 - 0.35 * self.imbalance + 0.002 * delay_ms
        temp_cost_per_share = self.temp_impact_eta * exec_shares * max(0.4, liquidity_multiplier)
        exec_price = self.mid_price - temp_cost_per_share
        self.cash_proceeds += exec_shares * exec_price
        self.remaining_shares = max(0.0, self.remaining_shares - exec_shares)

        # Permanent impact + order flow drift + diffusion
        perm_drop = self.perm_impact_kappa * exec_shares
        self.imbalance = float(0.65 * self.imbalance + self.rng.normal(0.0, 0.35))
        self.imbalance = float(np.clip(self.imbalance, -1.0, 1.0))
        dt = 1.0 / max(self.horizon_steps, 1)
        self.mid_price = (
            self.mid_price
            - perm_drop
            + 0.12 * self.sigma * np.sqrt(dt) * self.imbalance
            + self.sigma * np.sqrt(dt) * float(self.rng.standard_normal())
        )

        # Step shortfall cost + running Almgren-Chriss inventory variance penalty
        step_shortfall = exec_shares * (self.initial_mid - exec_price)
        inv_risk_cost = (
            self.risk_aversion_lambda * (self.sigma**2) * (self.remaining_shares**2) * dt * 1e-3
        )
        reward = -float(step_shortfall + inv_risk_cost)

        arrival_notional = self.parent_order_shares * self.initial_mid
        is_bps = (
            ((arrival_notional - self.cash_proceeds) / max(arrival_notional, 1.0)) * 10000.0
            if done
            else 0.0
        )

        return (
            self._get_obs(),
            reward,
            done,
            {
                "executed_shares": exec_shares,
                "remaining_shares": self.remaining_shares,
                "latency_ms": delay_ms,
                "implementation_shortfall_bps": float(is_bps),
            },
        )

