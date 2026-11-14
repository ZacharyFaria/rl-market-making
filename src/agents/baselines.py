"""Closed-Form Stochastic Optimal Control Baselines (Avellaneda-Stoikov, Almgren-Chriss, TWAP, VWAP)."""

from __future__ import annotations

import math

import numpy as np

from src.envs.sanity_check import avellaneda_stoikov_quotes


class AvellanedaStoikovAgent:
    """Analytical Avellaneda & Stoikov (2008) HJB Market Making Policy."""

    def __init__(
        self,
        gamma: float = 0.10,
        sigma: float = 0.25,
        k: float = 1.50,
        order_size: float = 10.0,
    ) -> None:
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.k = float(k)
        self.order_size = float(order_size)

    def act(self, obs: np.ndarray) -> tuple[float, float, float, float]:
        """Return `(delta_bid, delta_ask, bid_size, ask_size)` from 6D observation vector."""
        inventory = float(obs[0])
        time_rem = float(obs[1])
        mid_price = float(obs[2])
        db, da, _ = avellaneda_stoikov_quotes(
            mid_price=mid_price,
            inventory=inventory,
            gamma=self.gamma,
            sigma=self.sigma,
            k=self.k,
            time_remaining=time_rem,
        )
        return db, da, self.order_size, self.order_size


class AlmgrenChrissAgent:
    """Analytical Almgren & Chriss (2001) Hyperbolic Sine Optimal Execution Schedule.

    .. math::
        q_t^* = Q_0 \\frac{\\sinh(\\kappa_{\\text{AC}} (T - t))}{\\sinh(\\kappa_{\\text{AC}} T)}, \\qquad
        \\kappa_{\\text{AC}} = \\sqrt{\\frac{\\lambda \\sigma^2}{\\eta}}
    """

    def __init__(
        self,
        parent_shares: float = 1000.0,
        horizon_steps: int = 25,
        sigma: float = 0.30,
        temp_impact_eta: float = 0.00025,
        risk_aversion_lambda: float = 0.0015,
    ) -> None:
        self.parent_shares = float(parent_shares)
        self.horizon_steps = int(horizon_steps)
        self.kappa_ac = math.sqrt(
            max(risk_aversion_lambda * (sigma**2) / max(temp_impact_eta, 1e-8), 1e-6)
        )

    def schedule(self) -> np.ndarray:
        """Compute exact deterministic child order slice schedule across `horizon_steps`."""
        t_grid = np.linspace(0.0, 1.0, self.horizon_steps + 1)
        denom = math.sinh(self.kappa_ac)
        if denom < 1e-8:
            rem = self.parent_shares * (1.0 - t_grid)
        else:
            rem = self.parent_shares * np.sinh(self.kappa_ac * (1.0 - t_grid)) / denom
        slices = -np.diff(rem)
        return np.maximum(slices, 0.0)


class TWAPAgent:
    """Time-Weighted Average Price (TWAP) uniform execution baseline."""

    def __init__(self, parent_shares: float = 1000.0, horizon_steps: int = 25) -> None:
        self.parent_shares = float(parent_shares)
        self.horizon_steps = int(horizon_steps)

    def schedule(self) -> np.ndarray:
        """Return uniform child slices $Q_0 / N$."""
        return np.full(self.horizon_steps, self.parent_shares / max(self.horizon_steps, 1))


class VWAPAgent:
    """Volume-Weighted Average Price (VWAP) U-shaped intraday volume profile baseline."""

    def __init__(self, parent_shares: float = 1000.0, horizon_steps: int = 25) -> None:
        self.parent_shares = float(parent_shares)
        self.horizon_steps = int(horizon_steps)

    def schedule(self) -> np.ndarray:
        """Return U-shaped child order schedule matching diurnal market volume."""
        u = np.linspace(-1.0, 1.0, self.horizon_steps)
        weights = 1.0 + 0.65 * (u**2)
        weights /= weights.sum()
        return self.parent_shares * weights

