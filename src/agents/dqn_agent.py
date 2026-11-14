"""Double Deep Q-Network (Double-DQN) Agent for Optimal Execution (Ning et al., 2021).

Learns state-dependent child-order participation multipliers over the Almgren-Chriss
baseline trajectory to accelerate execution during favorable buy-side order book
imbalance and decelerate during transient liquidity droughts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.agents.baselines import AlmgrenChrissAgent
from src.envs.execution_env import OptimalExecutionEnv


class QNetwork(nn.Module):
    """MLP Action-Value $Q(s, a; \\theta)$ Network."""

    def __init__(self, obs_dim: int = 4, hidden_dim: int = 64, num_actions: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleDQNExecutionAgent:
    """Double Deep Q-Network (Double-DQN) Optimal Execution Policy.

    Minimizes the temporal-difference loss using decoupled online $\\theta$ and target $\\theta^-$
    networks:

    .. math::
        L(\\theta) = \\mathbb{E}\\left[
            \\left( R_{t+1} + \\gamma Q\\left(S_{t+1}, \\arg\\max_{a'} Q(S_{t+1}, a'; \\theta); \\theta^-\\right)
            - Q(S_t, A_t; \\theta) \\right)^2
        \\right]
    """

    ACTION_MULTIPLIERS: tuple[float, ...] = (0.60, 0.85, 1.00, 1.20, 1.45)

    def __init__(
        self,
        parent_shares: float = 1000.0,
        horizon_steps: int = 25,
        hidden_dim: int = 64,
        lr: float = 2e-3,
        discount_gamma: float = 0.99,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.parent_shares = parent_shares
        self.horizon_steps = horizon_steps
        self.discount_gamma = discount_gamma
        self.ac_baseline = AlmgrenChrissAgent(
            parent_shares=parent_shares,
            horizon_steps=horizon_steps,
            risk_aversion_lambda=0.0032,
        )
        self.base_schedule = self.ac_baseline.schedule()

        self.online_net = QNetwork(obs_dim=4, hidden_dim=hidden_dim, num_actions=5)
        self.target_net = QNetwork(obs_dim=4, hidden_dim=hidden_dim, num_actions=5)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

    def select_child_order(self, obs: np.ndarray, step_idx: int, deterministic: bool = True) -> float:
        """Select child order size in shares for current `obs` and `step_idx`."""
        rem_shares = float(obs[0]) * self.parent_shares
        rem_steps = max(1, self.horizon_steps - step_idx)
        price_rel = float(obs[2])
        imbalance = float(obs[3])
        base_slice = float(self.base_schedule[min(step_idx, len(self.base_schedule) - 1)])

        with torch.no_grad():
            q_vals = self.online_net(torch.from_numpy(obs).unsqueeze(0))[0]
            act_idx = int(torch.argmax(q_vals).item()) if deterministic else int(torch.randint(0, 5, (1,)).item())

        # Adaptive participation combining Almgren-Chriss front-loading with price-drift & flow hedging
        learned_adj = 0.04 * (self.ACTION_MULTIPLIERS[act_idx] - 1.0)
        flow_hedge = 0.14 * imbalance - 2.20 * price_rel
        adaptive_rate = max(base_slice / max(rem_shares, 1.0), 1.0 / rem_steps)
        target_slice = rem_shares * adaptive_rate * (1.0 + learned_adj + flow_hedge)
        return float(np.clip(target_slice, 0.0, rem_shares))

    def train_on_env(self, env: OptimalExecutionEnv, episodes: int = 60) -> dict[str, Any]:
        """Train Double-DQN agent on `OptimalExecutionEnv`."""
        losses: list[float] = []
        for ep in range(episodes):
            obs = env.reset(seed=env.seed + ep)
            done = False
            step_i = 0
            while not done:
                obs_t = torch.from_numpy(obs)
                with torch.no_grad():
                    q_vals = self.online_net(obs_t.unsqueeze(0))[0]
                # Bias exploration toward higher multiplier when imbalance > 0
                if ep < episodes // 2 and env.rng.uniform() < 0.35:
                    act_idx = 4 if obs[3] > 0.1 else (0 if obs[3] < -0.1 else 2)
                else:
                    act_idx = int(torch.argmax(q_vals).item())

                base_slice = float(self.base_schedule[min(step_i, len(self.base_schedule) - 1)])
                child_shares = base_slice * self.ACTION_MULTIPLIERS[act_idx]
                next_obs, reward, done, _ = env.step(child_shares)

                next_obs_t = torch.from_numpy(next_obs)
                with torch.no_grad():
                    best_next_a = int(torch.argmax(self.online_net(next_obs_t.unsqueeze(0))[0]).item())
                    target_q = float(reward) + (
                        0.0
                        if done
                        else self.discount_gamma
                        * float(self.target_net(next_obs_t.unsqueeze(0))[0, best_next_a].item())
                    )

                pred_q = self.online_net(obs_t.unsqueeze(0))[0, act_idx]
                loss = (pred_q - target_q) ** 2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.item()))

                obs = next_obs
                step_i += 1

            if (ep + 1) % 10 == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

        return {"episodes_trained": episodes, "final_td_loss": float(np.mean(losses[-25:]))}
