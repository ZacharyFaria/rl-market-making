"""Proximal Policy Optimization (PPO) Market Making Agent with Clipped Surrogate Loss.

Learns an adaptive residual quote-skewing policy on top of the Avellaneda-Stoikov HJB
reservation price, widening ask quotes during positive Hawkes buy-flow cascades and
widening bid quotes during negative Hawkes sell-flow cascades to mitigate adverse selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.envs.market_making_env import MarketMakingEnv
from src.envs.sanity_check import avellaneda_stoikov_quotes


class ActorCriticNet(nn.Module):
    """Two-headed Gaussian Policy Actor and State-Value Critic Network."""

    def __init__(self, obs_dim: int = 6, hidden_dim: int = 64, action_dim: int = 2) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -1.2))
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize residual output near zero so untrained policy starts near AS baseline
        nn.init.uniform_(self.actor_mean.weight, -0.02, 0.02)
        nn.init.zeros_(self.actor_mean.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(action_mean, action_std, value)`."""
        h = self.shared(obs)
        mean = 0.5 * torch.tanh(self.actor_mean(h))
        std = torch.exp(self.actor_log_std).expand_as(mean)
        val = self.critic(h).squeeze(-1)
        return mean, std, val


class PPOMarketMaker:
    """Proximal Policy Optimization (PPO) agent for Hawkes LOB Market Making.

    Optimizes the clipped surrogate objective:

    .. math::
        L^{\\text{CLIP}}(\\theta) = \\mathbb{E}\\left[
            \\min\\left( r_t(\\theta) \\hat{A}_t, \\; \\text{clip}(r_t(\\theta), 1 - \\epsilon, 1 + \\epsilon) \\hat{A}_t \\right)
        \\right]
    """

    def __init__(
        self,
        gamma_risk: float = 0.10,
        sigma: float = 0.25,
        k_intensity: float = 1.50,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        clip_eps: float = 0.20,
        gae_lambda: float = 0.95,
        discount_gamma: float = 0.99,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.gamma_risk = gamma_risk
        self.sigma = sigma
        self.k_intensity = k_intensity
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.discount_gamma = discount_gamma

        self.net = ActorCriticNet(obs_dim=6, hidden_dim=hidden_dim, action_dim=2)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _normalize_obs(self, obs: np.ndarray) -> torch.Tensor:
        normed = np.array(
            [
                obs[0] / 100.0,
                obs[1],
                (obs[2] - 100.0) / 5.0,
                obs[3] * 10.0,
                obs[4],
                (obs[5] - 240.0) / 100.0,
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(normed)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> tuple[float, float, float, float]:
        """Compute quote offsets `(delta_bid, delta_ask, bid_size, ask_size)`."""
        inventory, time_rem, mid_price = float(obs[0]), float(obs[1]), float(obs[2])
        imbalance, total_intensity = float(obs[4]), float(obs[5])

        base_db, base_da, _ = avellaneda_stoikov_quotes(
            mid_price=mid_price,
            inventory=inventory,
            gamma=self.gamma_risk,
            sigma=self.sigma,
            k=self.k_intensity,
            time_remaining=time_rem,
        )

        with torch.no_grad():
            obs_t = self._normalize_obs(obs).unsqueeze(0)
            mean, std, _ = self.net(obs_t)
            if deterministic:
                res = mean[0].numpy()
            else:
                dist = torch.distributions.Normal(mean[0], std[0])
                res = dist.sample().numpy()

        # Combine learned neural residual + microstructure adverse-selection shield
        intensity_surge = max(0.0, (total_intensity - 240.0) / 240.0)
        tox_bid_widen = max(0.0, -imbalance) * 0.35 + 0.12 * intensity_surge
        tox_ask_widen = max(0.0, imbalance) * 0.35 + 0.12 * intensity_surge
        inv_skew = 0.004 * inventory

        delta_b = max(0.02, float(base_db + tox_bid_widen + inv_skew + 0.15 * res[0]))
        delta_a = max(0.02, float(base_da + tox_ask_widen - inv_skew + 0.15 * res[1]))
        return delta_b, delta_a, 10.0, 10.0

    def train_on_env(self, env: MarketMakingEnv, episodes: int = 60) -> dict[str, Any]:
        """Train PPO Actor-Critic network via GAE rollouts and clipped surrogate updates."""
        episode_rewards: list[float] = []

        for ep in range(episodes):
            obs = env.reset(seed=env.seed + ep)
            obs_list: list[torch.Tensor] = []
            act_list: list[torch.Tensor] = []
            logp_list: list[torch.Tensor] = []
            val_list: list[torch.Tensor] = []
            rew_list: list[float] = []

            done = False
            while not done:
                obs_t = self._normalize_obs(obs)
                mean, std, val = self.net(obs_t.unsqueeze(0))
                dist = torch.distributions.Normal(mean[0], std[0])
                raw_a = dist.sample()
                logp = dist.log_prob(raw_a).sum()

                db, da, bs, as_sz = self.act(obs, deterministic=False)
                next_obs, reward, done, _ = env.step((db, da, bs, as_sz))

                obs_list.append(obs_t)
                act_list.append(raw_a.detach())
                logp_list.append(logp.detach())
                val_list.append(val[0].detach())
                rew_list.append(float(reward))
                obs = next_obs

            episode_rewards.append(float(np.sum(rew_list)))

            # Compute Generalized Advantage Estimation (GAE)
            T = len(rew_list)
            advantages = torch.zeros(T, dtype=torch.float32)
            returns = torch.zeros(T, dtype=torch.float32)
            vals = torch.stack(val_list)
            gae = 0.0
            for t in reversed(range(T)):
                next_val = 0.0 if t == T - 1 else float(vals[t + 1].item())
                delta = rew_list[t] + self.discount_gamma * next_val - float(vals[t].item())
                gae = delta + self.discount_gamma * self.gae_lambda * gae
                advantages[t] = gae
                returns[t] = advantages[t] + vals[t]

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_batch = torch.stack(obs_list)
            act_batch = torch.stack(act_list)
            old_logp_batch = torch.stack(logp_list)

            # PPO Clipped Surrogate update
            mean_b, std_b, val_b = self.net(obs_batch)
            dist_b = torch.distributions.Normal(mean_b, std_b)
            new_logp = dist_b.log_prob(act_batch).sum(dim=-1)
            ratio = torch.exp(new_logp - old_logp_batch)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((val_b - returns) ** 2).mean()
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

        return {
            "episodes_trained": episodes,
            "mean_episode_reward": float(np.mean(episode_rewards[-20:])),
        }

