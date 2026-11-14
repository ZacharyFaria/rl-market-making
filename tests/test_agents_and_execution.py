"""Unit tests for PPO Market Maker, Double-DQN Execution Agent, and HJB/AC Baselines."""

from __future__ import annotations

import numpy as np

from src.agents.baselines import (
    AlmgrenChrissAgent,
    AvellanedaStoikovAgent,
    TWAPAgent,
    VWAPAgent,
)
from src.agents.dqn_agent import DoubleDQNExecutionAgent
from src.agents.ppo_agent import PPOMarketMaker
from src.envs.execution_env import OptimalExecutionEnv
from src.envs.market_making_env import MarketMakingEnv


def test_almgren_chriss_twap_vwap_schedules_sum_to_parent_order() -> None:
    """Verify AC, TWAP, and VWAP schedules conserve total parent shares Q_0."""
    q0 = 1000.0
    steps = 25
    ac = AlmgrenChrissAgent(parent_shares=q0, horizon_steps=steps).schedule()
    twap = TWAPAgent(parent_shares=q0, horizon_steps=steps).schedule()
    vwap = VWAPAgent(parent_shares=q0, horizon_steps=steps).schedule()

    assert np.isclose(ac.sum(), q0)
    assert np.isclose(twap.sum(), q0)
    assert np.isclose(vwap.sum(), q0)
    # Almgren-Chriss front-loads execution compared to uniform TWAP
    assert ac[0] > twap[0]


def test_ppo_market_maker_and_double_dqn_execution_smoke() -> None:
    """Verify PPO Market Maker and Double-DQN Execution agents train and act cleanly."""
    mm_env = MarketMakingEnv(regime="hawkes", n_steps=30, seed=42)
    ppo = PPOMarketMaker(seed=42)
    as_base = AvellanedaStoikovAgent()

    obs = mm_env.reset(seed=42)
    assert len(ppo.act(obs)) == 4
    assert len(as_base.act(obs)) == 4
    train_res = ppo.train_on_env(mm_env, episodes=5)
    assert train_res["episodes_trained"] == 5

    ex_env = OptimalExecutionEnv(parent_order_shares=1000.0, horizon_steps=15, seed=42)
    dqn = DoubleDQNExecutionAgent(parent_shares=1000.0, horizon_steps=15, seed=42)
    dqn_res = dqn.train_on_env(ex_env, episodes=5)
    assert dqn_res["episodes_trained"] == 5

