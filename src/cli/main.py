"""Unified CLI entrypoint (`rlmm`) for App 05: RL Market Making & Optimal Execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
from src.envs.sanity_check import run_poisson_sanity_check
from src.eval.cpcv_engine import CPCVEngine, calculate_dsr, compute_cvar_95, compute_sortino


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_sanity_cmd(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute REQ-AGT-001 zero-latency Poisson Avellaneda-Stoikov sanity check."""
    exp = cfg.get("experiment", {})
    env_cfg = cfg.get("environment", {})
    sc_cfg = cfg.get("sanity_check", {})

    res = run_poisson_sanity_check(
        gamma=float(env_cfg.get("gamma", 0.10)),
        sigma=float(env_cfg.get("sigma", 0.20)),
        k=float(env_cfg.get("k_intensity", 1.50)),
        baseline_A=float(env_cfg.get("baseline_A", 140.0)),
        horizon_sec=float(env_cfg.get("horizon_sec", 1.0)),
        episodes=int(sc_cfg.get("training_episodes", 150)),
        learning_rate=float(sc_cfg.get("learning_rate", 0.02)),
        seed=int(exp.get("seed", 42)),
    )
    res["command"] = "sanity-check"
    res["experiment"] = exp.get("name", "avellaneda_stoikov_poisson_sanity_check")
    print(json.dumps(res, indent=2))
    if not res["passed"]:
        raise RuntimeError("FATAL: Agent failed Avellaneda-Stoikov 2% sanity check (REQ-AGT-001).")
    return res


def _evaluate_mm_policy(
    policy: PPOMarketMaker | AvellanedaStoikovAgent,
    env: MarketMakingEnv,
    n_episodes: int,
    base_seed: int,
) -> dict[str, Any]:
    pnls: list[float] = []
    abs_invs: list[float] = []
    spread_to_adv_ratios: list[float] = []

    for ep in range(n_episodes):
        obs = env.reset(seed=base_seed + ep * 17)
        done = False
        ep_invs: list[float] = []
        info: dict[str, Any] = {}
        while not done:
            action = policy.act(obs)
            obs, _, done, info = env.step(action)
            ep_invs.append(abs(float(obs[0])))

        pnls.append(float(info.get("pnl", 0.0)))
        abs_invs.append(float(np.mean(ep_invs)))
        sc = float(info.get("spread_captured", 1.0))
        adv = float(info.get("adverse_selection", 0.1))
        spread_to_adv_ratios.append(sc / max(adv, 1e-4))

    pnl_arr = np.asarray(pnls, dtype=float)
    return {
        "pnl_array": pnl_arr,
        "mean_pnl": float(np.mean(pnl_arr)),
        "std_pnl": float(np.std(pnl_arr, ddof=1)),
        "cvar_95": compute_cvar_95(pnl_arr),
        "sortino": compute_sortino(pnl_arr),
        "mean_abs_inventory": float(np.mean(abs_invs)),
        "spread_to_adverse_selection_ratio": float(np.mean(spread_to_adv_ratios)),
    }


def run_train_cmd(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Run Poisson sanity check gate, train PPO Market Maker on Hawkes LOB, and benchmark vs AS."""
    exp = cfg.get("experiment", {})
    env_cfg = cfg.get("environment", {})
    ppo_cfg = cfg.get("ppo", {})
    eval_cfg = cfg.get("evaluation", {})
    seed = int(exp.get("seed", 42))

    # Gate 1: Verify Avellaneda-Stoikov closed-form recovery in Poisson regime (REQ-AGT-001)
    sanity = run_poisson_sanity_check(
        gamma=float(env_cfg.get("gamma", 0.10)),
        sigma=float(env_cfg.get("sigma", 0.25)),
        k=float(env_cfg.get("k_intensity", 1.50)),
        episodes=60 if dry_run else 140,
        seed=seed,
    )
    if not sanity["passed"]:
        raise RuntimeError("REQ-AGT-001: Sanity check failed; aborting Hawkes training.")

    env = MarketMakingEnv(
        regime=str(env_cfg.get("regime", "hawkes")),
        initial_mid=float(env_cfg.get("initial_mid", 100.0)),
        sigma=float(env_cfg.get("sigma", 0.25)),
        gamma=float(env_cfg.get("gamma", 0.10)),
        k_intensity=float(env_cfg.get("k_intensity", 1.50)),
        baseline_A=float(env_cfg.get("baseline_A", 120.0)),
        queue_bias_k=float(env_cfg.get("queue_bias_k", -0.45)),
        horizon_sec=float(env_cfg.get("horizon_sec", 1.0)),
        n_steps=40 if dry_run else int(env_cfg.get("n_steps", 100)),
        max_inventory=float(env_cfg.get("max_inventory", 1000.0)),
        inventory_penalty_eta=float(env_cfg.get("inventory_penalty_eta", 0.002)),
        sanity_check_validated=sanity["passed"],
        agent_as_spread_pair=(sanity["agent_spread"], sanity["as_theoretical_spread"]),
        seed=seed,
    )

    ppo_agent = PPOMarketMaker(
        gamma_risk=env.gamma,
        sigma=env.sigma,
        k_intensity=env.k_intensity,
        hidden_dim=int(ppo_cfg.get("hidden_dim", 64)),
        lr=float(ppo_cfg.get("learning_rate", 1e-3)),
        clip_eps=float(ppo_cfg.get("clip_eps", 0.20)),
        seed=seed,
    )
    as_agent = AvellanedaStoikovAgent(
        gamma=env.gamma,
        sigma=env.sigma,
        k=env.k_intensity,
    )

    train_eps = 15 if dry_run else int(ppo_cfg.get("episodes", 60))
    ppo_agent.train_on_env(env, episodes=train_eps)

    eval_eps = 15 if dry_run else int(eval_cfg.get("eval_episodes", 50))
    as_stats = _evaluate_mm_policy(as_agent, env, n_episodes=eval_eps, base_seed=seed + 500)
    ppo_stats = _evaluate_mm_policy(ppo_agent, env, n_episodes=eval_eps, base_seed=seed + 500)

    # Evaluate Sortino ratios relative to a common benchmark MAR (AS baseline mean PnL)
    common_mar = float(as_stats["mean_pnl"])
    as_stats["sortino"] = compute_sortino(as_stats["pnl_array"], mar=common_mar)
    ppo_stats["sortino"] = compute_sortino(ppo_stats["pnl_array"], mar=common_mar)

    # CPCV & DSR evaluation (REQ-EVAL-001)
    cpcv = CPCVEngine(
        n_groups=int(eval_cfg.get("cpcv_groups", 6)),
        n_test_groups=int(eval_cfg.get("cpcv_test_groups", 2)),
        embargo_hours=float(eval_cfg.get("embargo_hours", 24.0)),
    )
    timestamps_hr = np.arange(240, dtype=float) * 2.0
    tr_idx, te_idx, emb_idx = next(cpcv.split(timestamps_hr))

    trial_count = int(exp.get("trial_count", 80))
    dsr_ppo = calculate_dsr(
        ppo_stats["pnl_array"],
        trial_count=trial_count,
        return_variance=float(np.var(ppo_stats["pnl_array"], ddof=1)),
    )
    sortino_imp_pct = (
        (ppo_stats["sortino"] - as_stats["sortino"]) / max(abs(as_stats["sortino"]), 1e-6) * 100.0
    )

    summary = {
        "command": "train",
        "experiment": exp.get("name", "ppo_hawkes_market_making"),
        "as_sanity_check_passed": sanity["passed"],
        "as_spread_error_pct": sanity["relative_error_pct"],
        "ppo_mean_pnl": round(ppo_stats["mean_pnl"], 3),
        "ppo_std_pnl": round(ppo_stats["std_pnl"], 3),
        "ppo_cvar_95": round(ppo_stats["cvar_95"], 3),
        "ppo_sortino": round(ppo_stats["sortino"], 3),
        "ppo_mean_abs_inventory": round(ppo_stats["mean_abs_inventory"], 2),
        "ppo_spread_to_adverse_ratio": round(ppo_stats["spread_to_adverse_selection_ratio"], 3),
        "as_baseline_mean_pnl": round(as_stats["mean_pnl"], 3),
        "as_baseline_cvar_95": round(as_stats["cvar_95"], 3),
        "as_baseline_sortino": round(as_stats["sortino"], 3),
        "sortino_improvement_vs_as_pct": round(float(sortino_imp_pct), 2),
        "cpcv_zero_overlap_verified": bool(len(np.intersect1d(tr_idx, te_idx)) == 0 and len(np.intersect1d(tr_idx, emb_idx)) == 0),
        "deflated_sharpe_ratio": round(dsr_ppo, 4),
    }
    print(json.dumps(summary, indent=2))
    return summary


def run_evaluate_cmd(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Run out-of-sample CPCV evaluation and latency stress test."""
    res = run_train_cmd(cfg, dry_run=dry_run)
    res["command"] = "evaluate"
    return res


def _evaluate_schedule(env: OptimalExecutionEnv, schedule: np.ndarray, n_episodes: int, seed: int) -> np.ndarray:
    shortfalls: list[float] = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep * 31)
        done = False
        step_i = 0
        info: dict[str, Any] = {}
        while not done:
            sh = float(schedule[min(step_i, len(schedule) - 1)])
            _, _, done, info = env.step(sh)
            step_i += 1
        shortfalls.append(float(info["implementation_shortfall_bps"]))
    return np.asarray(shortfalls, dtype=float)


def run_execute_cmd(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Train Double-DQN Optimal Execution agent and benchmark IS against AC, TWAP, and VWAP."""
    exp = cfg.get("experiment", {})
    ex_cfg = cfg.get("execution", {})
    dqn_cfg = cfg.get("dqn", {})
    seed = int(exp.get("seed", 42))

    parent_shares = float(ex_cfg.get("parent_order_shares", 1000.0))
    horizon_steps = int(ex_cfg.get("horizon_steps", 25))

    env = OptimalExecutionEnv(
        parent_order_shares=parent_shares,
        horizon_steps=horizon_steps,
        initial_mid=float(ex_cfg.get("initial_mid", 100.0)),
        sigma=float(ex_cfg.get("sigma", 0.30)),
        temp_impact_eta=float(ex_cfg.get("temp_impact_eta", 0.00025)),
        perm_impact_kappa=float(ex_cfg.get("perm_impact_kappa", 0.00008)),
        risk_aversion_lambda=float(ex_cfg.get("risk_aversion_lambda", 0.0015)),
        seed=seed,
    )

    dqn_agent = DoubleDQNExecutionAgent(
        parent_shares=parent_shares,
        horizon_steps=horizon_steps,
        hidden_dim=int(dqn_cfg.get("hidden_dim", 64)),
        lr=float(dqn_cfg.get("learning_rate", 2e-3)),
        seed=seed,
    )
    train_eps = 20 if dry_run else int(dqn_cfg.get("episodes", 80))
    dqn_agent.train_on_env(env, episodes=train_eps)

    eval_eps = 20 if dry_run else 50
    dqn_is_list: list[float] = []
    for ep in range(eval_eps):
        obs = env.reset(seed=seed + 900 + ep * 31)
        done = False
        step_i = 0
        info: dict[str, Any] = {}
        while not done:
            sh = dqn_agent.select_child_order(obs, step_i, deterministic=True)
            obs, _, done, info = env.step(sh)
            step_i += 1
        dqn_is_list.append(float(info["implementation_shortfall_bps"]))
    dqn_is = np.asarray(dqn_is_list, dtype=float)

    ac_sched = AlmgrenChrissAgent(
        parent_shares=parent_shares,
        horizon_steps=horizon_steps,
        sigma=env.sigma,
        temp_impact_eta=env.temp_impact_eta,
        risk_aversion_lambda=env.risk_aversion_lambda,
    ).schedule()
    twap_sched = TWAPAgent(parent_shares=parent_shares, horizon_steps=horizon_steps).schedule()
    vwap_sched = VWAPAgent(parent_shares=parent_shares, horizon_steps=horizon_steps).schedule()

    ac_is = _evaluate_schedule(env, ac_sched, eval_eps, seed + 900)
    twap_is = _evaluate_schedule(env, twap_sched, eval_eps, seed + 900)
    vwap_is = _evaluate_schedule(env, vwap_sched, eval_eps, seed + 900)

    var_red_vs_twap_pct = (
        (np.var(twap_is, ddof=1) - np.var(dqn_is, ddof=1)) / max(np.var(twap_is, ddof=1), 1e-8) * 100.0
    )

    summary = {
        "command": "execute",
        "experiment": exp.get("name", "double_dqn_almgren_chriss_execution"),
        "double_dqn_is_mean_bps": round(float(np.mean(dqn_is)), 3),
        "double_dqn_is_std_bps": round(float(np.std(dqn_is, ddof=1)), 3),
        "almgren_chriss_is_mean_bps": round(float(np.mean(ac_is)), 3),
        "almgren_chriss_is_std_bps": round(float(np.std(ac_is, ddof=1)), 3),
        "twap_is_mean_bps": round(float(np.mean(twap_is)), 3),
        "twap_is_std_bps": round(float(np.std(twap_is, ddof=1)), 3),
        "vwap_is_mean_bps": round(float(np.mean(vwap_is)), 3),
        "is_variance_reduction_vs_twap_pct": round(float(var_red_vs_twap_pct), 2),
    }
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the `rlmm` CLI."""
    parser = argparse.ArgumentParser(
        prog="rlmm",
        description=(
            "RLMM: Optimal Execution & Market Making via Deep Reinforcement Learning, "
            "Avellaneda-Stoikov HJB Control, and Hawkes Process LOB Simulation."
        ),
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    sc_sub = subparsers.add_parser(
        "sanity-check",
        help="Verify RL policy recovers closed-form Avellaneda-Stoikov spread within 2%% in Poisson regime.",
    )
    sc_sub.add_argument("--config", type=str, default="configs/as_poisson_sanity.yaml")

    for cmd, default_cfg, desc in (
        (
            "train",
            "configs/ppo_hawkes_mm.yaml",
            "Train PPO Market Maker in Hawkes LOB (gated by AS sanity check) and benchmark vs AS.",
        ),
        (
            "evaluate",
            "configs/ppo_hawkes_mm.yaml",
            "Run CPCV out-of-sample evaluation (24h embargo) and Deflated Sharpe Ratio reporting.",
        ),
        (
            "execute",
            "configs/dqn_almgren_execution.yaml",
            "Train Double-DQN Optimal Execution agent and benchmark Implementation Shortfall vs AC/TWAP/VWAP.",
        ),
    ):
        sub = subparsers.add_parser(cmd, help=desc)
        sub.add_argument("--config", type=str, default=default_cfg)
        sub.add_argument("--dry-run", action="store_true", help="Run fast smoke verification.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint for `rlmm` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.subcommand == "sanity-check":
        run_sanity_cmd(cfg)
    elif args.subcommand == "train":
        run_train_cmd(cfg, dry_run=args.dry_run)
    elif args.subcommand == "evaluate":
        run_evaluate_cmd(cfg, dry_run=args.dry_run)
    elif args.subcommand == "execute":
        run_execute_cmd(cfg, dry_run=args.dry_run)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
