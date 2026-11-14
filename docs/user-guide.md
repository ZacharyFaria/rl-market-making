# User Guide & CLI Operational Manual (`rlmm`)

**Application:** `05-rl-market-making`  
**CLI Command:** `rlmm`  

---

## 1. Environment Setup

Activate the shared virtual environment from the repository root (`05-rl-market-making`):

```bash
source .venv/bin/activate
rlmm --help
```

```text
usage: rlmm [-h] {sanity-check,train,evaluate,execute} ...

RLMM: Optimal Execution & Market Making via Deep Reinforcement Learning,
Avellaneda-Stoikov HJB Control, and Hawkes Process LOB Simulation.

positional arguments:
  {sanity-check,train,evaluate,execute}
    sanity-check        Verify RL policy recovers closed-form Avellaneda-
                        Stoikov spread within 2% in Poisson regime.
    train               Train PPO Market Maker in Hawkes LOB (gated by AS
                        sanity check) and benchmark vs AS.
    evaluate            Run CPCV out-of-sample evaluation (24h embargo) and
                        Deflated Sharpe Ratio reporting.
    execute             Train Double-DQN Optimal Execution agent and benchmark
                        Implementation Shortfall vs AC/TWAP/VWAP.
```

---

## 2. Standard CLI Workflows

### 2.1 Zero-Latency Poisson Sanity Check (`rlmm sanity-check`)

Verify that the agent recovers the closed-form Avellaneda-Stoikov (2008) optimal spread within the $2\%$ tolerance (`REQ-AGT-001`):

```bash
rlmm sanity-check --config configs/as_poisson_sanity.yaml
```

### 2.2 Training PPO Market Maker in Hawkes LOB (`rlmm train`)

Run the Poisson sanity gate, train `PPOMarketMaker` in the mutually exciting Hawkes LOB environment with $5\text{--}50\text{ ms}$ uniform stochastic latency (`REQ-SIM-001`), $k = -0.45$ L2 queue reconstruction (`REQ-SIM-002`), and 1,000-share inventory cap (`REQ-AGT-002`), and benchmark against `AvellanedaStoikovAgent`:

```bash
rlmm train --config configs/ppo_hawkes_mm.yaml
```

### 2.3 Out-of-Sample CPCV & Deflated Sharpe Evaluation (`rlmm evaluate`)

```bash
rlmm evaluate --config configs/ppo_hawkes_mm.yaml
```

### 2.4 Optimal Execution with Double-DQN vs. Almgren-Chriss & TWAP (`rlmm execute`)

Train `DoubleDQNExecutionAgent` to liquidate a 1,000-share parent order over 25 steps and compare Implementation Shortfall (IS) mean and variance against `AlmgrenChrissAgent`, `TWAPAgent`, and `VWAPAgent`:

```bash
rlmm execute --config configs/dqn_almgren_execution.yaml
```

---

## 3. Python API Quickstart

```python
import numpy as np
from src.data.queue_reconstructor import estimate_queue_position
from src.envs.matching_engine import Order, apply_stochastic_latency
from src.envs.sanity_check import run_poisson_sanity_check, validate_as_convergence
from src.envs.market_making_env import MarketMakingEnv
from src.eval.cpcv_engine import CPCVEngine, calculate_dsr

# 1. Verify Avellaneda-Stoikov Poisson sanity check (< 2% spread error)
sc = run_poisson_sanity_check(gamma=0.10, sigma=0.20, k=1.50, episodes=160, learning_rate=0.035)
assert validate_as_convergence(sc["agent_spread"], sc["as_theoretical_spread"])

# 2. Apply uniform 5-50 ms stochastic latency (REQ-SIM-001) & L2 queue bias k in (-1, 0] (REQ-SIM-002)
rng = np.random.default_rng(42)
order = Order(order_id="ORD-1", side="BUY", price=99.95, size=10.0, submission_time_ms=1000)
delay_ms = apply_stochastic_latency(order, rng)
rem_queue = estimate_queue_position(volume_ahead=80.0, k=-0.45)

# 3. Step MarketMakingEnv with 1,000-share hard inventory guardrail (REQ-AGT-002)
env = MarketMakingEnv(regime="hawkes", max_inventory=1000.0, seed=42)
obs = env.reset(seed=42)
obs, reward, done, info = env.step((0.12, 0.12, 10.0, 10.0))
```

