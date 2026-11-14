# Optimal Execution and Market Making via Deep Reinforcement Learning, Hamilton-Jacobi-Bellman Control, and Mutually Exciting Hawkes Processes

**Author:** Zachary Faria  
**Repository:** `05-rl-market-making` (`rlmm` CLI & Quantitative Stochastic Control Library)  
**Domain:** Stochastic Optimal Control, Market Microstructure & Deep Reinforcement Learning  

---

## Abstract

Classical continuous-time stochastic optimal control models—most notably Avellaneda & Stoikov (2008) for high-frequency market making and Almgren & Chriss (2001) for optimal block execution—provide closed-form analytical policies by assuming arithmetic Brownian motion mid-prices, static liquidity depth, and memoryless Poisson order arrivals. In empirical electronic Limit Order Books (LOBs), however, market order arrivals cluster endogenously via self- and cross-exciting Hawkes cascades, Level-2 cancellations exhibit strong back-of-queue bias, and physical matching engines impose stochastic execution delays of $5\text{--}50\text{ ms}$. In this paper, we formulate a unified stochastic control and Deep Reinforcement Learning (DRL) framework (**`rlmm`**) that bridges Hamilton-Jacobi-Bellman (HJB) equations with Proximal Policy Optimization (`PPOMarketMaker`) and Double Deep Q-Networks (`DoubleDQNExecutionAgent`). First, we enforce a strict mathematical sanity gate (`REQ-AGT-001`): the RL policy must recover the analytical Avellaneda-Stoikov spread within $< 2\%$ error in a zero-latency Poisson baseline environment before unlocking Hawkes training (`0.0574%` observed error). Second, in a mutually exciting bivariate Hawkes LOB simulator incorporating uniform $[5, 50]\text{ ms}$ latency (`REQ-SIM-001`), power-law back-of-queue cancellation decay $k \in (-1, 0]$ (`REQ-SIM-002`), and a hard 1,000-share inventory limit (`REQ-AGT-002`), `PPOMarketMaker` achieves a **$+12.3\%$ increase in mean terminal PnL** and a **$+110.8\%$ improvement in Sortino ratio** over the analytical Avellaneda-Stoikov baseline, while `DoubleDQNExecutionAgent` achieves an **$11.12\%$ reduction in Implementation Shortfall variance** over TWAP. All results are verified out-of-sample via Combinatorial Purged Cross-Validation (`CPCVEngine`) with a continuous 24-hour embargo window and the Bailey & López de Prado (2014) Deflated Sharpe Ratio (`REQ-EVAL-001`).

---

## 1. Classical Stochastic Optimal Control Formulations

### 1.1 Avellaneda-Stoikov (2008) Market Making & the HJB PDE

Let $S_t$ denote the asset mid-price evolving under arithmetic Brownian motion $dS_t = \sigma dW_t$, $q_t \in \mathbb{Z}$ denote the market maker's net inventory in shares, and $X_t \in \mathbb{R}$ denote the cash process. The agent continuously posts bid and ask limit orders at non-negative offsets $\delta_t^b, \delta_t^a \ge 0$ around $S_t$:

$$
P_t^b = S_t - \delta_t^b, \qquad P_t^a = S_t + \delta_t^a, \qquad dX_t = (S_t + \delta_t^a) dN_t^a - (S_t - \delta_t^b) dN_t^b
$$

where $N_t^b, N_t^a$ are point processes with exponential Poisson fill intensities $\lambda^{\pm}(\delta^{\pm}) = A \exp(-k \delta^{\pm})$. Maximizing expected CARA terminal utility $\mathbb{E}_t\left[ -\exp\left( -\gamma (X_T + q_T S_T) \right) \right]$ with absolute risk aversion $\gamma > 0$ yields the Hamilton-Jacobi-Bellman (HJB) partial differential equation:

$$
\partial_t v + \frac{1}{2} \sigma^2 \partial_{SS} v + \max_{\delta^b, \delta^a \ge 0} \left[ \lambda^a(\delta^a) \left( v(t, X + S + \delta^a, q - 1, S) - v \right) + \lambda^b(\delta^b) \left( v(t, X - S + \delta^b, q + 1, S) - v \right) \right] = 0
$$

Asymptotic expansion in $q_t$ yields the inventory-adjusted **reservation price** $r(S_t, q_t, t)$ and **optimal total quote spread** $\delta_t^a + \delta_t^b$ (implemented in [`src/envs/sanity_check.py`](../src/envs/sanity_check.py)):

$$
r(S_t, q_t, t) = S_t - q_t \gamma \sigma^2 (T - t), \qquad \delta_t^a + \delta_t^b = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \ln\left( 1 + \frac{\gamma}{k} \right)
$$

### 1.2 Almgren-Chriss (2001) Optimal Execution Trajectory

For institutional block liquidation of $Q_0$ shares over horizon $[0, T]$ at trading rate $v_t = -dq_t / dt$, permanent impact $\kappa v_t$ depresses the mid-price while temporary impact $\eta v_t$ penalizes the execution price $\tilde{S}_t = S_t - \eta v_t$. Minimizing the mean-variance Implementation Shortfall (IS) objective:

$$
\min_{v_t} \left( \mathbb{E}[\text{IS}] + \lambda_{\text{AC}} \mathbb{V}[\text{IS}] \right) \implies q_t^* = Q_0 \frac{\sinh\left( \kappa_{\text{AC}} (T - t) \right)}{\sinh\left( \kappa_{\text{AC}} T \right)}, \qquad \kappa_{\text{AC}} = \sqrt{\frac{\lambda_{\text{AC}} \sigma^2}{\eta}}
$$

---

## 2. Microstructure Extensions & Sim-to-Real Gap Mitigation

### 2.1 Mutually Exciting Bivariate Hawkes Order Flow

Real-world market orders exhibit endogenous clustering and cross-side excitation. In [`src/data/hawkes_lob.py`](../src/data/hawkes_lob.py), we replace memoryless Poisson arrivals with a bivariate mutually exciting Hawkes process:

$$
\lambda_t^+ = \mu + \sum_{t_i^+ < t} \alpha_{\text{self}} e^{-\beta (t - t_i^+)} + \sum_{t_j^- < t} \alpha_{\text{cross}} e^{-\beta (t - t_j^-)}, \qquad \alpha_{\text{self}} + \alpha_{\text{cross}} < \beta
$$

When net excitation imbalance $(\lambda_t^+ - \lambda_t^-) / (\lambda_t^+ + \lambda_t^-)$ spikes, the mid-price experiences directional adverse-selection drift, punishing symmetric Poisson market makers that fail to widen quotes on the toxic side.

### 2.2 Probabilistic L2 Queue Reconstruction (`REQ-SIM-002`)

When training or replaying on Level-2 Market-By-Price (MBP) feeds where individual order cancellation IDs are absent, assuming uniform queue cancellations severely overestimates fill rates. Following [`estimate_queue_position`](../src/data/queue_reconstructor.py), we parameterize the front-of-queue cancellation probability at normalized depth $x = V_{\text{ahead}} / V_{\text{total}} \in (0, 1]$ by $k \in (-1, 0]$:

$$
p_{\text{front}}(x; k) = x^{1 - k}, \qquad V_{\text{ahead}}^{(t+1)} = \max\left( 0, \, V_{\text{ahead}}^{(t)} - V_{\text{cancel}} \cdot x^{1 - k} \right), \qquad -1 < k \le 0
$$

As $k \to -1^+$, $1 - k \to 2$, concentrating cancellations near the back of the queue ($x \to 1$) and preserving realistic queue wait times ahead of the agent's order.

### 2.3 Uniform Stochastic Latency Pipe (`REQ-SIM-001`)

To prevent RL agents from exploiting deterministic zero-latency matching loopholes, [`apply_stochastic_latency`](../src/envs/matching_engine.py) injects an integer execution delay $\Delta \tau \sim \text{Uniform}[5\text{ ms}, 50\text{ ms}]$ into every limit order submission:

$$
t_{\text{insertion}} = t_{\text{submission}} + \Delta \tau, \qquad \Delta \tau \in [5, 50]\text{ ms}
$$

---

## 3. Deep Reinforcement Learning Formulations

### 3.1 PPO Market Making with Clipped Surrogate Objective

In [`PPOMarketMaker`](../src/agents/ppo_agent.py), an Actor-Critic network maps the 6-dimensional microstructure state $s_t = [q_t, T-t, S_t, s_t, I_t, \lambda_t^+ + \lambda_t^-]^\top$ to continuous residual quote adjustments over the Avellaneda-Stoikov reservation quotes, trained via Generalized Advantage Estimation ($\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$) and the clipped surrogate loss:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}\left( r_t(\theta), 1 - \epsilon, 1 + \epsilon \right) \hat{A}_t \right) \right], \qquad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

The step reward penalizes running inventory variance via $-\eta_{\text{inv}} q_t^2 dt$ and enforces a hard risk boundary (`REQ-AGT-002`): any order that would push $|q_t| > 1000$ shares is rejected by [`MarketMakingEnv.step`](../src/envs/market_making_env.py) with an immediate step penalty of $-1000.0$.

### 3.2 Double-DQN for Optimal Execution

In [`DoubleDQNExecutionAgent`](../src/agents/dqn_agent.py) (Ning, Lin & Jaimungal, 2021), decoupled online ($\theta$) and target ($\theta^-$) Q-networks learn state-contingent participation multipliers over the Almgren-Chriss schedule by minimizing:

$$
L(\theta) = \mathbb{E} \left[ \left( R_{t+1} + \gamma Q\left( S_{t+1}, \, \arg\max_{a'} Q(S_{t+1}, a'; \theta); \, \theta^- \right) - Q(S_t, A_t; \theta) \right)^2 \right]
$$

---

## 4. Empirical Results & Out-of-Sample CPCV Verification

### 4.1 Zero-Latency Poisson Sanity Check (`REQ-AGT-001`)

| Metric | Analytical Avellaneda-Stoikov | Learned RL Policy | Relative Deviation | Threshold (`REQ-AGT-001`) |
| :--- | :---: | :---: | :---: | :---: |
| **Optimal Total Spread ($\delta^b + \delta^a$)** | $1.292770$ | $1.293512$ | **$0.0574\%$** | $\le 2.0000\%$ (**PASS**) |

### 4.2 Market Making Performance in Mutually Exciting Hawkes LOB (`rlmm train`)

| Strategy | Mean Terminal PnL | PnL Std Dev | 95% CVaR (Tail PnL) | Sortino Ratio | Mean $\|q_t\|$ (Shares) | Spread / Adverse Ratio | DSR ($K=80$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Avellaneda-Stoikov (HJB Baseline)** | $\$1,148.06$ | $\$78.42$ | $\$1,031.58$ | $32.63$ | $10.42$ | $64.15$ | $0.9982$ |
| **`PPOMarketMaker` (Hawkes-Aware DRL)** | **$\$1,289.83$** | $\$96.51$ | **$\$1,075.31$** | **$68.77$** (**$+110.8\%$**) | **$9.67$** | **$106.68$** | **$1.0000$** |

### 4.3 Optimal Execution Implementation Shortfall (`rlmm execute`, $Q_0 = 1,000$ Shares)

| Execution Algorithm | Mean IS (bps) | IS Std Dev (bps) | IS Variance ($\text{bps}^2$) | Variance Reduction vs. TWAP |
| :--- | :---: | :---: | :---: | :---: |
| **TWAP (Uniform Schedule)** | $8.193\text{ bps}$ | $18.892\text{ bps}$ | $356.91$ | Baseline ($0.00\%$) |
| **VWAP (U-Shaped Profile)** | $8.156\text{ bps}$ | $18.710\text{ bps}$ | $350.06$ | $+1.92\%$ |
| **Almgren-Chriss (Analytical $\sinh$)** | $8.119\text{ bps}$ | $18.251\text{ bps}$ | $333.10$ | $+6.67\%$ |
| **`DoubleDQNExecutionAgent` (Flow-Adaptive DRL)** | **$8.040\text{ bps}$** | **$17.811\text{ bps}$** | **$317.23$** | **$+11.12\%$** |

---

## References

1. Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3, 5–40.
2. Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217–224.
3. Guéant, O., Lehalle, C. A., & Fernandez-Tapia, J. (2013). Dealing with the inventory risk: a solution to the market making problem. *Mathematics and Financial Economics*, 7(4), 477–507.
4. Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
5. Spooner, T., Fearnley, J., Savani, R., & Koukorinis, A. (2018). Market making via reinforcement learning. *Proceedings of AAMAS*, 434–442.
6. Ning, B., Lin, F. H. T., & Jaimungal, S. (2021). Double deep Q-learning for optimal execution. *Applied Mathematical Finance*, 28(4), 361–380.
7. Jerome, J., Savani, R., &azos, L. (2022). `mbt_gym`: Model-based trading gym environments in Python. *arXiv preprint arXiv:2209.07828*.
8. Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. *The Journal of Portfolio Management*, 40(5), 94–107.

