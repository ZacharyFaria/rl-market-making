# Quantitative Validation Methodology, HJB Sanity Gate & CPCV Protocol

**Application:** `05-rl-market-making` (`rlmm`)  

---

## 1. Requirement Traceability Matrix

| Requirement ID | Subsystem | I/O Contract & Invariant | Automated Test | Status |
| :--- | :--- | :--- | :--- | :---: |
| **`REQ-SIM-001`** | Matching Engine Latency | `apply_stochastic_latency(order: Order, rng: np.random.Generator) -> int` enforces $5 \le \Delta \tau \le 50\text{ ms}$ on 100% of orders and raises `ValueError` if $< 5\text{ ms}$. | [`tests/test_matching_engine.py`](../tests/test_matching_engine.py) | Passed |
| **`REQ-SIM-002`** | L2 Queue Reconstructor | `estimate_queue_position(volume_ahead: float, k: float) -> float` raises `ValueError` when $k \notin (-1, 0]$ and assigns higher back-of-queue cancellation probability as $k \to -1^+$. | [`tests/test_queue_reconstructor.py`](../tests/test_queue_reconstructor.py) | Passed |
| **`REQ-AGT-001`** | Avellaneda-Stoikov Sanity Gate | `validate_as_convergence(agent_spread: float, as_spread: float) -> bool` requires $\le 2\%$ spread error (`0.0574%` observed) and blocks `MarketMakingEnv(regime="hawkes")` if `False`. | [`tests/test_sanity_check.py`](../tests/test_sanity_check.py) | Passed |
| **`REQ-AGT-002`** | Inventory Risk Guardrail | `MarketMakingEnv.step(action)` rejects any order that pushes $\|q_t\| > 1000$ shares (e.g., $950 + 100$), retains $q_t = 950$, and applies a $-1000.0$ step penalty. | [`tests/test_market_making_env.py`](../tests/test_market_making_env.py) | Passed |
| **`REQ-EVAL-001`** | CPCV & Deflated Sharpe Ratio | `calculate_dsr(returns_array, trial_count, return_variance)` penalizes multiple testing across `trial_count` folds, and `CPCVEngine` enforces a continuous 24-hour embargo with zero train/test overlap. | [`tests/test_cpcv_engine.py`](../tests/test_cpcv_engine.py) | Passed |

---

## 2. Combinatorial Purged Cross-Validation (`CPCVEngine`) with 24-Hour Embargo

In [`src/eval/cpcv_engine.py`](../src/eval/cpcv_engine.py), `CPCVEngine` partitions a temporal dataset into $N = 6$ contiguous groups and iterates over all $\binom{6}{2} = 15$ out-of-sample test pairs:

1. **Pre-Test Purging:** Excises `purge_steps` observations immediately preceding each test group so no open inventory trajectory bridges the boundary between training and test folds.
2. **24-Hour Continuous Embargo:** Excises all observations in $(t_{\text{test, end}}, \, t_{\text{test, end}} + 24.0\text{ hours}]$ (`embargo_hours >= 24.0`), ensuring Hawkes self-excitation memory and volatility clustering decay completely before training resumes.
3. **Zero-Intersection Assertion:** Every fold explicitly asserts:

$$
\mathcal{I}_{\text{train}} \cap \mathcal{I}_{\text{test}} = \emptyset \qquad \text{and} \qquad \mathcal{I}_{\text{train}} \cap \mathcal{I}_{\text{embargo}} = \emptyset
$$

---

## 3. Running the Verification Suite

Run all 12 unit and BDD verification tests from the repository root:

```bash
pytest -v
```

```text
tests/test_agents_and_execution.py::test_almgren_chriss_twap_vwap_schedules_sum_to_parent_order PASSED
tests/test_agents_and_execution.py::test_ppo_market_maker_and_double_dqn_execution_smoke PASSED
tests/test_cpcv_engine.py::test_req_eval_001_cpcv_zero_temporal_overlap_and_24h_embargo PASSED
tests/test_cpcv_engine.py::test_req_eval_001_deflated_sharpe_ratio_haircut PASSED
tests/test_market_making_env.py::test_req_agt_002_inventory_950_plus_100_rejected_with_minus_1000_penalty PASSED
tests/test_market_making_env.py::test_req_agt_002_negative_inventory_breach_rejected PASSED
tests/test_matching_engine.py::test_req_sim_001_uniform_latency_between_5_and_50ms PASSED
tests/test_matching_engine.py::test_req_sim_001_prohibits_zero_or_sub_5ms_latency PASSED
tests/test_queue_reconstructor.py::test_req_sim_002_domain_validation_on_k PASSED
tests/test_queue_reconstructor.py::test_req_sim_002_back_of_queue_cancellation_bias_monotonicity PASSED
tests/test_sanity_check.py::test_req_agt_001_validate_as_convergence_2_percent_threshold PASSED
tests/test_sanity_check.py::test_req_agt_001_poisson_baseline_recovery_and_hawkes_gate PASSED
============================== 12 passed in 1.13s ==============================
```

