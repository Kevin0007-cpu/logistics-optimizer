"""
optimization/decision_theory.py
═══════════════════════════════════════════════════════════════════════════════
Decision Theory Module
Criteria implemented (all under uncertainty — no probability info needed
except for Laplace which assumes equal probability):

  1. Maximax     — optimist: pick strategy with highest possible payoff
  2. Maximin     — pessimist: pick strategy that maximises the worst outcome
  3. Minimax Regret — minimise the maximum opportunity loss (regret)
  4. Laplace     — assume equal probability, pick highest average payoff
  5. Hurwicz     — weighted mix of best/worst (coefficient of optimism α)

Works for both profit (maximization) and cost (minimization) matrices.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CRITERION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def maximax(payoff_matrix, strategy_names, state_names):
    """
    Maximax criterion (optimistic decision maker).
    Step: For each strategy, find the maximum payoff.
          Select the strategy with the overall maximum.

    Used when: Decision maker is optimistic / risk-seeking.
    """
    matrix = np.array(payoff_matrix, dtype=float)
    row_max = matrix.max(axis=1)          # best outcome per strategy
    best_idx = int(np.argmax(row_max))    # strategy with best possible outcome

    return {
        "criterion":      "Maximax",
        "row_scores":     dict(zip(strategy_names, row_max.round(4))),
        "best_strategy":  strategy_names[best_idx],
        "best_value":     round(row_max[best_idx], 4),
        "description":    "Optimistic: choose strategy with highest maximum payoff.",
        "detail_df":      pd.DataFrame({"Strategy": strategy_names,
                                        "Max Payoff": row_max.round(4),
                                        "Selected": ["✔" if i==best_idx else "" for i in range(len(strategy_names))]})
    }


def maximin(payoff_matrix, strategy_names, state_names):
    """
    Maximin criterion (pessimistic / conservative decision maker).
    Step: For each strategy, find the minimum payoff.
          Select the strategy with the highest minimum (best of worst).

    Used when: Decision maker is risk-averse / wants guaranteed floor.
    """
    matrix = np.array(payoff_matrix, dtype=float)
    row_min = matrix.min(axis=1)           # worst outcome per strategy
    best_idx = int(np.argmax(row_min))     # strategy where worst is least bad

    return {
        "criterion":      "Maximin",
        "row_scores":     dict(zip(strategy_names, row_min.round(4))),
        "best_strategy":  strategy_names[best_idx],
        "best_value":     round(row_min[best_idx], 4),
        "description":    "Pessimistic: choose strategy with highest minimum payoff.",
        "detail_df":      pd.DataFrame({"Strategy": strategy_names,
                                        "Min Payoff": row_min.round(4),
                                        "Selected": ["✔" if i==best_idx else "" for i in range(len(strategy_names))]})
    }


def minimax_regret(payoff_matrix, strategy_names, state_names):
    """
    Minimax Regret criterion (Savage criterion).
    Step: Build regret (opportunity loss) matrix:
            regret[i][j] = max(col j) - payoff[i][j]
          For each strategy, find max regret.
          Choose strategy with minimum of those max regrets.

    Used when: Decision maker wants to minimise post-decision regret.
    """
    matrix = np.array(payoff_matrix, dtype=float)
    col_max = matrix.max(axis=0)            # best payoff in each state
    regret  = col_max - matrix              # opportunity loss matrix

    row_max_regret = regret.max(axis=1)     # worst regret per strategy
    best_idx = int(np.argmin(row_max_regret))

    regret_df = pd.DataFrame(regret.round(4),
                             index=strategy_names, columns=state_names)
    regret_df["Max Regret"] = row_max_regret.round(4)
    regret_df["Selected"]   = ["✔" if i==best_idx else "" for i in range(len(strategy_names))]

    return {
        "criterion":       "Minimax Regret",
        "regret_matrix":   regret,
        "regret_df":       regret_df,
        "row_scores":      dict(zip(strategy_names, row_max_regret.round(4))),
        "best_strategy":   strategy_names[best_idx],
        "best_value":      round(row_max_regret[best_idx], 4),
        "description":     "Savage: minimise the maximum opportunity loss (regret).",
    }


def laplace(payoff_matrix, strategy_names, state_names):
    """
    Laplace criterion (equal probability / Bayes criterion).
    Step: Assign equal probability (1/n) to each state of nature.
          Compute expected payoff per strategy.
          Select strategy with highest expected payoff.

    Used when: No information about state probabilities (treat as equal).
    """
    matrix = np.array(payoff_matrix, dtype=float)
    n_states = matrix.shape[1]
    expected = matrix.mean(axis=1)          # equal weight = simple average
    best_idx = int(np.argmax(expected))

    prob = round(1 / n_states, 4)
    return {
        "criterion":      "Laplace",
        "probability":    prob,
        "row_scores":     dict(zip(strategy_names, expected.round(4))),
        "best_strategy":  strategy_names[best_idx],
        "best_value":     round(expected[best_idx], 4),
        "description":    f"Equal probability ({prob} each): choose highest expected payoff.",
        "detail_df":      pd.DataFrame({"Strategy": strategy_names,
                                        "Expected Payoff": expected.round(4),
                                        "Selected": ["✔" if i==best_idx else "" for i in range(len(strategy_names))]})
    }


def hurwicz(payoff_matrix, strategy_names, state_names, alpha=0.5):
    """
    Hurwicz criterion (coefficient of optimism).
    Step: Weighted average of best and worst outcomes per strategy:
            H[i] = α × max(row i) + (1-α) × min(row i)
          Select strategy with highest H.

    Parameters
    ----------
    alpha : float in [0,1]
      α=1 → pure optimist (Maximax)
      α=0 → pure pessimist (Maximin)
      α=0.5 → equal weight (default)

    Used when: Decision maker has partial optimism quantified by α.
    """
    matrix  = np.array(payoff_matrix, dtype=float)
    row_max = matrix.max(axis=1)
    row_min = matrix.min(axis=1)
    H       = alpha * row_max + (1 - alpha) * row_min
    best_idx = int(np.argmax(H))

    return {
        "criterion":      f"Hurwicz (α={alpha})",
        "alpha":          alpha,
        "row_scores":     dict(zip(strategy_names, H.round(4))),
        "best_strategy":  strategy_names[best_idx],
        "best_value":     round(H[best_idx], 4),
        "description":    f"Weighted optimism α={alpha}: H = α·max + (1-α)·min per row.",
        "detail_df":      pd.DataFrame({
            "Strategy": strategy_names,
            "Max": row_max.round(4),
            "Min": row_min.round(4),
            "H Score": H.round(4),
            "Selected": ["✔" if i==best_idx else "" for i in range(len(strategy_names))]
        })
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MASTER SOLVER (public API)
# ─────────────────────────────────────────────────────────────────────────────

def solve_decision(payoff_matrix, strategy_names=None,
                   state_names=None, alpha=0.5):
    """
    Run all five decision criteria and return a comparison summary.

    Parameters
    ----------
    payoff_matrix  : 2-D list/array  (rows=strategies, cols=states of nature)
    strategy_names : list of str
    state_names    : list of str
    alpha          : float  (Hurwicz coefficient of optimism)

    Returns
    -------
    dict with individual criterion results + comparison DataFrame
    """
    matrix = np.array(payoff_matrix, dtype=float)
    m, n   = matrix.shape

    if strategy_names is None:
        strategy_names = [f"Strategy_{i+1}" for i in range(m)]
    if state_names is None:
        state_names    = [f"State_{j+1}"    for j in range(n)]

    results = {
        "maximax":        maximax(matrix, strategy_names, state_names),
        "maximin":        maximin(matrix, strategy_names, state_names),
        "minimax_regret": minimax_regret(matrix, strategy_names, state_names),
        "laplace":        laplace(matrix, strategy_names, state_names),
        "hurwicz":        hurwicz(matrix, strategy_names, state_names, alpha),
    }

    # Build comparison table
    comparison = pd.DataFrame({
        "Criterion":      [r["criterion"]     for r in results.values()],
        "Best Strategy":  [r["best_strategy"] for r in results.values()],
        "Score/Value":    [r["best_value"]    for r in results.values()],
        "Interpretation": [r["description"]   for r in results.values()],
    })

    # Score tally — which strategy is recommended most?
    tally = {}
    for r in results.values():
        s = r["best_strategy"]
        tally[s] = tally.get(s, 0) + 1
    best_overall = max(tally, key=tally.get)

    # Full payoff DataFrame for display
    payoff_df = pd.DataFrame(matrix, index=strategy_names, columns=state_names)

    return {
        "criteria_results": results,
        "comparison_table": comparison,
        "strategy_tally":   tally,
        "recommended":      best_overall,
        "payoff_df":        payoff_df,
        "strategy_names":   strategy_names,
        "state_names":      state_names,
        "alpha":            alpha,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    payoff = [[200, 100,  50],
              [300, 150, -50],
              [100,  80,  60],
              [250, 200,-100]]

    strategies = ["Strategy_1","Strategy_2","Strategy_3","Strategy_4"]
    states     = ["High_Demand","Medium_Demand","Low_Demand"]

    result = solve_decision(payoff, strategies, states, alpha=0.6)

    print("\n" + "═"*55)
    print("  DECISION THEORY ANALYSIS")
    print("═"*55)
    print(result["comparison_table"].to_string(index=False))
    print(f"\n🏆 Overall Recommended Strategy: {result['recommended']}")
    print(f"   (Recommended by {result['strategy_tally'][result['recommended']]} out of 5 criteria)")
