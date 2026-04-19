"""
optimization/game_theory.py
═══════════════════════════════════════════════════════════════════════════════
Game Theory Module — Zero-Sum Two-Player Games

Concepts implemented:
  1. Payoff Matrix construction & display
  2. Saddle Point detection (Pure Strategy solution)
  3. Dominance Reduction (simplify matrix by removing dominated strategies)
  4. Mixed Strategy (2×2 and m×n via LP)
  5. Value of the Game
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SADDLE POINT
# ─────────────────────────────────────────────────────────────────────────────

def find_saddle_point(payoff_matrix, row_names, col_names):
    """
    Find the saddle point (if it exists) for a pure strategy solution.

    A saddle point is an element that is:
      - The MINIMUM of its row  (Player A's security level)
      - The MAXIMUM of its column (Player B's security level)

    If saddle point exists → pure strategy equilibrium.
    Value of game = saddle point value.
    """
    matrix   = np.array(payoff_matrix, dtype=float)
    row_min  = matrix.min(axis=1)         # maximin for row player
    col_max  = matrix.max(axis=0)         # minimax for col player

    saddle_points = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] == row_min[i] and matrix[i][j] == col_max[j]:
                saddle_points.append({
                    "row":       row_names[i],
                    "col":       col_names[j],
                    "row_idx":   i,
                    "col_idx":   j,
                    "value":     matrix[i][j],
                })

    return {
        "saddle_points": saddle_points,
        "exists":        len(saddle_points) > 0,
        "row_minimums":  dict(zip(row_names, row_min)),
        "col_maximums":  dict(zip(col_names, col_max)),
        "maximin":       max(row_min),
        "minimax":       min(col_max),
        "game_value":    saddle_points[0]["value"] if saddle_points else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DOMINANCE REDUCTION
# ─────────────────────────────────────────────────────────────────────────────

def dominance_reduction(payoff_matrix, row_names, col_names):
    """
    Reduce the payoff matrix by removing dominated strategies iteratively.

    Dominance rules:
      Row player (maximizer):
        Row i DOMINATES row k if payoff[i][j] ≥ payoff[k][j] for all j.
        → Remove row k (dominated).
      Col player (minimizer):
        Col j DOMINATES col l if payoff[i][j] ≤ payoff[i][l] for all i.
        → Remove col l (dominated).

    Returns reduced matrix and list of elimination steps.
    """
    matrix     = np.array(payoff_matrix, dtype=float)
    row_names  = list(row_names)
    col_names  = list(col_names)
    steps      = []

    changed = True
    while changed:
        changed = False

        # Check row dominance
        to_remove_rows = []
        for i in range(matrix.shape[0]):
            for k in range(matrix.shape[0]):
                if i != k and k not in to_remove_rows:
                    if np.all(matrix[i] >= matrix[k]):     # row i dominates row k
                        to_remove_rows.append(k)
                        steps.append({
                            "type":      "Row",
                            "eliminated":row_names[k],
                            "dominated_by": row_names[i],
                            "reason":    f"Row '{row_names[i]}' ≥ Row '{row_names[k]}' in all states"
                        })
                        changed = True

        if to_remove_rows:
            keep = [i for i in range(matrix.shape[0]) if i not in to_remove_rows]
            matrix    = matrix[keep, :]
            row_names = [row_names[i] for i in keep]

        # Check column dominance
        to_remove_cols = []
        for j in range(matrix.shape[1]):
            for l in range(matrix.shape[1]):
                if j != l and l not in to_remove_cols:
                    if np.all(matrix[:, j] <= matrix[:, l]):   # col j dominates col l
                        to_remove_cols.append(l)
                        steps.append({
                            "type":         "Col",
                            "eliminated":   col_names[l],
                            "dominated_by": col_names[j],
                            "reason":       f"Col '{col_names[j]}' ≤ Col '{col_names[l]}' in all rows"
                        })
                        changed = True

        if to_remove_cols:
            keep      = [j for j in range(matrix.shape[1]) if j not in to_remove_cols]
            matrix    = matrix[:, keep]
            col_names = [col_names[j] for j in keep]

    return {
        "reduced_matrix": matrix,
        "reduced_rows":   row_names,
        "reduced_cols":   col_names,
        "steps":          steps,
        "reduced_df":     pd.DataFrame(matrix, index=row_names, columns=col_names),
        "eliminated":     len(steps) > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MIXED STRATEGY (via LP)
# ─────────────────────────────────────────────────────────────────────────────

def mixed_strategy_lp(payoff_matrix, row_names, col_names):
    """
    Solve for mixed strategy Nash Equilibrium using Linear Programming.

    Row Player (Maximizer) LP:
        max V
        s.t. sum_i p_i * payoff[i][j] ≥ V  for all j
             sum_i p_i = 1
             p_i ≥ 0

    Col Player (Minimizer) LP:
        min W
        s.t. sum_j q_j * payoff[i][j] ≤ W  for all i
             sum_j q_j = 1
             q_j ≥ 0

    Returns optimal mixed strategies and game value.
    """
    matrix = np.array(payoff_matrix, dtype=float)
    m, n   = matrix.shape

    # ── Row player LP ─────────────────────────────────────────────────────────
    # Variables are p_1..p_m and V. Maximize V, so linprog minimizes -V.
    row_obj = np.zeros(m + 1)
    row_obj[-1] = -1
    row_A_ub = []
    row_b_ub = []
    for j in range(n):
        row = np.zeros(m + 1)
        row[:m] = -matrix[:, j]
        row[-1] = 1
        row_A_ub.append(row)
        row_b_ub.append(0)
    row_A_eq = [np.r_[np.ones(m), 0]]
    row_b_eq = [1]
    row_bounds = [(0, None)] * m + [(None, None)]
    row_res = linprog(
        row_obj,
        A_ub=np.array(row_A_ub),
        b_ub=np.array(row_b_ub),
        A_eq=np.array(row_A_eq),
        b_eq=np.array(row_b_eq),
        bounds=row_bounds,
        method="highs",
    )
    if not row_res.success:
        raise ValueError(f"Unable to solve row-player mixed strategy: {row_res.message}")

    p_vals = [round(float(x), 6) for x in row_res.x[:m]]
    V_val = round(float(row_res.x[-1]), 6)

    # ── Col player LP ─────────────────────────────────────────────────────────
    # Variables are q_1..q_n and W. Minimize W.
    col_obj = np.zeros(n + 1)
    col_obj[-1] = 1
    col_A_ub = []
    col_b_ub = []
    for i in range(m):
        row = np.zeros(n + 1)
        row[:n] = matrix[i, :]
        row[-1] = -1
        col_A_ub.append(row)
        col_b_ub.append(0)
    col_A_eq = [np.r_[np.ones(n), 0]]
    col_b_eq = [1]
    col_bounds = [(0, None)] * n + [(None, None)]
    col_res = linprog(
        col_obj,
        A_ub=np.array(col_A_ub),
        b_ub=np.array(col_b_ub),
        A_eq=np.array(col_A_eq),
        b_eq=np.array(col_b_eq),
        bounds=col_bounds,
        method="highs",
    )
    if not col_res.success:
        raise ValueError(f"Unable to solve column-player mixed strategy: {col_res.message}")

    q_vals = [round(float(x), 6) for x in col_res.x[:n]]
    W_val = round(float(col_res.x[-1]), 6)

    row_strategy = pd.DataFrame({
        "Strategy":    row_names,
        "Probability": p_vals,
    })
    col_strategy = pd.DataFrame({
        "Strategy":    col_names,
        "Probability": q_vals,
    })

    return {
        "row_mixed":     row_strategy,
        "col_mixed":     col_strategy,
        "game_value":    V_val,
        "row_probs":     p_vals,
        "col_probs":     q_vals,
        "row_names":     row_names,
        "col_names":     col_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MASTER SOLVER (public API)
# ─────────────────────────────────────────────────────────────────────────────

def solve_game(payoff_matrix, row_names=None, col_names=None):
    """
    Full game theory analysis:
      1. Saddle point check
      2. Dominance reduction
      3. Mixed strategy (always computed)

    Returns comprehensive result dict.
    """
    matrix = np.array(payoff_matrix, dtype=float)
    m, n   = matrix.shape

    if row_names is None:
        row_names = [f"A_Strategy_{i+1}" for i in range(m)]
    if col_names is None:
        col_names = [f"B_Strategy_{j+1}" for j in range(n)]

    # 1 – Saddle point
    saddle = find_saddle_point(matrix, row_names, col_names)

    # 2 – Dominance reduction
    dominance = dominance_reduction(matrix, row_names, col_names)

    # 3 – Mixed strategy on reduced matrix
    red_matrix = dominance["reduced_matrix"]
    red_rows   = dominance["reduced_rows"]
    red_cols   = dominance["reduced_cols"]
    mixed      = mixed_strategy_lp(red_matrix, red_rows, red_cols)

    # Original payoff df
    payoff_df = pd.DataFrame(matrix, index=row_names, columns=col_names)

    return {
        "payoff_df":   payoff_df,
        "saddle":      saddle,
        "dominance":   dominance,
        "mixed":       mixed,
        "game_value":  saddle["game_value"] if saddle["exists"] else mixed["game_value"],
        "is_pure":     saddle["exists"],
        "row_names":   row_names,
        "col_names":   col_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    payoff = [[ 3,-1, 2],
              [-2, 4, 1],
              [ 1, 2,-3]]

    result = solve_game(payoff,
        row_names=["A_Strat_1","A_Strat_2","A_Strat_3"],
        col_names=["B_Strat_1","B_Strat_2","B_Strat_3"])

    print("═"*55)
    print(f"  Saddle Point Exists: {result['saddle']['exists']}")
    print(f"  Game Value:          {result['game_value']}")
    print(f"  Is Pure Strategy:    {result['is_pure']}")
    print("═"*55)
    print("\nRow Player Mixed Strategy:")
    print(result['mixed']['row_mixed'].to_string(index=False))
    print("\nCol Player Mixed Strategy:")
    print(result['mixed']['col_mixed'].to_string(index=False))
    print(f"\nDominance steps: {len(result['dominance']['steps'])}")
