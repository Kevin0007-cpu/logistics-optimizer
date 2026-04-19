"""
optimization/transportation.py
═══════════════════════════════════════════════════════════════════════════════
Transportation Problem Solver
  Phase 1 – Initial Basic Feasible Solution (IBFS) via:
            Vogel's Approximation Method (VAM)
  Phase 2 – Optimality Test & Improvement via:
            Modified Distribution Method (MODI / u-v method)

Concepts used:
  • Balance check (add dummy row/column if needed)
  • Penalty calculation (VAM)
  • Opportunity costs (MODI u-v)
  • Stepping-stone path for loop improvement
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import copy


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — BALANCE THE PROBLEM
# ─────────────────────────────────────────────────────────────────────────────

def balance_transportation(cost, supply, demand):
    """
    Ensure total supply == total demand.
    If not balanced, add a dummy source (row) or dummy destination (col)
    with zero cost and the surplus amount.

    Returns
    -------
    cost   : np.ndarray  (possibly padded)
    supply : list
    demand : list
    is_balanced : bool  (was it already balanced?)
    """
    cost   = np.array(cost, dtype=float)
    supply = list(supply)
    demand = list(demand)

    ts, td = sum(supply), sum(demand)

    if ts == td:
        return cost, supply, demand, True

    if ts > td:
        # Add dummy destination column with zero cost
        dummy_col = np.zeros((cost.shape[0], 1))
        cost      = np.hstack([cost, dummy_col])
        demand.append(ts - td)           # dummy demand absorbs surplus supply
        return cost, supply, demand, False

    else:
        # Add dummy source row with zero cost
        dummy_row = np.zeros((1, cost.shape[1]))
        cost      = np.vstack([cost, dummy_row])
        supply.append(td - ts)           # dummy supply absorbs surplus demand
        return cost, supply, demand, False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — VAM (Vogel's Approximation Method) for IBFS
# ─────────────────────────────────────────────────────────────────────────────

def _row_penalty(cost, row_done, col_done):
    """
    Penalty for a single row = difference between two smallest available costs.
    Returns (penalty, row_index, best_col_index).
    """
    penalties = []
    for i in range(cost.shape[0]):
        if row_done[i]:
            continue
        available = [(cost[i, j], j) for j in range(cost.shape[1]) if not col_done[j]]
        if len(available) == 0:
            continue
        available.sort()
        if len(available) == 1:
            pen = available[0][0]          # only one choice → penalty is that cost
        else:
            pen = available[1][0] - available[0][0]
        penalties.append((pen, i, available[0][1]))   # (penalty, row, best_col)
    return penalties


def _col_penalty(cost, row_done, col_done):
    """
    Penalty for a single column = difference between two smallest available costs.
    Returns (penalty, best_row_index, col_index).
    """
    penalties = []
    for j in range(cost.shape[1]):
        if col_done[j]:
            continue
        available = [(cost[i, j], i) for i in range(cost.shape[0]) if not row_done[i]]
        if len(available) == 0:
            continue
        available.sort()
        if len(available) == 1:
            pen = available[0][0]
        else:
            pen = available[1][0] - available[0][0]
        penalties.append((pen, available[0][1], j))   # (penalty, best_row, col)
    return penalties


def vam(cost, supply, demand):
    """
    Vogel's Approximation Method.
    Finds an initial basic feasible solution (IBFS).

    Algorithm:
    1. For each row & col, compute penalty = (2nd min - min cost).
    2. Select row/col with maximum penalty.
    3. Allocate as much as possible to the minimum cost cell in that row/col.
    4. Eliminate exhausted row/col. Repeat until all supply/demand satisfied.

    Returns
    -------
    allocation : np.ndarray   (same shape as cost)
    total_cost : float
    steps      : list of dicts (step-by-step trace for UI)
    """
    cost   = np.array(cost, dtype=float)
    supply = list(supply)
    demand = list(demand)
    m, n   = cost.shape

    allocation = np.zeros((m, n))
    row_done   = [False] * m
    col_done   = [False] * n
    steps      = []

    iteration = 0

    while not all(row_done) and not all(col_done):
        iteration += 1

        row_pens = _row_penalty(cost, row_done, col_done)
        col_pens = _col_penalty(cost, row_done, col_done)
        all_pens = row_pens + col_pens

        if not all_pens:
            break

        # Choose the maximum penalty
        all_pens.sort(reverse=True)
        best_pen, best_r, best_c = all_pens[0]

        # Allocate min(supply[r], demand[c])
        alloc = min(supply[best_r], demand[best_c])
        allocation[best_r][best_c] += alloc
        supply[best_r] -= alloc
        demand[best_c] -= alloc

        steps.append({
            "iteration": iteration,
            "penalty":   round(best_pen, 2),
            "row":       best_r,
            "col":       best_c,
            "allocated": alloc,
            "supply_remaining": list(supply),
            "demand_remaining": list(demand),
        })

        # Eliminate exhausted row or column
        if supply[best_r] == 0:
            row_done[best_r] = True
        if demand[best_c] == 0:
            col_done[best_c] = True

    total_cost = float(np.sum(cost * allocation))
    return allocation, total_cost, steps


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MODI METHOD (u-v method) for Optimality
# ─────────────────────────────────────────────────────────────────────────────

def _compute_uv(cost, allocation):
    """
    Compute u (row duals) and v (column duals) using the relation:
       cost[i][j] = u[i] + v[j]   for all basic (allocated) cells.
    Set u[0] = 0 and solve the system.
    """
    m, n = cost.shape
    u = [None] * m
    v = [None] * n
    u[0] = 0

    # Iteratively solve u[i] + v[j] = c[i][j] for basic cells
    for _ in range(m + n):
        for i in range(m):
            for j in range(n):
                if allocation[i][j] > 0:
                    if u[i] is not None and v[j] is None:
                        v[j] = cost[i][j] - u[i]
                    elif v[j] is not None and u[i] is None:
                        u[i] = cost[i][j] - v[j]

    # Fill any remaining None (disconnected variables) with 0
    u = [x if x is not None else 0 for x in u]
    v = [x if x is not None else 0 for x in v]
    return np.array(u), np.array(v)


def _stepping_stone_path(allocation, enter_r, enter_c):
    """
    Find the closed loop (stepping-stone path) starting from (enter_r, enter_c).
    Returns list of (row, col) tuples alternating +/-.
    """
    m, n = allocation.shape

    def find_loop(path):
        r, c = path[-1]
        # Try to close the loop back to start
        if len(path) >= 4 and path[0][1] == c:
            return path
        # Extend along the current row (if we're moving along row)
        if len(path) % 2 == 1:   # move along row (fix row, change col)
            for jj in range(n):
                if jj != c and allocation[r][jj] > 0 and (r, jj) not in path:
                    result = find_loop(path + [(r, jj)])
                    if result:
                        return result
        else:                      # move along col (fix col, change row)
            for ii in range(m):
                if ii != r and allocation[ii][c] > 0 and (ii, c) not in path:
                    result = find_loop(path + [(ii, c)])
                    if result:
                        return result
        return None

    return find_loop([(enter_r, enter_c)])


def modi(cost, allocation):
    """
    MODI (Modified Distribution) Method.
    Tests optimality and iteratively improves the solution.

    Algorithm:
    1. Compute u, v duals for basic cells.
    2. Compute opportunity cost d[i][j] = c[i][j] - u[i] - v[j] for non-basic.
    3. If all d[i][j] >= 0 → optimal.
    4. Else, find entering cell (most negative d), find loop, shift allocation.
    5. Repeat.

    Returns
    -------
    allocation : np.ndarray   (optimised)
    total_cost : float
    iterations : list of dicts
    """
    cost       = np.array(cost, dtype=float)
    allocation = np.array(allocation, dtype=float)
    m, n       = cost.shape
    iterations = []

    for iteration in range(100):   # max 100 improvement passes
        u, v = _compute_uv(cost, allocation)

        # Reduced costs for non-basic cells
        reduced = np.full((m, n), np.nan)
        for i in range(m):
            for j in range(n):
                if allocation[i][j] == 0:
                    reduced[i][j] = cost[i][j] - u[i] - v[j]

        # Find most negative reduced cost
        min_val = np.nanmin(reduced)
        if min_val >= -1e-8:
            # All reduced costs ≥ 0 → optimal
            iterations.append({"iteration": iteration, "status": "optimal",
                                "u": list(u), "v": list(v)})
            break

        # Entering cell
        enter_pos = np.unravel_index(np.nanargmin(reduced), reduced.shape)
        enter_r, enter_c = enter_pos

        # Find stepping-stone loop
        loop = _stepping_stone_path(allocation, enter_r, enter_c)
        if loop is None:
            break   # degenerate: no loop found (rare)

        # Theta = min of allocations at '-' positions (odd indices)
        minus_vals = [allocation[loop[k][0]][loop[k][1]] for k in range(1, len(loop), 2)]
        theta = min(minus_vals)

        # Update allocations along the loop
        for k, (ri, ci) in enumerate(loop):
            if k % 2 == 0:
                allocation[ri][ci] += theta
            else:
                allocation[ri][ci] -= theta

        total_cost = float(np.sum(cost * allocation))
        iterations.append({
            "iteration":  iteration,
            "entering":   (enter_r, enter_c),
            "theta":      theta,
            "reduced_cost": round(min_val, 4),
            "total_cost": round(total_cost, 2),
            "loop":       loop,
        })

    total_cost = float(np.sum(cost * allocation))
    return allocation, total_cost, iterations


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MASTER SOLVER (public API)
# ─────────────────────────────────────────────────────────────────────────────

def solve_transportation(cost_matrix, supply, demand,
                         row_names=None, col_names=None):
    """
    Full pipeline: balance → VAM (IBFS) → MODI (optimal).

    Parameters
    ----------
    cost_matrix : 2-D list/array
    supply      : 1-D list  (supply at each source)
    demand      : 1-D list  (demand at each destination)
    row_names   : list of str  (optional)
    col_names   : list of str  (optional)

    Returns
    -------
    dict with keys:
        balanced_cost, balanced_supply, balanced_demand,
        vam_allocation, vam_cost, vam_steps,
        optimal_allocation, optimal_cost, modi_iterations,
        row_names, col_names, was_balanced
    """
    # Step 1 – Balance
    bal_cost, bal_supply, bal_demand, was_balanced = balance_transportation(
        cost_matrix, supply, demand)

    m, n = bal_cost.shape
    if row_names is None:
        row_names = [f"Source_{i+1}" for i in range(m)]
    if col_names is None:
        col_names = [f"Dest_{j+1}" for j in range(n)]

    # Extend names if dummy was added
    if len(row_names) < m:
        row_names = list(row_names) + ["Dummy_Source"]
    if len(col_names) < n:
        col_names = list(col_names) + ["Dummy_Dest"]

    # Step 2 – VAM for IBFS
    vam_alloc, vam_cost, vam_steps = vam(bal_cost, bal_supply, bal_demand)

    # Step 3 – MODI for optimal solution
    opt_alloc, opt_cost, modi_iters = modi(bal_cost, vam_alloc.copy())

    return {
        "balanced_cost":      bal_cost,
        "balanced_supply":    bal_supply,
        "balanced_demand":    bal_demand,
        "was_balanced":       was_balanced,
        "vam_allocation":     vam_alloc,
        "vam_cost":           round(vam_cost, 2),
        "vam_steps":          vam_steps,
        "optimal_allocation": opt_alloc,
        "optimal_cost":       round(opt_cost, 2),
        "modi_iterations":    modi_iters,
        "row_names":          row_names,
        "col_names":          col_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cost   = [[4,8,1],[7,2,3],[3,6,9]]
    supply = [120, 80, 80]
    demand = [150, 70, 60]

    result = solve_transportation(cost, supply, demand,
        row_names=["WH_A","WH_B","WH_C"],
        col_names=["Store_1","Store_2","Store_3"])

    print(f"\n✅ VAM Initial Cost   : {result['vam_cost']}")
    print(f"✅ MODI Optimal Cost  : {result['optimal_cost']}")
    print(f"   Was Balanced?      : {result['was_balanced']}")
    print(f"   MODI Iterations    : {len(result['modi_iterations'])}")
    print("\nOptimal Allocation:")
    print(pd.DataFrame(result['optimal_allocation'],
        index=result['row_names'], columns=result['col_names']).to_string())
