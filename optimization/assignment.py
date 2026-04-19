"""
optimization/assignment.py
═══════════════════════════════════════════════════════════════════════════════
Assignment Problem Solver — Hungarian Algorithm
  Works for minimization. For maximization, converts by subtracting from max.

Algorithm steps:
  1.  Row reduction  (subtract row minimum from each row)
  2.  Column reduction (subtract col minimum from each col)
  3.  Cover all zeros with minimum number of lines
  4.  If #lines == n → optimal assignment found
  5.  Else find minimum uncovered element, subtract from uncovered,
      add to double-covered, go to step 3.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import copy
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — MAKE SQUARE (pad with zeros if rectangular)
# ─────────────────────────────────────────────────────────────────────────────

def make_square(cost):
    """Pad a rectangular cost matrix to square with large dummy costs (0 for min)."""
    cost = np.array(cost, dtype=float)
    m, n = cost.shape
    if m == n:
        return cost, m, False
    size = max(m, n)
    square = np.zeros((size, size))
    square[:m, :n] = cost
    return square, size, True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ROW & COLUMN REDUCTION
# ─────────────────────────────────────────────────────────────────────────────

def row_reduce(matrix):
    """Subtract the minimum of each row from every element in that row."""
    reduced = matrix.copy()
    for i in range(reduced.shape[0]):
        reduced[i] -= reduced[i].min()
    return reduced

def col_reduce(matrix):
    """Subtract the minimum of each column from every element in that col."""
    reduced = matrix.copy()
    for j in range(reduced.shape[1]):
        reduced[:, j] -= reduced[:, j].min()
    return reduced


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MINIMUM LINE COVER (to count zeros)
# ─────────────────────────────────────────────────────────────────────────────

def min_line_cover(matrix):
    """
    Find the minimum number of horizontal/vertical lines to cover all zeros.
    Uses a greedy matching approach:
      1. Find a maximal independent zero assignment.
      2. Mark rows without assignment.
      3. Mark columns that have zeros in marked rows.
      4. Mark rows that have assignments in marked columns.
      5. Repeat 3-4 until stable.
      6. Covered lines = unmarked rows + marked cols.

    Returns
    -------
    covered_rows : set of row indices (lines drawn through rows)
    covered_cols : set of col indices (lines drawn through cols)
    assignment   : dict {row: col} — partial zero assignment
    """
    n = matrix.shape[0]

    # Find a zero assignment greedily
    row_assign = [-1] * n   # row_assign[i] = col assigned to row i (-1 = none)
    col_assign = [-1] * n   # col_assign[j] = row assigned to col j (-1 = none)

    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 0 and col_assign[j] == -1 and row_assign[i] == -1:
                row_assign[i] = j
                col_assign[j] = i

    # Mark rows with no assignment
    marked_rows = set(i for i in range(n) if row_assign[i] == -1)
    marked_cols = set()

    changed = True
    while changed:
        changed = False
        # Mark cols that have a zero in a marked row
        for i in marked_rows:
            for j in range(n):
                if matrix[i][j] == 0 and j not in marked_cols:
                    marked_cols.add(j)
                    changed = True
        # Mark rows that have an assignment in a marked col
        for j in marked_cols:
            r = col_assign[j]
            if r != -1 and r not in marked_rows:
                marked_rows.add(r)
                changed = True

    # Covered = unmarked rows + marked cols
    covered_rows = set(range(n)) - marked_rows
    covered_cols = marked_cols
    assignment = {i: row_assign[i] for i in range(n) if row_assign[i] != -1}
    return covered_rows, covered_cols, assignment


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — ZERO ASSIGNMENT (extract optimal from zero matrix)
# ─────────────────────────────────────────────────────────────────────────────

def extract_assignment(matrix):
    """
    Extract a complete one-to-one assignment from zeros of the reduced matrix.
    Uses a recursive backtracking approach for correctness.

    Returns list of (row, col) tuples.
    """
    n = matrix.shape[0]
    assignment = []
    assigned_cols = set()

    def backtrack(row):
        if row == n:
            return True
        for col in range(n):
            if matrix[row][col] == 0 and col not in assigned_cols:
                assigned_cols.add(col)
                assignment.append((row, col))
                if backtrack(row + 1):
                    return True
                assigned_cols.remove(col)
                assignment.pop()
        return False

    backtrack(0)
    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FULL HUNGARIAN ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

def hungarian(cost_matrix, maximize=False):
    """
    Hungarian Algorithm for the Assignment Problem.

    Parameters
    ----------
    cost_matrix : 2-D list or np.ndarray
    maximize    : bool  (if True, convert to maximization problem)

    Returns
    -------
    assignment  : list of (row, col) tuples — optimal assignment
    total_cost  : float
    steps       : list of dicts — step-by-step trace
    """
    original = np.array(cost_matrix, dtype=float)

    # Maximization → convert to minimization
    if maximize:
        matrix = original.max() - original
    else:
        matrix = original.copy()

    # Make square if needed
    matrix, n, was_padded = make_square(matrix)
    orig_sq = original.copy()
    if was_padded:
        orig_sq_full = np.zeros((n, n))
        orig_sq_full[:original.shape[0], :original.shape[1]] = original
        orig_sq = orig_sq_full

    steps = []
    steps.append({"step": "initial", "matrix": matrix.copy(),
                  "description": "Original cost matrix (after max→min conversion if needed)"})

    # Step 1 – Row reduction
    matrix = row_reduce(matrix)
    steps.append({"step": "row_reduce", "matrix": matrix.copy(),
                  "description": "After row reduction (subtract row min)"})

    # Step 2 – Column reduction
    matrix = col_reduce(matrix)
    steps.append({"step": "col_reduce", "matrix": matrix.copy(),
                  "description": "After column reduction (subtract col min)"})

    # Steps 3-5 – Iterate until n lines cover all zeros
    for iteration in range(100):
        covered_rows, covered_cols, zero_assign = min_line_cover(matrix)
        num_lines = len(covered_rows) + len(covered_cols)

        steps.append({
            "step":          f"cover_{iteration}",
            "matrix":        matrix.copy(),
            "covered_rows":  list(covered_rows),
            "covered_cols":  list(covered_cols),
            "num_lines":     num_lines,
            "description":   f"Iteration {iteration}: {num_lines} lines cover all zeros (need {n})"
        })

        if num_lines >= n:
            break   # Optimal found

        # Find minimum uncovered element
        uncovered_vals = []
        for i in range(n):
            for j in range(n):
                if i not in covered_rows and j not in covered_cols:
                    uncovered_vals.append(matrix[i][j])
        min_uncovered = min(uncovered_vals)

        # Update matrix
        for i in range(n):
            for j in range(n):
                if i not in covered_rows and j not in covered_cols:
                    matrix[i][j] -= min_uncovered
                elif i in covered_rows and j in covered_cols:
                    matrix[i][j] += min_uncovered

        steps.append({
            "step":          f"update_{iteration}",
            "matrix":        matrix.copy(),
            "min_uncovered": min_uncovered,
            "description":   f"Subtracted {min_uncovered} from uncovered, added to intersections"
        })

    # Extract final assignment. SciPy's implementation handles ties and
    # degenerate zero covers more reliably than the trace-only reduced matrix.
    rows, cols = linear_sum_assignment(matrix)
    assignment = list(zip(rows.tolist(), cols.tolist()))

    # Compute total cost from ORIGINAL matrix
    orig_arr  = np.array(cost_matrix, dtype=float)
    total_cost = sum(orig_arr[r][c] for r, c in assignment
                     if r < orig_arr.shape[0] and c < orig_arr.shape[1])

    steps.append({"step": "assignment", "pairs": assignment,
                  "total_cost": round(total_cost, 2),
                  "description": "Optimal assignment extracted"})

    return assignment, round(total_cost, 2), steps


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PUBLIC SOLVER API
# ─────────────────────────────────────────────────────────────────────────────

def solve_assignment(cost_matrix, row_names=None, col_names=None, maximize=False):
    """
    Solve the assignment problem.

    Returns dict with assignment details and named results.
    """
    cost_arr = np.array(cost_matrix, dtype=float)
    m, n     = cost_arr.shape

    if row_names is None:
        row_names = [f"Agent_{i+1}" for i in range(m)]
    if col_names is None:
        col_names = [f"Task_{j+1}" for j in range(n)]

    assignment, total_cost, steps = hungarian(cost_matrix, maximize=maximize)

    # Build result table
    results = []
    for r, c in assignment:
        if r < m and c < n:
            results.append({
                "Agent":      row_names[r],
                "Task":       col_names[c],
                "Cost/Profit": cost_arr[r][c],
            })

    return {
        "assignment":   assignment,
        "total_cost":   total_cost,
        "steps":        steps,
        "result_table": pd.DataFrame(results),
        "row_names":    row_names,
        "col_names":    col_names,
        "maximize":     maximize,
        "cost_matrix":  cost_arr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cost = [[9, 2, 7, 8],
            [6, 4, 3, 7],
            [5, 8, 1, 8],
            [7, 6, 9, 4]]

    result = solve_assignment(cost,
        row_names=["Worker_1","Worker_2","Worker_3","Worker_4"],
        col_names=["Task_A","Task_B","Task_C","Task_D"])

    print(f"\n✅ Total Minimum Cost: {result['total_cost']}")
    print("\nOptimal Assignment:")
    print(result['result_table'].to_string(index=False))
    print(f"\nHungarian steps taken: {len(result['steps'])}")
