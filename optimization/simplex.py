"""
optimization/simplex.py
═══════════════════════════════════════════════════════════════════════════════
Linear Programming Solver
  Method 1 – Simplex Method      (standard maximization)
  Method 2 – Big-M Method        (handles ≥ constraints via artificial vars)
  Method 3 – Dual Problem        (derive and solve the dual)
  Method 4 – PuLP (scipy-backed) (verification & general LP)

All methods provide step-by-step tableau traces.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — TABLEAU-BASED SIMPLEX (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

class SimplexSolver:
    """
    Solves standard-form LP:
        Maximize   c^T x
        Subject to Ax ≤ b,  x ≥ 0

    Internally adds slack variables to convert to equality form.
    """

    def __init__(self, c, A, b):
        """
        Parameters
        ----------
        c : list  — objective coefficients (maximization)
        A : 2-D list — constraint LHS coefficients
        b : list  — constraint RHS values (must be ≥ 0)
        """
        self.c_orig = np.array(c, dtype=float)
        self.A_orig = np.array(A, dtype=float)
        self.b_orig = np.array(b, dtype=float)
        self.m, self.n = self.A_orig.shape  # m constraints, n decision vars

    def _build_tableau(self):
        """
        Build the initial simplex tableau by adding slack variables.
        Tableau shape: (m+1) × (n + m + 1)
          Rows 0..m-1 : constraints
          Row m        : objective function (negated for maximization)
        """
        m, n = self.m, self.n
        # Slack variable identity block
        slack = np.eye(m)
        # RHS column
        rhs   = self.b_orig.reshape(-1, 1)
        # Constraint rows: [A | I | b]
        constraints = np.hstack([self.A_orig, slack, rhs])
        # Objective row: [-c | 0...0 | 0]  (negated because we look for row ≥ 0)
        obj_row = np.hstack([-self.c_orig, np.zeros(m), [0]])
        # Full tableau
        tableau = np.vstack([constraints, obj_row])
        return tableau

    def _col_labels(self):
        n, m = self.n, self.m
        decision = [f"x{i+1}" for i in range(n)]
        slacks   = [f"s{i+1}" for i in range(m)]
        return decision + slacks + ["RHS"]

    def _basis_labels(self, basis):
        labels = [f"x{i+1}" if i < self.n else f"s{i-self.n+1}" for i in basis]
        return labels

    def solve(self, var_names=None):
        """
        Run the simplex algorithm.

        Returns
        -------
        dict:
          solution    : {var: value}
          optimal     : float  (optimal objective value)
          tableaux    : list of DataFrames  (one per iteration)
          status      : str
        """
        m, n  = self.m, self.n
        T     = self._build_tableau()
        basis = list(range(n, n + m))   # slack variables are initial basis
        col_labels = self._col_labels()
        tableaux   = []

        if var_names is None:
            var_names = [f"x{i+1}" for i in range(n)]
        full_names = var_names + [f"s{i+1}" for i in range(m)] + ["RHS"]

        def snapshot(T, basis, label=""):
            df = pd.DataFrame(T, columns=full_names,
                              index=self._basis_labels(basis) + ["Z"])
            df.index.name = label
            return df.round(4)

        tableaux.append(snapshot(T, basis, "Initial Tableau"))

        for iteration in range(200):
            obj_row = T[-1, :-1]

            # Optimality check: all coefficients in obj row ≥ 0?
            if np.all(obj_row >= -1e-8):
                break

            # Pivot column = most negative coefficient in objective row
            pivot_col = int(np.argmin(obj_row))

            # Ratio test (minimum ratio rule) for pivot row
            ratios = []
            for i in range(m):
                if T[i, pivot_col] > 1e-8:
                    ratios.append(T[i, -1] / T[i, pivot_col])
                else:
                    ratios.append(np.inf)

            if all(r == np.inf for r in ratios):
                return {"status": "unbounded", "tableaux": tableaux}

            pivot_row = int(np.argmin(ratios))

            # Update basis
            basis[pivot_row] = pivot_col

            # Pivot operation (elementary row operations)
            T[pivot_row] /= T[pivot_row, pivot_col]          # normalize pivot row
            for i in range(T.shape[0]):
                if i != pivot_row:
                    T[i] -= T[i, pivot_col] * T[pivot_row]   # eliminate pivot col

            tableaux.append(snapshot(T, basis, f"Iteration {iteration+1}"))

        # Extract solution
        solution = {}
        for i, var_idx in enumerate(basis):
            if var_idx < len(var_names):
                solution[var_names[var_idx]] = round(T[i, -1], 6)
        for name in var_names:
            if name not in solution:
                solution[name] = 0.0

        optimal = round(T[-1, -1], 6)

        return {
            "status":   "optimal",
            "solution": solution,
            "optimal":  optimal,
            "tableaux": tableaux,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — BIG-M METHOD (handles ≥ and = constraints)
# ─────────────────────────────────────────────────────────────────────────────

class BigMSolver:
    """
    Solves general LP using the Big-M method.
    Supports ≤, ≥, and = constraints.
    Converts everything to standard equality form using slack/surplus/artificial.

    Maximize   c^T x
    Subject to constraints with types in ['<=', '>=', '=']
    """

    BIG_M = 1e6   # Penalty for artificial variables

    def __init__(self, c, A, b, constraint_types, var_names=None):
        self.c    = np.array(c, dtype=float)
        self.A    = np.array(A, dtype=float)
        self.b    = np.array(b, dtype=float)
        self.types = constraint_types
        self.n_orig = len(c)
        self.m      = len(b)
        self.var_names = var_names or [f"x{i+1}" for i in range(self.n_orig)]

    def _build_bigm_tableau(self):
        """
        Build tableau for Big-M:
          ≤ → add slack  (+s)
          ≥ → subtract surplus (-s) + add artificial (+a) with -M in objective
          =  → add artificial (+a) with -M in objective
        """
        M   = self.BIG_M
        m   = self.m
        rows = []        # tableau rows (constraint)
        obj  = list(-self.c)   # negated obj (we maximize → look for row ≥ 0)
        basis = []
        col_map = list(self.var_names)  # track column names

        n_slk = 0
        n_art = 0

        # Build each constraint row
        for i in range(m):
            row   = list(self.A[i])
            rhs   = self.b[i]
            ctype = self.types[i]

            # Extend all previous rows and obj for new variable
            # (handled after: we collect separately)
            rows.append((row, rhs, ctype, i))

        # Count columns needed
        slack_count = sum(1 for t in self.types if t == "<=")
        surplus_count = sum(1 for t in self.types if t == ">=")
        art_count = sum(1 for t in self.types if t in [">=", "="])

        total_vars = self.n_orig + slack_count + surplus_count + art_count
        T   = np.zeros((m + 1, total_vars + 1))   # +1 for RHS

        # Fill constraint rows
        art_obj_adjustment = []
        slk_idx = self.n_orig
        art_idx = self.n_orig + slack_count + surplus_count
        col_names = list(self.var_names)

        s_counter = 0
        sr_counter = 0
        a_counter = 0

        for i in range(m):
            row, rhs, ctype, _ = rows[i]
            T[i, :self.n_orig] = row
            T[i, -1] = rhs

            if ctype == "<=":
                T[i, self.n_orig + s_counter] = 1    # slack
                basis.append(self.n_orig + s_counter)
                s_counter += 1
            elif ctype == ">=":
                T[i, self.n_orig + s_counter] = -1   # surplus
                s_counter += 1
                a_col = self.n_orig + slack_count + surplus_count + a_counter
                T[i, a_col] = 1                       # artificial
                basis.append(a_col)
                art_obj_adjustment.append(a_col)
                a_counter += 1
            elif ctype == "=":
                a_col = self.n_orig + slack_count + surplus_count + a_counter
                T[i, a_col] = 1
                basis.append(a_col)
                art_obj_adjustment.append(a_col)
                a_counter += 1

        # Objective row: original coefficients (negated) + -M for artificials
        T[-1, :self.n_orig] = -self.c
        for a_col in art_obj_adjustment:
            T[-1, a_col] = M

        # Eliminate artificial variables from objective row (Big-M adjustment)
        for i, b_var in enumerate(basis):
            if b_var in art_obj_adjustment:
                T[-1] -= T[-1, b_var] * T[i]

        # Column names for display
        slack_names = [f"s{i+1}" for i in range(slack_count + surplus_count)]
        art_names   = [f"a{i+1}" for i in range(art_count)]
        all_col_names = list(self.var_names) + slack_names + art_names + ["RHS"]

        return T, basis, all_col_names, art_obj_adjustment

    def solve(self):
        T, basis, col_names, art_cols = self._build_bigm_tableau()
        m = self.m
        tableaux = []

        def snapshot(T, basis, label=""):
            b_labels = []
            for idx in basis:
                b_labels.append(col_names[idx] if idx < len(col_names)-1 else "?")
            df = pd.DataFrame(T, columns=col_names,
                              index=b_labels + ["Z"])
            df.index.name = label
            return df.round(4)

        tableaux.append(snapshot(T, basis, "Big-M Initial"))

        for iteration in range(300):
            obj_row = T[-1, :-1]
            if np.all(obj_row >= -1e-8):
                break

            pivot_col = int(np.argmin(obj_row))
            ratios = []
            for i in range(m):
                if T[i, pivot_col] > 1e-8:
                    ratios.append(T[i, -1] / T[i, pivot_col])
                else:
                    ratios.append(np.inf)

            if all(r == np.inf for r in ratios):
                return {"status": "unbounded", "tableaux": tableaux}

            pivot_row = int(np.argmin(ratios))
            basis[pivot_row] = pivot_col

            T[pivot_row] /= T[pivot_row, pivot_col]
            for i in range(T.shape[0]):
                if i != pivot_row:
                    T[i] -= T[i, pivot_col] * T[pivot_row]

            tableaux.append(snapshot(T, basis, f"Big-M Iter {iteration+1}"))

        # Check feasibility (any artificial in basis with nonzero value?)
        n_orig = self.n_orig
        solution = {}
        for i, var_idx in enumerate(basis):
            if var_idx < len(self.var_names):
                solution[self.var_names[var_idx]] = round(T[i, -1], 6)
        for name in self.var_names:
            if name not in solution:
                solution[name] = 0.0

        return {
            "status":   "optimal",
            "solution": solution,
            "optimal":  round(T[-1, -1], 6),
            "tableaux": tableaux,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DUALITY
# ─────────────────────────────────────────────────────────────────────────────

def derive_dual(c, A, b, constraint_types=None, var_names=None, constraint_names=None):
    """
    Derive the dual LP from the primal.

    Primal (maximization with ≤ constraints):
        max  c^T x     s.t. Ax ≤ b, x ≥ 0

    Dual (minimization):
        min  b^T y     s.t. A^T y ≥ c, y ≥ 0

    Returns
    -------
    dict with dual_c, dual_A, dual_b and human-readable description.
    """
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape

    dual_c = b          # dual objective coefficients
    dual_A = A.T        # dual constraint matrix
    dual_b = c          # dual RHS

    primal_vars      = var_names or [f"x{i+1}" for i in range(n)]
    dual_vars        = [f"y{j+1}" for j in range(m)]
    constraint_names = constraint_names or [f"C{i+1}" for i in range(m)]

    # Build human-readable
    lines = []
    lines.append("PRIMAL PROBLEM (Maximization):")
    lines.append(f"  Maximize Z = " + " + ".join(f"{c[i]}{primal_vars[i]}" for i in range(n)))
    lines.append("  Subject to:")
    for i in range(m):
        terms = " + ".join(f"{A[i][j]}{primal_vars[j]}" for j in range(n))
        lines.append(f"    {constraint_names[i]}: {terms} ≤ {b[i]}")
    lines.append(f"  {', '.join(primal_vars)} ≥ 0")

    lines.append("\nDUAL PROBLEM (Minimization):")
    lines.append(f"  Minimize W = " + " + ".join(f"{b[i]}{dual_vars[i]}" for i in range(m)))
    lines.append("  Subject to:")
    for j in range(n):
        terms = " + ".join(f"{A[i][j]}{dual_vars[i]}" for i in range(m))
        lines.append(f"    {primal_vars[j]}: {terms} ≥ {c[j]}")
    lines.append(f"  {', '.join(dual_vars)} ≥ 0")

    lines.append("\nDUALITY THEOREM: Optimal Primal Z* = Optimal Dual W*")

    return {
        "dual_c":        dual_c,
        "dual_A":        dual_A,
        "dual_b":        dual_b,
        "dual_vars":     dual_vars,
        "primal_vars":   primal_vars,
        "description":   "\n".join(lines),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PuLP SOLVER (verification)
# ─────────────────────────────────────────────────────────────────────────────

def solve_lp_pulp(c, A, b, constraint_types=None,
                  var_names=None, maximize=True, problem_name="LP"):
    """
    Solve LP using scipy.optimize.linprog as the verification solver.
    Supports ≤, ≥, and = constraints.

    Returns
    -------
    dict with solution, objective, status, constraint_slack
    """
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(c)
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n)]
    if constraint_types is None:
        constraint_types = ["<="] * len(b)

    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for row, rhs, ctype in zip(A, b, constraint_types):
        if ctype == "<=":
            A_ub.append(row)
            b_ub.append(rhs)
        elif ctype == ">=":
            A_ub.append(-row)
            b_ub.append(-rhs)
        elif ctype == "=":
            A_eq.append(row)
            b_eq.append(rhs)
        else:
            raise ValueError(f"Unsupported constraint type: {ctype}")

    objective = -c if maximize else c
    res = linprog(
        objective,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=[(0, None)] * n,
        method="highs",
    )

    if not res.success:
        return {
            "status":     res.message,
            "solution":   {name: None for name in var_names},
            "objective":  None,
            "solver":     "SciPy linprog",
            "problem":    problem_name,
        }

    solution = {var_names[i]: round(float(res.x[i]), 6) for i in range(n)}
    obj_val  = round(float(np.dot(c, res.x)), 6)

    return {
        "status":     "Optimal",
        "solution":   solution,
        "objective":  obj_val,
        "solver":     "SciPy linprog",
        "problem":    problem_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MASTER LP SOLVER (public API)
# ─────────────────────────────────────────────────────────────────────────────

def solve_lp(c, A, b, constraint_types=None, var_names=None,
             constraint_names=None, maximize=True):
    """
    Unified LP solver returning Simplex, Big-M, Dual derivation, and PuLP results.

    Parameters
    ----------
    c                : list  — objective coefficients
    A                : 2-D list — constraint coefficients
    b                : list  — RHS values
    constraint_types : list of '<=', '>=', '='  (default all '<=')
    var_names        : list of str
    constraint_names : list of str
    maximize         : bool (default True)

    Returns
    -------
    dict with simplex, bigm, dual, pulp sub-results
    """
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape

    if constraint_types is None:
        constraint_types = ["<="] * m
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n)]
    if constraint_names is None:
        constraint_names = [f"C{i+1}" for i in range(m)]

    results = {}

    # 1 – Standard Simplex (only for all ≤ problems)
    if all(t == "<=" for t in constraint_types):
        solver = SimplexSolver(c if maximize else -c, A, b)
        results["simplex"] = solver.solve(var_names=var_names)
        if not maximize and results["simplex"]["status"] == "optimal":
            results["simplex"]["optimal"] = -results["simplex"]["optimal"]
    else:
        results["simplex"] = {"status": "skipped",
                               "note": "Simplex requires all ≤ constraints"}

    # 2 – Big-M method
    bigm_c = c if maximize else -c
    bigm_solver = BigMSolver(bigm_c, A, b, constraint_types, var_names)
    results["bigm"] = bigm_solver.solve()
    if not maximize and results["bigm"]["status"] == "optimal":
        results["bigm"]["optimal"] = -results["bigm"]["optimal"]

    # 3 – Dual derivation (for all-≤ primal)
    if all(t == "<=" for t in constraint_types):
        results["dual"] = derive_dual(c, A, b, constraint_types,
                                      var_names, constraint_names)
    else:
        results["dual"] = {"description": "Dual derivation shown for standard ≤ form only."}

    # 4 – Independent verification
    results["pulp"] = solve_lp_pulp(
        c, A, b,
        constraint_types=constraint_types,
        var_names=var_names, maximize=maximize
    )

    results["var_names"]        = var_names
    results["constraint_names"] = constraint_names
    results["c"]                = c
    results["A"]                = A
    results["b"]                = b

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Maximize Z = 5x1 + 4x2
    # s.t.  6x1 + 4x2 ≤ 24
    #        x1 + 2x2 ≤  6
    #        x1, x2  ≥  0
    c = [5, 4]
    A = [[6, 4], [1, 2]]
    b = [24, 6]

    res = solve_lp(c, A, b, var_names=["x1","x2"])

    print(f"\n✅ Simplex Optimal: Z = {res['simplex']['optimal']}")
    print(f"   Solution: {res['simplex']['solution']}")
    print(f"✅ Big-M   Optimal: Z = {res['bigm']['optimal']}")
    print(f"   Solution: {res['bigm']['solution']}")
    print(f"✅ PuLP    Optimal: Z = {res['pulp']['objective']}")
    print(f"\n{res['dual']['description']}")
