"""
utils/helpers.py
Shared utility functions: printing, validation, Plotly charts.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
 
 
# ── Pretty Print ─────────────────────────────────────────────────────────────
def print_matrix(matrix, row_labels=None, col_labels=None, title="Matrix"):
    print(f"\n{'═'*55}\n  {title}\n{'═'*55}")
    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    print(df.to_string())
    print(f"{'─'*55}\n")
 
def matrix_to_df(matrix, row_prefix="Row", col_prefix="Col"):
    rows = [f"{row_prefix}_{i+1}" for i in range(matrix.shape[0])]
    cols = [f"{col_prefix}_{j+1}" for j in range(matrix.shape[1])]
    return pd.DataFrame(matrix, index=rows, columns=cols)
 
 
# ── Validation ────────────────────────────────────────────────────────────────
def validate_transport(cost, supply, demand):
    ts, td = sum(supply), sum(demand)
    diff = ts - td
    if diff == 0:   return True, "balanced", 0
    elif diff > 0:  return False, "excess_supply", diff
    else:           return False, "excess_demand", abs(diff)
 
def validate_square_matrix(matrix):
    r, c = np.array(matrix).shape
    if r != c:
        raise ValueError(f"Matrix must be square, got ({r}×{c})")
    return True
 
 
# ── Plotly Charts ─────────────────────────────────────────────────────────────
def plot_transport_heatmap(cost_matrix, allocation_matrix, row_labels, col_labels):
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Cost Matrix", "Optimal Allocation"])
    fig.add_trace(go.Heatmap(z=cost_matrix, x=col_labels, y=row_labels,
        colorscale="Blues", text=cost_matrix, texttemplate="%{text}",
        showscale=False, name="Cost"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=allocation_matrix, x=col_labels, y=row_labels,
        colorscale="Greens", text=allocation_matrix, texttemplate="%{text}",
        showscale=False, name="Allocation"), row=1, col=2)
    fig.update_layout(title="Transportation: Cost vs Allocation",
                      height=380, template="plotly_white")
    return fig
 
def plot_assignment_result(cost_matrix, assignment, row_labels, col_labels):
    mask = np.zeros_like(cost_matrix, dtype=float)
    for r, c in assignment:
        mask[r][c] = 1
    annotations = []
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            marker = " ✔" if mask[i][j] else ""
            annotations.append(dict(
                x=col_labels[j], y=row_labels[i],
                text=f"{cost_matrix[i][j]}{marker}", showarrow=False,
                font=dict(color="white" if mask[i][j] else "black",
                          size=13 if mask[i][j] else 11)))
    fig = go.Figure(data=go.Heatmap(
        z=mask, x=col_labels, y=row_labels,
        colorscale=[[0,"#e8f4f8"],[1,"#1a6fa0"]], showscale=False))
    fig.update_layout(annotations=annotations,
        title="Assignment Result (✔ = assigned)", height=380, template="plotly_white")
    return fig
 
def plot_decision_analysis(decision_df):
    scores = pd.DataFrame({
        "Maximax (best of best)": decision_df.max(axis=1),
        "Minimax (best of worst)": decision_df.min(axis=1),
        "Laplace (average)": decision_df.mean(axis=1),
        "Minimax Regret": (decision_df.max().max() - decision_df).max(axis=1),
    })
    fig = px.bar(scores, barmode="group",
        title="Decision Theory – Strategy Scores by Criterion",
        labels={"value":"Payoff / Regret","index":"Strategy"},
        template="plotly_white")
    return fig
 
def plot_demand_forecast(history_df, forecast_df, target_col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df["month"], y=history_df[target_col],
        mode="lines+markers", name="Historical", line=dict(color="#1a6fa0")))
    fig.add_trace(go.Scatter(x=forecast_df["month"], y=forecast_df["predicted"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#e8522e", dash="dash")))
    fig.update_layout(title=f"Demand Forecast – {target_col}",
        xaxis_title="Month", yaxis_title="Units",
        template="plotly_white", height=380)
    return fig
 
def plot_comparison_bar(labels, values, title="Method Comparison"):
    fig = px.bar(x=values, y=labels, orientation="h",
        title=title, template="plotly_white",
        labels={"x":"Total Cost","y":"Method"},
        color=values, color_continuous_scale="Blues_r")
    fig.update_layout(height=320, showlegend=False)
    return fig
 
def plot_payoff_heatmap(payoff_matrix, row_labels, col_labels):
    fig = go.Figure(data=go.Heatmap(
        z=payoff_matrix, x=col_labels, y=row_labels,
        colorscale="RdYlGn", text=payoff_matrix, texttemplate="%{text}",
        showscale=True))
    fig.update_layout(title="Game Theory Payoff Matrix (Player A perspective)",
        height=380, template="plotly_white")
    return fig
 
def plot_simplex_sensitivity(obj_coeffs, var_names):
    fig = px.bar(x=var_names, y=obj_coeffs,
        title="LP Objective Coefficients",
        labels={"x":"Variable","y":"Coefficient"},
        template="plotly_white", color=obj_coeffs,
        color_continuous_scale="Blues")
    return fig
 
def summarize_result(method_name, cost, details=None):
    return {"Method": method_name, "Total Cost / Value": round(cost, 4),
            "Details": details or {}}
 
def highlight_min(s):
    return ["background-color:#d4edda" if v else "" for v in s==s.min()]
 
def highlight_max(s):
    return ["background-color:#fff3cd" if v else "" for v in s==s.max()]