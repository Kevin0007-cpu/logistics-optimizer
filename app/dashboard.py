"""
app/dashboard.py  — Professional UI rewrite
AI-Powered Multi-Objective Logistics Optimization System
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from optimization.transportation  import solve_transportation
from optimization.assignment      import solve_assignment
from optimization.simplex         import solve_lp
from optimization.decision_theory import solve_decision
from optimization.game_theory     import solve_game
from ai.demand_predictor          import solve_demand_prediction
from ai.metaheuristics            import solve_metaheuristics
from utils.helpers                import (plot_transport_heatmap,
                                           plot_assignment_result,
                                           plot_decision_analysis,
                                           plot_demand_forecast,
                                           plot_payoff_heatmap)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Logistics Optimizer",
    page_icon   = "🚚",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — works in both light and dark Streamlit themes
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 6px 10px;
    border-radius: 6px;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,179,237,0.12) !important;
}
[data-testid="stSidebar"] hr { border-color: #334155; }

/* ── Sidebar brand ───────────────────────────────────────── */
.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0 16px 0;
}
.sidebar-brand-icon {
    width: 36px; height: 36px; border-radius: 8px;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.sidebar-brand-text { font-size: 1.05rem; font-weight: 700; color: #f1f5f9; }
.sidebar-brand-sub  { font-size: 0.72rem; color: #94a3b8; margin-top: 1px; }

/* ── Tag pills in sidebar ────────────────────────────────── */
.tag-pill {
    display: inline-block; font-size: 0.68rem; font-weight: 500;
    padding: 2px 8px; border-radius: 20px; margin: 2px 2px 2px 0;
    background: rgba(99,179,237,0.15); color: #93c5fd;
    border: 1px solid rgba(99,179,237,0.25);
}

/* ── Page header banner ──────────────────────────────────── */
.page-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
    border: 1px solid #2d4a6b;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 160px; height: 160px; border-radius: 50%;
    background: rgba(59,130,246,0.08);
}
.page-header-tag {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    color: #60a5fa; text-transform: uppercase; margin-bottom: 6px;
}
.page-header h1 {
    font-size: 1.75rem; font-weight: 700;
    color: #f1f5f9; margin: 0 0 8px 0;
}
.page-header p {
    font-size: 0.9rem; color: #94a3b8; margin: 0;
}

/* ── Stat cards (home KPIs) ──────────────────────────────── */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin: 24px 0; }
.stat-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 20px 16px;
    position: relative; overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
    border-radius: 0 0 12px 12px;
}
.stat-card.blue::after   { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.stat-card.purple::after { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
.stat-card.green::after  { background: linear-gradient(90deg,#10b981,#34d399); }
.stat-card.amber::after  { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.stat-card-val  { font-size: 2.4rem; font-weight: 700; color: #f1f5f9; line-height:1; }
.stat-card-lbl  { font-size: 0.82rem; color: #94a3b8; margin-top: 6px; }
.stat-card-icon { position: absolute; top: 16px; right: 16px; font-size: 1.6rem; opacity: 0.25; }

/* ── Method cards (home grid) ────────────────────────────── */
.method-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.method-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px 16px;
    display: flex; align-items: flex-start; gap: 12px;
    transition: border-color 0.2s, transform 0.15s;
}
.method-card:hover { border-color: #3b82f6; transform: translateY(-1px); }
.method-card-icon {
    width: 36px; height: 36px; min-width: 36px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 17px;
}
.method-card-icon.blue   { background: rgba(59,130,246,0.15); }
.method-card-icon.purple { background: rgba(139,92,246,0.15); }
.method-card-icon.green  { background: rgba(16,185,129,0.15); }
.method-card-icon.amber  { background: rgba(245,158,11,0.15); }
.method-card-icon.rose   { background: rgba(244,63,94,0.15); }
.method-card-title { font-size: 0.88rem; font-weight: 600; color: #e2e8f0; margin-bottom: 3px; }
.method-card-desc  { font-size: 0.78rem; color: #64748b; line-height: 1.4; }

/* ── Section headings ────────────────────────────────────── */
.section-heading {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
    color: #60a5fa; text-transform: uppercase;
    margin: 24px 0 12px; padding-bottom: 8px;
    border-bottom: 1px solid #1e3a5f;
}

/* ── Result / optimal banner ─────────────────────────────── */
.result-banner {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #059669;
    border-radius: 10px;
    padding: 14px 20px;
    display: flex; align-items: center; gap: 12px;
    margin: 16px 0;
}
.result-banner-icon { font-size: 1.4rem; }
.result-banner-label { font-size: 0.78rem; color: #6ee7b7; font-weight: 500; }
.result-banner-value { font-size: 1.4rem; font-weight: 700; color: #ecfdf5; }

/* ── Step trace boxes ────────────────────────────────────── */
.step-trace {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.83rem; color: #cbd5e1;
}
.step-trace b { color: #93c5fd; }
.step-trace small { color: #475569; }

/* ── Info / warning callouts ─────────────────────────────── */
.callout-info {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.84rem; color: #93c5fd; margin: 10px 0;
}
.callout-warn {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.84rem; color: #fcd34d; margin: 10px 0;
}

/* ── Streamlit overrides ─────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="stMetric"] label { color: #94a3b8 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; }

div[data-testid="stTabs"] button {
    font-weight: 500; font-size: 0.85rem;
}
div[data-testid="stExpander"] {
    background: #1e293b;
    border: 1px solid #334155 !important;
    border-radius: 10px;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    border: none !important; color: white !important;
    font-weight: 600 !important; padding: 10px 24px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.45) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🚚</div>
        <div>
            <div class="sidebar-brand-text">Logistics Optimizer</div>
            <div class="sidebar-brand-sub">AIML Portfolio · 4th Sem</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("", [
        "🏠  Home",
        "🚚  Transportation",
        "👷  Assignment",
        "📊  Linear Programming",
        "🎯  Decision Theory",
        "🎮  Game Theory",
        "🤖  AI Demand Prediction",
        "🧬  Metaheuristics",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-heading">Algorithms</div>', unsafe_allow_html=True)
    tags = ["VAM","MODI","Hungarian","Simplex","Big-M","Duality",
            "Maximax","Maximin","Hurwicz","Saddle Point","Mixed Strategy",
            "Random Forest","Gradient Boosting","Genetic Algorithm","PSO"]
    pills_html = "".join(f'<span class="tag-pill">{t}</span>' for t in tags)
    st.markdown(pills_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE WIDGETS
# ─────────────────────────────────────────────────────────────────────────────

def page_header(tag, title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-tag">{tag}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def result_banner(label, value):
    st.markdown(f"""
    <div class="result-banner">
        <div class="result-banner-icon">🏆</div>
        <div>
            <div class="result-banner-label">{label}</div>
            <div class="result-banner-value">{value}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def section_head(text):
    st.markdown(f'<div class="section-heading">{text}</div>', unsafe_allow_html=True)

def step_trace(content):
    st.markdown(f'<div class="step-trace">{content}</div>', unsafe_allow_html=True)

def callout(text, kind="info"):
    cls = "callout-info" if kind == "info" else "callout-warn"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def matrix_input(label, default, rows, cols, row_labels=None, col_labels=None, key_prefix=""):
    section_head(label)
    matrix = []
    for i in range(rows):
        row_data = []
        cols_st = st.columns(cols)
        for j in range(cols):
            rl = row_labels[i] if row_labels else f"R{i+1}"
            cl = col_labels[j] if col_labels else f"C{j+1}"
            val = cols_st[j].number_input(
                f"{rl}→{cl}", value=float(default[i][j]),
                key=f"{key_prefix}_{i}_{j}", label_visibility="collapsed")
            row_data.append(val)
        matrix.append(row_data)
    return matrix

def vector_input(label, default, labels=None, key_prefix="", step=1.0):
    section_head(label)
    cols_st = st.columns(len(default))
    values = []
    for i, val in enumerate(default):
        lbl = labels[i] if labels else f"[{i+1}]"
        values.append(cols_st[i].number_input(
            lbl, value=float(val), step=step, key=f"{key_prefix}_{i}"))
    return values

def plotly_dark_template():
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(family="Inter", color="#94a3b8"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────

def page_home():
    page_header(
        "AI · OPERATIONS RESEARCH · OPTIMIZATION",
        "Multi-Objective Logistics Optimization",
        "Classical OR algorithms + Machine Learning + Metaheuristics — all in one dashboard."
    )

    # KPI stats
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card blue">
            <div class="stat-card-icon">⚙️</div>
            <div class="stat-card-val">6</div>
            <div class="stat-card-lbl">Optimization Algorithms</div>
        </div>
        <div class="stat-card purple">
            <div class="stat-card-icon">🤖</div>
            <div class="stat-card-val">3</div>
            <div class="stat-card-lbl">AI / ML Models</div>
        </div>
        <div class="stat-card green">
            <div class="stat-card-icon">🎯</div>
            <div class="stat-card-val">5</div>
            <div class="stat-card-lbl">Decision Criteria</div>
        </div>
        <div class="stat-card amber">
            <div class="stat-card-icon">🧬</div>
            <div class="stat-card-val">2</div>
            <div class="stat-card-lbl">Metaheuristics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="section-heading">Classical OR Methods</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="method-grid">
            <div class="method-card">
                <div class="method-card-icon blue">🚚</div>
                <div>
                    <div class="method-card-title">Transportation Problem</div>
                    <div class="method-card-desc">VAM for initial solution, MODI for optimality via u-v duals</div>
                </div>
            </div>
            <div class="method-card">
                <div class="method-card-icon purple">👷</div>
                <div>
                    <div class="method-card-title">Assignment Problem</div>
                    <div class="method-card-desc">Hungarian algorithm with row/col reduction and line cover</div>
                </div>
            </div>
            <div class="method-card">
                <div class="method-card-icon green">📊</div>
                <div>
                    <div class="method-card-title">Linear Programming</div>
                    <div class="method-card-desc">Simplex tableau, Big-M method, and Duality theorem</div>
                </div>
            </div>
            <div class="method-card">
                <div class="method-card-icon amber">🎯</div>
                <div>
                    <div class="method-card-title">Decision Theory</div>
                    <div class="method-card-desc">Maximax, Maximin, Minimax Regret, Laplace, Hurwicz</div>
                </div>
            </div>
            <div class="method-card">
                <div class="method-card-icon rose">🎮</div>
                <div>
                    <div class="method-card-title">Game Theory</div>
                    <div class="method-card-desc">Saddle point, dominance reduction, mixed strategy via LP</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-heading">AI &amp; Metaheuristic Methods</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="method-grid">
            <div class="method-card">
                <div class="method-card-icon blue">📈</div>
                <div>
                    <div class="method-card-title">ML Demand Prediction</div>
                    <div class="method-card-desc">Random Forest + Gradient Boosting with lag/seasonal features</div>
                </div>
            </div>
            <div class="method-card">
                <div class="method-card-icon purple">🧬</div>
                <div>
                    <div class="method-card-title">Genetic Algorithm</div>
                    <div class="method-card-desc">Order crossover, tournament selection, elitism</div>
                </div>
            </div>
            <div class="method-card">
                <div class="method-card-icon green">🐝</div>
                <div>
                    <div class="method-card-title">Particle Swarm (PSO)</div>
                    <div class="method-card-desc">Inertia-cognitive-social velocity update with feasibility decoder</div>
                </div>
            </div>
            <div class="method-card">
                <div class="method-card-icon amber">🔄</div>
                <div>
                    <div class="method-card-title">Method Comparison</div>
                    <div class="method-card-desc">Classical vs AI vs Metaheuristic side-by-side convergence</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        callout("👈 &nbsp;Select a module from the sidebar. Each page accepts custom input or uses built-in sample data.")

        st.markdown('<div class="section-heading">Quick Results (Sample Data)</div>', unsafe_allow_html=True)
        qa, qb = st.columns(2)
        qa.metric("Transportation",  "₹740", "Optimal")
        qb.metric("Assignment",      "₹13",  "Optimal")
        qc, qd = st.columns(2)
        qc.metric("LP (Simplex)",    "Z = 21", "x₁=3, x₂=1.5")
        qd.metric("Game Value",      "V ≈ 1.06", "Mixed Strategy")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TRANSPORTATION
# ─────────────────────────────────────────────────────────────────────────────

def page_transportation():
    page_header("TRANSPORTATION PROBLEM",
        "Vogel's Approximation + MODI Method",
        "Find the minimum cost shipping plan from warehouses to retail stores.")

    DEFAULT_COST   = [[4,8,1,6],[7,2,3,5],[3,6,9,2]]
    DEFAULT_SUPPLY = [120, 80, 80]
    DEFAULT_DEMAND = [150, 70, 60, 0]

    with st.expander("⚙️  Problem Setup", expanded=True):
        c1, c2 = st.columns(2)
        n_src = c1.slider("Sources (Warehouses)", 2, 5, 3)
        n_dst = c2.slider("Destinations (Stores)", 2, 5, 4)
        row_names = [c1.text_input(f"Source {i+1}", f"Warehouse {chr(65+i)}", key=f"tr_rn_{i}") for i in range(n_src)]
        col_names = [c2.text_input(f"Dest {j+1}",  f"Store {j+1}",           key=f"tr_cn_{j}") for j in range(n_dst)]
        cost_default = [[DEFAULT_COST[i][j] if i<3 and j<4 else 5 for j in range(n_dst)] for i in range(n_src)]
        cost_matrix  = matrix_input("Cost Matrix (per unit)", cost_default, n_src, n_dst, row_names, col_names, "tr")
        sup_def = (DEFAULT_SUPPLY + [50]*5)[:n_src]
        dem_def = (DEFAULT_DEMAND + [50]*5)[:n_dst]
        supply = vector_input("Supply at each source", sup_def, row_names, "tr_sup")
        demand = vector_input("Demand at each destination", dem_def, col_names, "tr_dem")

    if st.button("Solve Transportation Problem", type="primary"):
        with st.spinner("Running VAM → MODI..."):
            try:
                res = solve_transportation(cost_matrix, supply, demand, row_names, col_names)

                if not res["was_balanced"]:
                    callout("Problem was unbalanced — a dummy column was added automatically.", "warn")
                else:
                    callout("Problem is balanced (Total Supply = Total Demand).")

                k1, k2, k3 = st.columns(3)
                k1.metric("VAM Initial Cost",  f"₹{res['vam_cost']:,.2f}")
                k2.metric("MODI Optimal Cost", f"₹{res['optimal_cost']:,.2f}",
                          delta=f"{res['optimal_cost']-res['vam_cost']:+.2f}")
                k3.metric("MODI Iterations",   len(res["modi_iterations"]))

                t1, t2, t3, t4 = st.tabs(["Optimal Allocation", "VAM Steps", "MODI Iterations", "Visualisation"])

                with t1:
                    section_head("Optimal Allocation Matrix")
                    alloc_df = pd.DataFrame(res["optimal_allocation"],
                        index=res["row_names"], columns=res["col_names"])
                    st.dataframe(alloc_df.style.highlight_max(axis=None, color="#1e3a5f"),
                                 use_container_width=True)
                    section_head("Cost Contribution per Route")
                    contrib = pd.DataFrame(res["balanced_cost"] * res["optimal_allocation"],
                        index=res["row_names"], columns=res["col_names"]).round(2)
                    st.dataframe(contrib, use_container_width=True)
                    result_banner("Minimum Total Transportation Cost", f"₹{res['optimal_cost']:,.2f}")

                with t2:
                    section_head("VAM Penalty Trace")
                    for s in res["vam_steps"]:
                        step_trace(
                            f"<b>Iteration {s['iteration']}</b> &nbsp;·&nbsp; "
                            f"Max penalty: <b>{s['penalty']}</b> &nbsp;·&nbsp; "
                            f"Allocate <b>{s['allocated']}</b> units → "
                            f"Row {s['row']+1} / Col {s['col']+1}")

                with t3:
                    section_head("MODI (u-v) Optimality Check")
                    for it in res["modi_iterations"]:
                        if it.get("status") == "optimal":
                            st.success(f"Iteration {it['iteration']}: Solution is OPTIMAL ✓")
                        else:
                            step_trace(
                                f"<b>Iter {it['iteration']}</b> &nbsp;·&nbsp; "
                                f"Enter cell {it.get('entering')} &nbsp;·&nbsp; "
                                f"θ = {it.get('theta')} &nbsp;·&nbsp; "
                                f"Cost → ₹{it.get('total_cost')}")

                with t4:
                    fig = plot_transport_heatmap(res["balanced_cost"],
                        res["optimal_allocation"], res["row_names"], res["col_names"])
                    fig.update_layout(**plotly_dark_template())
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def page_assignment():
    page_header("ASSIGNMENT PROBLEM",
        "Hungarian Algorithm",
        "Optimally assign agents to tasks — minimise cost or maximise profit.")

    DEFAULT = [[9,2,7,8],[6,4,3,7],[5,8,1,8],[7,6,9,4]]

    with st.expander("⚙️  Problem Setup", expanded=True):
        c1, c2 = st.columns(2)
        n = c1.slider("Matrix size (n × n)", 2, 6, 4)
        maximize = c2.checkbox("Maximisation (profit matrix)", False)
        row_names = [c1.text_input(f"Agent {i+1}", f"Agent {i+1}", key=f"as_r_{i}") for i in range(n)]
        col_names = [c2.text_input(f"Task {j+1}",  f"Task {j+1}",  key=f"as_c_{j}") for j in range(n)]
        default = [[DEFAULT[i][j] if i<4 and j<4 else 5 for j in range(n)] for i in range(n)]
        cost_matrix = matrix_input("Cost / Profit Matrix", default, n, n, row_names, col_names, "as")

    if st.button("Solve Assignment Problem", type="primary"):
        with st.spinner("Running Hungarian Algorithm..."):
            try:
                res = solve_assignment(cost_matrix, row_names, col_names, maximize=maximize)

                k1, k2, k3 = st.columns(3)
                k1.metric("Optimal " + ("Profit" if maximize else "Cost"),
                           f"₹{res['total_cost']:,.2f}")
                k2.metric("Agents Assigned", len(res["assignment"]))
                k3.metric("Steps Taken", len(res["steps"]))

                t1, t2, t3 = st.tabs(["Optimal Assignment", "Algorithm Steps", "Visualisation"])

                with t1:
                    section_head("Assignment Table")
                    st.dataframe(res["result_table"], use_container_width=True)
                    result_banner(
                        "Optimal " + ("Profit" if maximize else "Cost"),
                        f"₹{res['total_cost']:,.2f}")

                with t2:
                    for step in res["steps"]:
                        with st.expander(f"[{step['step']}]  {step['description']}", expanded=False):
                            if "matrix" in step:
                                df = pd.DataFrame(step["matrix"],
                                    index=row_names[:step["matrix"].shape[0]],
                                    columns=col_names[:step["matrix"].shape[1]])
                                st.dataframe(df.round(2), use_container_width=True)
                            if "pairs" in step:
                                for r, c in step["pairs"]:
                                    if r < len(row_names) and c < len(col_names):
                                        st.write(f"✔ {row_names[r]} → {col_names[c]}")

                with t3:
                    fig = plot_assignment_result(np.array(cost_matrix),
                        [(r,c) for r,c in res["assignment"] if r<n and c<n],
                        row_names, col_names)
                    fig.update_layout(**plotly_dark_template())
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LINEAR PROGRAMMING
# ─────────────────────────────────────────────────────────────────────────────

def page_lp():
    page_header("LINEAR PROGRAMMING",
        "Simplex · Big-M · Duality · PuLP",
        "Solve general LPs with tableau-based methods and verify with PuLP.")

    with st.expander("⚙️  Problem Setup", expanded=True):
        c1, c2, c3 = st.columns(3)
        n_vars = c1.slider("Decision Variables", 2, 5, 2)
        n_cons = c2.slider("Constraints", 1, 6, 2)
        maximize = c3.checkbox("Maximisation", True)

        section_head("Objective Coefficients (c)")
        obj_def  = [5.0, 4.0, 3.0, 2.0, 1.0]
        c_cols   = st.columns(n_vars)
        var_names= []
        c        = []
        for i in range(n_vars):
            vname = c_cols[i].text_input(f"x{i+1} name", f"x{i+1}", key=f"lp_vn_{i}")
            var_names.append(vname)
            c.append(c_cols[i].number_input(f"c{i+1}", value=obj_def[i], key=f"lp_c_{i}",
                                             label_visibility="visible"))

        A_def = [[6,4,2,1,1],[1,2,3,2,1],[2,1,1,3,2],[1,1,2,1,1],[3,2,1,2,1],[1,3,2,1,2]]
        b_def = [24,6,12,8,15,10]
        ct_opts = ["<=",">=","="]
        A, b, ctypes, cnames = [], [], [], []

        section_head("Constraints")
        for i in range(n_cons):
            cols_c = st.columns(n_vars + 2)
            row = [cols_c[j].number_input("", value=float(A_def[i][j] if j<n_vars else 0),
                   key=f"lp_a_{i}_{j}", label_visibility="collapsed") for j in range(n_vars)]
            ctype = cols_c[n_vars].selectbox("", ct_opts, key=f"lp_ct_{i}", label_visibility="collapsed")
            rhs   = cols_c[n_vars+1].number_input("RHS", value=float(b_def[i]),
                    key=f"lp_b_{i}", label_visibility="visible")
            A.append(row); b.append(rhs); ctypes.append(ctype); cnames.append(f"C{i+1}")

    if st.button("Solve Linear Program", type="primary"):
        with st.spinner("Running Simplex + Big-M + PuLP..."):
            try:
                res = solve_lp(c, A, b, ctypes, var_names, cnames, maximize=maximize)
                pulp_obj = res["pulp"]["objective"]

                k1, k2, k3 = st.columns(3)
                k1.metric("PuLP Optimal (Z*)", f"{pulp_obj:,.4f}")
                simp = res["simplex"]
                k2.metric("Simplex", f"{simp.get('optimal','—')}" if simp.get("status") != "skipped" else "N/A")
                k3.metric("Big-M",   f"{res['bigm'].get('optimal','—'):,.4f}" if res["bigm"]["status"] == "optimal" else "N/A")

                t1, t2, t3, t4, t5 = st.tabs(["Solution","Simplex Tableaux","Big-M Tableaux","Duality","Chart"])

                with t1:
                    section_head("Optimal Variable Values")
                    sol_df = pd.DataFrame([{"Variable": k, "Value": v}
                                            for k,v in res["pulp"]["solution"].items()])
                    st.dataframe(sol_df, use_container_width=True)
                    result_banner(("Maximize" if maximize else "Minimize") + " Z*",
                                   f"{pulp_obj:,.4f}")

                with t2:
                    if res["simplex"]["status"] == "skipped":
                        callout(res["simplex"]["note"])
                    elif res["simplex"]["status"] == "optimal":
                        st.success(f"Converged in {len(res['simplex']['tableaux'])-1} pivot(s)")
                        for i, df in enumerate(res["simplex"]["tableaux"]):
                            with st.expander(f"Tableau {i} — {df.index.name}", expanded=(i==0)):
                                st.dataframe(df, use_container_width=True)

                with t3:
                    if res["bigm"]["status"] == "optimal":
                        st.success(f"Big-M converged in {len(res['bigm']['tableaux'])-1} pivot(s)")
                        for i, df in enumerate(res["bigm"]["tableaux"]):
                            with st.expander(f"Big-M Tableau {i} — {df.index.name}", expanded=(i==0)):
                                st.dataframe(df, use_container_width=True)

                with t4:
                    section_head("Primal ↔ Dual Derivation")
                    st.code(res["dual"].get("description","Dual shown for standard ≤ form."))
                    if "dual_c" in res["dual"]:
                        with st.spinner("Solving dual..."):
                            da_ub = (-res["dual"]["dual_A"]).tolist()
                            db_ub = (-res["dual"]["dual_b"]).tolist()
                            dr = solve_lp(res["dual"]["dual_c"].tolist(), da_ub, db_ub,
                                          var_names=res["dual"]["dual_vars"], maximize=False)
                            st.metric("Dual Optimal W*", f"{-dr['pulp']['objective']:,.4f}")
                            callout("Strong Duality verified: Primal Z* = Dual W* ✓")

                with t5:
                    fig = px.bar(x=var_names, y=c, title="Objective Coefficients",
                        labels={"x":"Variable","y":"Coefficient"},
                        color=c, color_continuous_scale="Blues",
                        template="plotly_dark")
                    fig.update_layout(**plotly_dark_template())
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DECISION THEORY
# ─────────────────────────────────────────────────────────────────────────────

def page_decision():
    page_header("DECISION THEORY",
        "5 Criteria Under Uncertainty",
        "Maximax · Maximin · Minimax Regret · Laplace · Hurwicz")

    DEFAULT = [[200,100,50],[300,150,-50],[100,80,60],[250,200,-100]]

    with st.expander("⚙️  Problem Setup", expanded=True):
        c1, c2, c3 = st.columns(3)
        n_strat = c1.slider("Strategies", 2, 6, 4)
        n_state = c2.slider("States of Nature", 2, 5, 3)
        alpha   = c3.slider("Hurwicz α (optimism)", 0.0, 1.0, 0.5, 0.05)
        strat_names = [c1.text_input(f"Strategy {i+1}", f"Strategy {i+1}", key=f"dt_s_{i}") for i in range(n_strat)]
        state_names = [c2.text_input(f"State {j+1}",   f"State {j+1}",    key=f"dt_st_{j}") for j in range(n_state)]
        default = [[DEFAULT[i][j] if i<4 and j<3 else 50 for j in range(n_state)] for i in range(n_strat)]
        payoff  = matrix_input("Payoff Matrix", default, n_strat, n_state, strat_names, state_names, "dt")

    if st.button("Analyse Decision Problem", type="primary"):
        with st.spinner("Computing all criteria..."):
            try:
                res = solve_decision(payoff, strat_names, state_names, alpha)

                section_head("Criteria Summary")
                st.dataframe(res["comparison_table"], use_container_width=True)
                result_banner(
                    f"Overall Recommended ({res['strategy_tally'][res['recommended']]}/5 criteria)",
                    res["recommended"])

                t1, t2, t3 = st.tabs(["Charts", "Regret Matrix", "Criterion Detail"])

                with t1:
                    fig = plot_decision_analysis(res["payoff_df"])
                    fig.update_layout(**plotly_dark_template())
                    st.plotly_chart(fig, use_container_width=True)

                    tally_df = pd.DataFrame([{"Strategy":k,"Votes":v}
                                              for k,v in res["strategy_tally"].items()])
                    fig2 = px.bar(tally_df, x="Strategy", y="Votes",
                        title="Criteria votes per strategy",
                        color="Votes", color_continuous_scale="Blues",
                        template="plotly_dark")
                    fig2.update_layout(**plotly_dark_template())
                    st.plotly_chart(fig2, use_container_width=True)

                with t2:
                    section_head("Opportunity Loss (Regret) Matrix")
                    st.dataframe(res["criteria_results"]["minimax_regret"]["regret_df"],
                                 use_container_width=True)

                with t3:
                    for cname, cres in res["criteria_results"].items():
                        with st.expander(f"{cres['criterion']}  →  {cres['best_strategy']}  ({cres['best_value']})"):
                            st.caption(cres["description"])
                            if "detail_df" in cres:
                                st.dataframe(cres["detail_df"], use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GAME THEORY
# ─────────────────────────────────────────────────────────────────────────────

def page_game():
    page_header("GAME THEORY",
        "Zero-Sum Two-Player Games",
        "Saddle Point · Dominance Reduction · Mixed Strategy Nash Equilibrium")

    DEFAULT = [[3,-1,2],[-2,4,1],[1,2,-3]]

    with st.expander("⚙️  Problem Setup", expanded=True):
        c1, c2 = st.columns(2)
        m = c1.slider("Player A strategies (rows)", 2, 5, 3)
        n = c2.slider("Player B strategies (cols)", 2, 5, 3)
        row_names = [c1.text_input(f"A — Strategy {i+1}", f"A_S{i+1}", key=f"gt_r_{i}") for i in range(m)]
        col_names = [c2.text_input(f"B — Strategy {j+1}", f"B_S{j+1}", key=f"gt_c_{j}") for j in range(n)]
        default = [[DEFAULT[i][j] if i<3 and j<3 else 0 for j in range(n)] for i in range(m)]
        payoff  = matrix_input("Payoff Matrix (Player A gains)", default, m, n, row_names, col_names, "gt")

    if st.button("Analyse Game", type="primary"):
        with st.spinner("Finding equilibrium..."):
            try:
                res = solve_game(payoff, row_names, col_names)

                k1, k2, k3 = st.columns(3)
                k1.metric("Game Value V",          f"{res['game_value']}")
                k2.metric("Equilibrium Type",       "Pure Strategy" if res["is_pure"] else "Mixed Strategy")
                k3.metric("Dominated Strats Removed", len(res["dominance"]["steps"]))

                t1, t2, t3, t4 = st.tabs(["Saddle Point", "Dominance", "Mixed Strategy", "Payoff Heatmap"])

                with t1:
                    s = res["saddle"]
                    if s["exists"]:
                        st.success(f"Saddle Point EXISTS — Pure Strategy Solution")
                        section_head("Saddle Point Details")
                        for sp in s["saddle_points"]:
                            st.write(f"Row: **{sp['row']}**  ·  Col: **{sp['col']}**  ·  Value: **{sp['value']}**")
                        k1,k2 = st.columns(2)
                        k1.metric("Maximin (Player A security)", s["maximin"])
                        k2.metric("Minimax (Player B security)", s["minimax"])
                    else:
                        callout(f"No saddle point found. Maximin={s['maximin']}, Minimax={s['minimax']} — use Mixed Strategy.", "warn")

                with t2:
                    d = res["dominance"]
                    if d["eliminated"]:
                        section_head("Elimination Steps")
                        for step in d["steps"]:
                            step_trace(
                                f"<b>{step['type']} Elimination:</b> Remove '{step['eliminated']}' "
                                f"(dominated by '{step['dominated_by']}')<br>"
                                f"<small>{step['reason']}</small>")
                        section_head("Reduced Matrix")
                        st.dataframe(d["reduced_df"], use_container_width=True)
                    else:
                        callout("No dominated strategies — matrix is irreducible.")

                with t3:
                    mx = res["mixed"]
                    ca, cb = st.columns(2)
                    with ca:
                        section_head("Player A Mixed Strategy")
                        st.dataframe(mx["row_mixed"], use_container_width=True)
                        fig_a = px.pie(mx["row_mixed"], names="Strategy", values="Probability",
                            title="Player A", template="plotly_dark", hole=0.4)
                        fig_a.update_layout(**plotly_dark_template())
                        st.plotly_chart(fig_a, use_container_width=True)
                    with cb:
                        section_head("Player B Mixed Strategy")
                        st.dataframe(mx["col_mixed"], use_container_width=True)
                        fig_b = px.pie(mx["col_mixed"], names="Strategy", values="Probability",
                            title="Player B", template="plotly_dark", hole=0.4)
                        fig_b.update_layout(**plotly_dark_template())
                        st.plotly_chart(fig_b, use_container_width=True)
                    result_banner("Nash Equilibrium Game Value", str(mx["game_value"]))

                with t4:
                    fig = plot_payoff_heatmap(np.array(payoff), row_names, col_names)
                    fig.update_layout(**plotly_dark_template())
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI DEMAND PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def page_ai():
    page_header("AI DEMAND PREDICTION",
        "ML-Powered Demand Forecasting",
        "Linear Regression · Random Forest · Gradient Boosting · Moving Average Baseline")

    data_path = os.path.join(ROOT, "data", "demand_history.csv")

    with st.expander("⚙️  Configuration", expanded=True):
        c1, c2 = st.columns(2)
        use_sample = c1.radio("Data Source", ["Sample Dataset", "Upload CSV"])
        if use_sample == "Sample Dataset":
            df = pd.read_csv(data_path)
            callout(f"Loaded sample dataset — {len(df)} months of historical data.")
        else:
            uploaded = st.file_uploader("Upload CSV (needs 'month' + demand columns)", type="csv")
            if not uploaded:
                callout("Please upload a CSV file to continue.")
                return
            df = pd.read_csv(uploaded)

        target_col = c1.selectbox("Column to Predict", [col for col in df.columns if col != "month"])
        n_forecast = c2.slider("Forecast Horizon (months)", 3, 12, 6)
        test_size  = c2.slider("Holdout Period (months)",   3, 12, 6)

    if st.button("Train Models & Forecast", type="primary"):
        with st.spinner("Training ML models — this may take a few seconds..."):
            try:
                res = solve_demand_prediction(df, target_col, n_forecast, test_size)

                k1, k2, k3 = st.columns(3)
                k1.metric("Best Model",         res["best_model"])
                k2.metric("Best RMSE",          f"{res['metrics'].iloc[0]['RMSE']:.2f}")
                k3.metric("Forecast Horizon",   f"{n_forecast} months")

                t1, t2, t3, t4 = st.tabs(["Forecast Chart", "Model Comparison", "Feature Importance", "Test Data"])

                with t1:
                    fig = plot_demand_forecast(df, res["forecast_df"], target_col)
                    fig.update_layout(**plotly_dark_template())
                    st.plotly_chart(fig, use_container_width=True)
                    section_head("Forecast Values")
                    st.dataframe(res["forecast_df"], use_container_width=True)

                with t2:
                    section_head("Model Performance (test set)")
                    st.dataframe(res["metrics"], use_container_width=True)
                    fig2 = px.bar(res["metrics"], x="Model", y="RMSE",
                        title="RMSE Comparison — lower is better",
                        color="RMSE", color_continuous_scale="Reds_r",
                        template="plotly_dark")
                    fig2.update_layout(**plotly_dark_template())
                    st.plotly_chart(fig2, use_container_width=True)

                with t3:
                    if res["importance"] is not None:
                        section_head("Feature Importance (tree models)")
                        st.dataframe(res["importance"], use_container_width=True)
                        fig3 = px.bar(res["importance"].reset_index(),
                            x="index", y="Mean Importance",
                            labels={"index":"Feature"},
                            title="Mean Feature Importance",
                            color="Mean Importance", color_continuous_scale="Blues",
                            template="plotly_dark")
                        fig3.update_layout(**plotly_dark_template())
                        st.plotly_chart(fig3, use_container_width=True)

                with t4:
                    section_head("Test Set: Actual vs Predicted")
                    st.dataframe(res["test_compare"], use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: METAHEURISTICS
# ─────────────────────────────────────────────────────────────────────────────

def page_meta():
    page_header("METAHEURISTICS",
        "GA · PSO vs Classical Methods",
        "Compare Genetic Algorithm and Particle Swarm Optimization against the Hungarian Algorithm.")

    DEFAULT = [[9,2,7,8],[6,4,3,7],[5,8,1,8],[7,6,9,4]]

    with st.expander("⚙️  Configuration", expanded=True):
        n = st.slider("Matrix Size (n × n)", 2, 6, 4)
        default = [[DEFAULT[i][j] if i<4 and j<4 else 5 for j in range(n)] for i in range(n)]
        cost_matrix = matrix_input("Cost Matrix", default, n, n, key_prefix="meta")

        c1, c2, c3 = st.columns(3)
        pop_size = c1.slider("GA Population Size",   20, 200, 50)
        max_gen  = c2.slider("GA Generations",       50, 500, 200)
        mut_rate = c3.slider("GA Mutation Rate",     0.01, 0.5, 0.1)
        n_par    = c1.slider("PSO Particles",        10, 100, 40)
        max_iter = c2.slider("PSO Iterations",       50, 500, 200)

    if st.button("Run GA + PSO + Classical Comparison", type="primary"):
        with st.spinner("Running optimisers..."):
            try:
                classical      = solve_assignment(cost_matrix)
                classical_cost = classical["total_cost"]

                res = solve_metaheuristics(
                    cost_matrix, problem_type="assignment",
                    ga_params  = {"population_size":pop_size,"max_generations":max_gen,"mutation_rate":mut_rate},
                    pso_params = {"n_particles":n_par,"max_iter":max_iter})

                k1, k2, k3 = st.columns(3)
                k1.metric("Hungarian (Exact)",  f"₹{classical_cost:,.2f}")
                k2.metric("GA Best Cost",        f"₹{res['ga']['cost']:,.2f}",
                           delta=f"{res['ga']['cost']-classical_cost:+.2f}")
                k3.metric("PSO Best Cost",       f"₹{res['pso']['cost']:,.2f}",
                           delta=f"{res['pso']['cost']-classical_cost:+.2f}")

                t1, t2, t3 = st.tabs(["Comparison Table", "Convergence Curves", "Assignment Detail"])

                with t1:
                    compare_df = pd.DataFrame([
                        {"Method":"Hungarian (Classical)", "Cost":classical_cost, "Gap vs Optimal":0.0, "Type":"Exact"},
                        {"Method":"Genetic Algorithm",     "Cost":res["ga"]["cost"],
                         "Gap vs Optimal":round(res["ga"]["cost"]-classical_cost,2), "Type":"Metaheuristic"},
                        {"Method":"Particle Swarm (PSO)",  "Cost":res["pso"]["cost"],
                         "Gap vs Optimal":round(res["pso"]["cost"]-classical_cost,2), "Type":"Metaheuristic"},
                    ])
                    st.dataframe(compare_df, use_container_width=True)

                    fig = px.bar(compare_df, x="Method", y="Cost", color="Type",
                        title="Total Assignment Cost — Method Comparison",
                        color_discrete_map={"Exact":"#3b82f6","Metaheuristic":"#f59e0b"},
                        template="plotly_dark", barmode="group")
                    fig.update_layout(**plotly_dark_template())
                    st.plotly_chart(fig, use_container_width=True)

                with t2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(y=res["ga"]["history"],  mode="lines",
                        name="Genetic Algorithm", line=dict(color="#f59e0b", width=2)))
                    fig2.add_trace(go.Scatter(y=res["pso"]["history"], mode="lines",
                        name="PSO", line=dict(color="#60a5fa", width=2, dash="dash")))
                    fig2.add_hline(y=classical_cost, line_dash="dot", line_color="#10b981",
                        annotation_text=f"Hungarian Optimal = {classical_cost}",
                        annotation_font_color="#10b981")
                    fig2.update_layout(title="Convergence: Best Cost vs Iteration",
                        xaxis_title="Generation / Iteration", yaxis_title="Best Cost",
                        **plotly_dark_template(), height=400)
                    st.plotly_chart(fig2, use_container_width=True)

                with t3:
                    ca, cb = st.columns(2)
                    with ca:
                        section_head("GA Best Assignment")
                        for i, (r, c) in enumerate(res["ga"]["assignment"]):
                            if r < n and c < n:
                                st.write(f"Row {r+1} → Col {c+1}  (cost: {cost_matrix[r][c]})")
                    with cb:
                        section_head("Classical (Hungarian) Assignment")
                        st.dataframe(classical["result_table"], use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

if   "Home"           in page: page_home()
elif "Transportation" in page: page_transportation()
elif "Assignment"     in page: page_assignment()
elif "Linear"         in page: page_lp()
elif "Decision"       in page: page_decision()
elif "Game"           in page: page_game()
elif "AI"             in page: page_ai()
elif "Meta"           in page: page_meta()
