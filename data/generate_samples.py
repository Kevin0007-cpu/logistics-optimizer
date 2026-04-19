"""
data/generate_samples.py
Generates all sample datasets used by the system.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
DATA_DIR = Path(__file__).resolve().parent

# ── Transportation Problem ──────────────────────────────────────────────────
transport_cost = pd.DataFrame(
    [[4, 8, 1, 6],
     [7, 2, 3, 5],
     [3, 6, 9, 2]],
    index=["Warehouse_A", "Warehouse_B", "Warehouse_C"],
    columns=["Store_1", "Store_2", "Store_3", "Store_4"]
)
supply = pd.Series([120, 80, 80], index=["Warehouse_A","Warehouse_B","Warehouse_C"])
demand = pd.Series([150, 70, 60, 0], index=["Store_1","Store_2","Store_3","Store_4"])
# balance
if supply.sum() > demand.sum():
    demand["Store_4"] = supply.sum() - demand[:3].sum()
transport_cost.to_csv(DATA_DIR / "transport_cost.csv")
supply.to_csv(DATA_DIR / "transport_supply.csv", header=["Supply"])
demand.to_csv(DATA_DIR / "transport_demand.csv", header=["Demand"])

# ── Assignment Problem ──────────────────────────────────────────────────────
assign_cost = pd.DataFrame(
    [[9, 2, 7, 8],
     [6, 4, 3, 7],
     [5, 8, 1, 8],
     [7, 6, 9, 4]],
    index=["Worker_1","Worker_2","Worker_3","Worker_4"],
    columns=["Task_A","Task_B","Task_C","Task_D"]
)
assign_cost.to_csv(DATA_DIR / "assignment_cost.csv")

# ── Decision Theory ─────────────────────────────────────────────────────────
decision_matrix = pd.DataFrame(
    [[200, 100,  50],
     [300, 150, -50],
     [100,  80,  60],
     [250, 200,-100]],
    index=["Strategy_1","Strategy_2","Strategy_3","Strategy_4"],
    columns=["High_Demand","Medium_Demand","Low_Demand"]
)
decision_matrix.to_csv(DATA_DIR / "decision_matrix.csv")

# ── Game Theory ─────────────────────────────────────────────────────────────
payoff_matrix = pd.DataFrame(
    [[ 3,-1, 2],
     [-2, 4, 1],
     [ 1, 2,-3]],
    index=["A_Strat_1","A_Strat_2","A_Strat_3"],
    columns=["B_Strat_1","B_Strat_2","B_Strat_3"]
)
payoff_matrix.to_csv(DATA_DIR / "payoff_matrix.csv")

# ── Demand History (ML) ─────────────────────────────────────────────────────
months = pd.date_range(start="2022-01-01", periods=36, freq="MS")
trend  = np.linspace(200, 350, 36)
seas   = 40 * np.sin(2*np.pi*np.arange(36)/12)
noise  = np.random.normal(0, 15, 36)
demand_hist = pd.DataFrame({
    "month": months,
    "product_A_demand": (trend + seas + noise).astype(int),
    "product_B_demand": (trend*0.7 + seas*0.5 + noise).astype(int),
    "price": np.random.uniform(10,20,36).round(2),
    "promotion": np.random.choice([0,1],36,p=[0.7,0.3]),
    "competitor_price": np.random.uniform(12,22,36).round(2),
})
demand_hist.to_csv(DATA_DIR / "demand_history.csv", index=False)

print("All datasets generated.")
