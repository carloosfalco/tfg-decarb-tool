# app.py
# Industrial Decarbonization Decision Tool (TFG prototype)
# Author: Carlos Falcó
# Streamlit app: upload initiatives CSV -> compute metrics -> optimize portfolio (PuLP) -> export results

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# PuLP (optimization)
import pulp


# -----------------------------
# Config / UI
# -----------------------------
st.set_page_config(
    page_title="Industrial Decarbonization Decision Tool",
    page_icon="🌿",
    layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------
REQUIRED_COLUMNS = [
    "id",
    "initiative",
    "capex_eur",
    "annual_opex_saving_eur",
    "annual_co2_reduction_t",
    "implementation_months",
    "strategic_score_1_5",
    "notes",
]

OPTIONAL_COLUMNS = [
    "confidence_0_1",       # if client provides their confidence (0..1)
    "required_info",        # text: list of required fields separated by ';'
    "provided_info",        # text: list of provided fields separated by ';'
]

NUMERIC_COLUMNS = [
    "capex_eur",
    "annual_opex_saving_eur",
    "annual_co2_reduction_t",
    "implementation_months",
    "strategic_score_1_5",
    "confidence_0_1",
]


def safe_read_csv(uploaded_file) -> pd.DataFrame:
    """Robust CSV reader to avoid EmptyDataError and weird encodings."""
    if uploaded_file is None:
        raise ValueError("No file uploaded")

    raw = uploaded_file.getvalue()
    if raw is None or len(raw) == 0:
        raise ValueError("Uploaded file is empty")

    # Try common encodings / separators
    # First try default (comma), then semicolon
    for sep in [",", ";"]:
        for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc)
                if df.shape[1] >= 2:
                    return df
            except pd.errors.EmptyDataError:
                raise ValueError("CSV has no columns (empty or malformed).")
            except Exception:
                continue

    raise ValueError(
        "Could not parse the CSV. Make sure it is a valid CSV with headers and comma/semicolon separators."
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names and strip spaces."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns; keep as float where needed."""
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # basic checks
    if "id" in df.columns:
        if df["id"].isna().any():
            errors.append("Column 'id' has missing values.")
    if "initiative" in df.columns:
        if df["initiative"].astype(str).str.strip().eq("").any():
            errors.append("Column 'initiative' has blank values.")
    return (len(errors) == 0, errors)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def apply_confidence_penalty(value: float, confidence: float, floor: float = 0.4) -> float:
    """
    Penalize a numeric value based on confidence (0..1).
    floor controls minimum multiplier when confidence=0.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if confidence is None or (isinstance(confidence, float) and np.isnan(confidence)):
        confidence = 0.6  # default mid confidence if unknown
    confidence = clamp(float(confidence), 0.0, 1.0)
    floor = clamp(float(floor), 0.0, 1.0)
    multiplier = floor + (1.0 - floor) * confidence
    return float(value) * multiplier


def parse_semicolon_list(s: str) -> List[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip().lower() for x in s.split(";") if x.strip()]


def infer_confidence_row(row: pd.Series) -> float:
    """
    If confidence_0_1 is provided, use it.
    Else infer using required_info vs provided_info overlap if present.
    Else default 0.65.
    """
    if "confidence_0_1" in row.index and pd.notna(row["confidence_0_1"]):
        return clamp(float(row["confidence_0_1"]), 0.0, 1.0)

    req = parse_semicolon_list(row.get("required_info", ""))
    prv = parse_semicolon_list(row.get("provided_info", ""))

    if req:
        if not prv:
            return 0.45
        overlap = len(set(req).intersection(set(prv)))
        ratio = overlap / max(1, len(set(req)))
        # Map ratio to [0.4..0.95]
        return clamp(0.4 + 0.55 * ratio, 0.0, 1.0)

    return 0.65


def compute_metrics(
    df: pd.DataFrame,
    horizon_years: int,
    discount_rate: float,
    co2_price: float,
    confidence_floor: float,
) -> pd.DataFrame:
    df = df.copy()

    # Infer confidence
    df["confidence"] = df.apply(infer_confidence_row, axis=1)

    # CO2 value and benefit
    df["co2_value_eur_per_year"] = df["annual_co2_reduction_t"] * co2_price
    df["total_annual_benefit_eur"] = df["annual_opex_saving_eur"] + df["co2_value_eur_per_year"]

    # NPV (simple: benefits start after implementation; discount yearly)
    # Implementation delay in years:
    df["implementation_years"] = df["implementation_months"] / 12.0

    def npv_row(r: pd.Series) -> float:
        capex = r["capex_eur"]
        benefit = r["total_annual_benefit_eur"]
        if pd.isna(capex) or pd.isna(benefit):
            return np.nan
        if benefit <= 0:
            return -float(capex)

        delay = float(r["implementation_years"]) if pd.notna(r["implementation_years"]) else 0.0
        # Benefits begin after delay; approximate by shifting the first benefit year
        # If delay=0.0 -> start year 1; if delay in (0..1] -> start year 2, etc.
        start_year = int(math.floor(delay)) + 1
        npv = -float(capex)
        for t in range(start_year, horizon_years + 1):
            npv += float(benefit) / ((1.0 + discount_rate) ** t)
        return npv

    df["npv_eur"] = df.apply(npv_row, axis=1)

    # Payback (years) = capex / annual benefit (if benefit>0)
    df["payback_years"] = np.where(
        df["total_annual_benefit_eur"] > 0,
        df["capex_eur"] / df["total_annual_benefit_eur"],
        np.nan,
    )

    # Penalized NPV (confidence)
    df["npv_penalized_eur"] = df.apply(
        lambda r: apply_confidence_penalty(r["npv_eur"], r["confidence"], confidence_floor),
        axis=1,
    )

    # Add sanity fields
    df["strategic_score_1_5"] = df["strategic_score_1_5"].clip(1, 5)

    return df


def optimize_portfolio(
    df: pd.DataFrame,
    budget_eur: float,
    min_co2_t: float,
    objective: str,
    w_npv: float,
    w_co2: float,
    w_strategy: float,
) -> Tuple[pd.DataFrame, dict]:
    """
    Binary selection: pick initiatives under CAPEX budget.
    Optionally: CO2 constraint (>= min_co2_t if min_co2_t > 0).
    Objective options:
      - "Maximize penalized NPV"
      - "Maximize CO2 reduction"
      - "Balanced score (NPV + CO2 + strategy)"
    """
    df = df.copy()

    # Basic cleaning to avoid NaNs in solver
    df["capex_eur"] = df["capex_eur"].fillna(0.0)
    df["annual_co2_reduction_t"] = df["annual_co2_reduction_t"].fillna(0.0)
    df["npv_penalized_eur"] = df["npv_penalized_eur"].fillna(-1e9)  # bad if missing
    df["strategic_score_1_5"] = df["strategic_score_1_5"].fillna(3.0)

    # Create model
    model = pulp.LpProblem("DecarbPortfolio", pulp.LpMaximize)

    ids = df["id"].astype(str).tolist()
    x = pulp.LpVariable.dicts("x", ids, lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Constraints
    model += pulp.lpSum(x[i] * float(df.loc[df["id"].astype(str) == i, "capex_eur"].iloc[0]) for i in ids) <= float(budget_eur)

    if min_co2_t and min_co2_t > 0:
        model += pulp.lpSum(x[i] * float(df.loc[df["id"].astype(str) == i, "annual_co2_reduction_t"].iloc[0]) for i in ids) >= float(min_co2_t)

    # Objective
    if objective == "Maximize penalized NPV":
        model += pulp.lpSum(x[i] * float(df.loc[df["id"].astype(str) == i, "npv_penalized_eur"].iloc[0]) for i in ids)
    elif objective == "Maximize CO2 reduction":
        model += pulp.lpSum(x[i] * float(df.loc[df["id"].astype(str) == i, "annual_co2_reduction_t"].iloc[0]) for i in ids)
    else:
        # Balanced: normalize each component
        npv_vals = df["npv_penalized_eur"].values.astype(float)
        co2_vals = df["annual_co2_reduction_t"].values.astype(float)
        strat_vals = df["strategic_score_1_5"].values.astype(float)

        def norm(arr: np.ndarray) -> np.ndarray:
            a = np.array(arr, dtype=float)
            lo, hi = np.nanmin(a), np.nanmax(a)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
                return np.zeros_like(a)
            return (a - lo) / (hi - lo)

        df["npv_norm"] = norm(npv_vals)
        df["co2_norm"] = norm(co2_vals)
        df["strat_norm"] = norm(strat_vals)

        model += pulp.lpSum(
            x[i] * (
                w_npv * float(df.loc[df["id"].astype(str) == i, "npv_norm"].iloc[0])
                + w_co2 * float(df.loc[df["id"].astype(str) == i, "co2_norm"].iloc[0])
                + w_strategy * float(df.loc[df["id"].astype(str) == i, "strat_norm"].iloc[0])
            )
            for i in ids
        )

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = model.solve(solver)

    selected_ids = [i for i in ids if pulp.value(x[i]) > 0.5]
    df["selected"] = df["id"].astype(str).isin(selected_ids)

    summary = {
        "status": pulp.LpStatus.get(status, str(status)),
        "selected_count": int(df["selected"].sum()),
        "capex_selected": float(df.loc[df["selected"], "capex_eur"].sum()),
        "co2_selected": float(df.loc[df["selected"], "annual_co2_reduction_t"].sum()),
        "npv_selected": float(df.loc[df["selected"], "npv_penalized_eur"].sum()),
    }
    return df, summary


def template_csv_bytes() -> bytes:
    example = pd.DataFrame(
        [
            {
                "id": 1,
                "initiative": "Solar PV (self-consumption)",
                "capex_eur": 400000,
                "annual_opex_saving_eur": 95000,
                "annual_co2_reduction_t": 250,
                "implementation_months": 4,
                "strategic_score_1_5": 3,
                "notes": "If applicable; depends on roof/land availability and load profile.",
                "required_info": "roof_area_m2;location;orientation;annual_kwh",
                "provided_info": "",
                "confidence_0_1": 0.55,
            },
            {
                "id": 2,
                "initiative": "High efficiency motors + VFD",
                "capex_eur": 120000,
                "annual_opex_saving_eur": 45000,
                "annual_co2_reduction_t": 90,
                "implementation_months": 3,
                "strategic_score_1_5": 2,
                "notes": "Usually easier to quantify with motor inventory and operating hours.",
                "required_info": "motor_inventory;operating_hours;electricity_price",
                "provided_info": "",
                "confidence_0_1": 0.70,
            },
            {
                "id": 3,
                "initiative": "Compressed air leak program",
                "capex_eur": 60000,
                "annual_opex_saving_eur": 25000,
                "annual_co2_reduction_t": 40,
                "implementation_months": 2,
                "strategic_score_1_5": 1,
                "notes": "",
                "required_info": "compressed_air_kwh;leak_rate;electricity_price",
                "provided_info": "",
                "confidence_0_1": 0.75,
            },
        ]
    )
    return example.to_csv(index=False).encode("utf-8")


# -----------------------------
# Layout
# -----------------------------
st.title("Industrial Decarbonization Decision Tool")
st.caption("Prototype TFG – Carlos Falcó")

with st.sidebar:
    st.header("Inputs")

    horizon_years = st.slider("Project horizon (years)", min_value=1, max_value=10, value=5, step=1)
    discount_rate_pct = st.slider("Discount rate (%)", min_value=0.0, max_value=25.0, value=8.0, step=0.25)
    discount_rate = discount_rate_pct / 100.0

    co2_price = st.number_input("CO₂ price (€/t)", min_value=0.0, value=80.0, step=5.0)

    budget_eur = st.number_input("CAPEX budget (€)", min_value=0.0, value=600000.0, step=10000.0)
    min_co2_t = st.number_input("Minimum annual CO₂ reduction target (t/year) [optional]", min_value=0.0, value=0.0, step=10.0)

    st.subheader("Confidence handling")
    confidence_floor = st.slider("Minimum multiplier at low confidence", 0.0, 1.0, 0.4, 0.05)

    st.subheader("Optimization objective")
    objective = st.selectbox(
        "Objective",
        ["Maximize penalized NPV", "Maximize CO2 reduction", "Balanced score (NPV + CO2 + strategy)"],
        index=0,
    )

    w_npv, w_co2, w_strategy = 0.55, 0.30, 0.15
    if objective == "Balanced score (NPV + CO2 + strategy)":
        w_npv = st.slider("Weight: NPV", 0.0, 1.0, 0.55, 0.05)
        w_co2 = st.slider("Weight: CO₂", 0.0, 1.0, 0.30, 0.05)
        w_strategy = st.slider("Weight: Strategic score", 0.0, 1.0, 0.15, 0.05)
        s = max(1e-9, w_npv + w_co2 + w_strategy)
        w_npv, w_co2, w_strategy = w_npv / s, w_co2 / s, w_strategy / s

    st.divider()
    st.subheader("Client template")
    st.download_button(
        "Download CSV template (example)",
        data=template_csv_bytes(),
        file_name="client_initiatives_template.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.markdown("### 1) Upload initiatives CSV")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV (you can start from the template).")
    st.stop()

# Read + validate
try:
    df_raw = safe_read_csv(uploaded)
    df_raw = normalize_columns(df_raw)
    df_raw = coerce_numeric(df_raw)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

ok, errors = validate_schema(df_raw)
if not ok:
    st.error("CSV validation failed:")
    for err in errors:
        st.write(f"- {err}")
    st.stop()

# Fill missing optional columns
for c in OPTIONAL_COLUMNS:
    if c not in df_raw.columns:
        df_raw[c] = ""

# Metrics
df = compute_metrics(
    df_raw,
    horizon_years=horizon_years,
    discount_rate=discount_rate,
    co2_price=co2_price,
    confidence_floor=confidence_floor,
)

# Show initiative evaluation table
st.markdown("### 2) Initiative evaluation")
cols_to_show = [
    "id",
    "initiative",
    "capex_eur",
    "annual_opex_saving_eur",
    "annual_co2_reduction_t",
    "co2_value_eur_per_year",
    "total_annual_benefit_eur",
    "confidence",
    "npv_eur",
    "npv_penalized_eur",
    "payback_years",
    "implementation_months",
    "strategic_score_1_5",
    "notes",
]
cols_to_show = [c for c in cols_to_show if c in df.columns]
st.dataframe(df[cols_to_show], use_container_width=True, hide_index=True)

# Optimization
st.markdown("### 3) Portfolio optimization")
df_opt, summary = optimize_portfolio(
    df=df,
    budget_eur=budget_eur,
    min_co2_t=min_co2_t,
    objective=objective,
    w_npv=w_npv,
    w_co2=w_co2,
    w_strategy=w_strategy,
)

# Summary cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Status", summary["status"])
c2.metric("Selected initiatives", summary["selected_count"])
c3.metric("CAPEX selected (€)", f"{summary['capex_selected']:,.0f}")
c4.metric("CO₂ reduction selected (t/y)", f"{summary['co2_selected']:,.1f}")

st.metric("Total penalized NPV selected (€)", f"{summary['npv_selected']:,.0f}")

# Selected table
st.markdown("#### Selected initiatives")
selected_df = df_opt[df_opt["selected"]].copy()
st.dataframe(selected_df[cols_to_show], use_container_width=True, hide_index=True)

# Charts
st.markdown("### 4) Visuals")
chart_df = df_opt.copy()
chart_df["selected_label"] = np.where(chart_df["selected"], "Selected", "Not selected")

fig1 = px.scatter(
    chart_df,
    x="capex_eur",
    y="annual_co2_reduction_t",
    size="total_annual_benefit_eur",
    color="selected_label",
    hover_name="initiative",
    title="CAPEX vs CO₂ reduction (bubble size = annual benefit)",
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(
    chart_df.sort_values("npv_penalized_eur", ascending=False),
    x="initiative",
    y="npv_penalized_eur",
    color="selected_label",
    title="Penalized NPV by initiative",
)
fig2.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig2, use_container_width=True)

# Export results
st.markdown("### 5) Export results")
export_cols = [
    "id",
    "initiative",
    "selected",
    "capex_eur",
    "annual_opex_saving_eur",
    "annual_co2_reduction_t",
    "co2_value_eur_per_year",
    "total_annual_benefit_eur",
    "confidence",
    "npv_eur",
    "npv_penalized_eur",
    "payback_years",
    "implementation_months",
    "strategic_score_1_5",
    "notes",
    "required_info",
    "provided_info",
]
export_cols = [c for c in export_cols if c in df_opt.columns]
export = df_opt[export_cols].copy()

csv_bytes = export.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download portfolio results (CSV)",
    data=csv_bytes,
    file_name="portfolio_results.csv",
    mime="text/csv",
    use_container_width=True,
)

st.success("Ready ✅ If you update the CSV and re-upload, the tool recalculates and re-optimizes automatically.")
