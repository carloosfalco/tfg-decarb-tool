import streamlit as st
import pandas as pd
import numpy as np
import pulp

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Decarbonization Portfolio Optimizer", layout="wide")

# =========================
# HELPERS
# =========================
REQUIRED_COLS = [
    "id",
    "initiative",
    "capex_eur",
    "annual_opex_saving_eur",
    "annual_co2_reduction_t",
    "implementation_months",
    "strategic_score_1_5",
]

NUM_COLS = [
    "capex_eur",
    "annual_opex_saving_eur",
    "annual_co2_reduction_t",
    "implementation_months",
    "strategic_score_1_5",
]


def compute_npv(capex: float, annual_benefit: float, years: int, r: float) -> float:
    # NPV = -CAPEX + sum(benefit/(1+r)^t)
    return -capex + sum((annual_benefit / ((1 + r) ** t)) for t in range(1, years + 1))


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """
    Robust reader for Streamlit uploads:
    - uploaded_file.seek(0) to allow multiple reads across tabs
    - sep=None + engine='python' to autodetect comma/semicolon
    - strips BOM and whitespace
    """
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8-sig")
        if df.empty:
            raise ValueError("Uploaded CSV is empty.")
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")


def validate_template(df: pd.DataFrame) -> list:
    return [c for c in REQUIRED_COLS if c not in df.columns]


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def optimize_portfolio(
    df: pd.DataFrame,
    budget: float,
    max_impl_months: float | None,
    min_annual_co2: float | None,
    weights: dict,
):
    """
    Maximize:
      w_npv * NPV + w_co2 * CO2_total + w_strat * strategic_score

    Subject to:
      sum(CAPEX_i * x_i) <= budget
      sum(impl_months_i * x_i) <= max_impl_months   (optional)
      sum(annual_co2_i * x_i) >= min_annual_co2     (optional)
    """
    model = pulp.LpProblem("Portfolio_Selection", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in df.index}

    # Objective
    model += pulp.lpSum(
        x[i]
        * (
            weights["npv"] * df.loc[i, "npv_eur"]
            + weights["co2"] * df.loc[i, "co2_total_t"]
            + weights["strat"] * df.loc[i, "strategic_score_1_5"]
        )
        for i in df.index
    )

    # Budget constraint
    model += pulp.lpSum(x[i] * df.loc[i, "capex_eur"] for i in df.index) <= budget

    # Optional constraints
    if max_impl_months is not None:
        model += pulp.lpSum(x[i] * df.loc[i, "implementation_months"] for i in df.index) <= max_impl_months

    if min_annual_co2 is not None:
        model += pulp.lpSum(x[i] * df.loc[i, "annual_co2_reduction_t"] for i in df.index) >= min_annual_co2

    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    selected_idx = [i for i in df.index if pulp.value(x[i]) == 1]
    return selected_idx


def build_template_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1, "Solar PV (self-consumption)", 400000, 95000, 250, 4, 3, ""],
            [2, "High efficiency motors + VFD", 120000, 45000, 90, 3, 2, ""],
            [3, "Compressed air leak program", 60000, 25000, 40, 2, 1, ""],
            [4, "Waste heat recovery", 250000, 65000, 130, 6, 4, ""],
            [5, "EMS + submetering + analytics", 100000, 20000, 25, 4, 5, ""],
            [6, "HVAC optimization + controls", 150000, 28000, 40, 3, 3, ""],
            [7, "Process optimization (yield / scrap)", 90000, 50000, 15, 5, 4, ""],
            [8, "Electrification (partial)", 420000, 60000, 180, 7, 4, ""],
        ],
        columns=[
            "id",
            "initiative",
            "capex_eur",
            "annual_opex_saving_eur",
            "annual_co2_reduction_t",
            "implementation_months",
            "strategic_score_1_5",
            "notes",
        ],
    )


# =========================
# UI
# =========================
st.title("Decarbonization Portfolio Optimizer")
st.caption("Upload client initiatives → set assumptions → evaluate → optimize a portfolio under real constraints.")

# Sidebar
st.sidebar.header("1) Upload")
uploaded = st.sidebar.file_uploader("Upload CSV (client_template.csv)", type=["csv"])

st.sidebar.header("2) Assumptions")
years = st.sidebar.slider("Horizon (years)", 3, 7, 5)
discount_rate = st.sidebar.slider("Discount rate (%)", 0.0, 20.0, 8.0) / 100
co2_price = st.sidebar.number_input("CO₂ price (€/t)", value=80, step=10)

st.sidebar.header("3) Constraints")
budget = st.sidebar.number_input("CAPEX budget (€)", value=600000, step=50000)

use_impl = st.sidebar.checkbox("Add implementation capacity constraint", value=False)
max_impl_months = None
if use_impl:
    max_impl_months = st.sidebar.number_input("Max implementation months (sum)", value=18.0, step=1.0)

use_co2_target = st.sidebar.checkbox("Add minimum annual CO₂ target", value=False)
min_annual_co2 = None
if use_co2_target:
    min_annual_co2 = st.sidebar.number_input("Min annual CO₂ reduction (t/yr)", value=300.0, step=10.0)

st.sidebar.header("4) Objective weights")
st.sidebar.caption("Default = maximize NPV. Increase CO₂/Strategic to shift priorities.")
w_npv = st.sidebar.slider("Weight: NPV", 0.0, 1.0, 1.0, 0.05)
w_co2 = st.sidebar.slider("Weight: CO₂ (total over horizon)", 0.0, 1.0, 0.20, 0.05)
w_strat = st.sidebar.slider("Weight: Strategic score", 0.0, 1.0, 0.10, 0.05)
weights = {"npv": w_npv, "co2": w_co2, "strat": w_strat}

# Tabs
tab1, tab2, tab3 = st.tabs(["Client Template", "Evaluation", "Optimization Results"])

# -------------------------
# TAB 1: TEMPLATE
# -------------------------
with tab1:
    st.subheader("Client data template (send this to the client)")
    st.write("The client fills in one row per initiative (typically 8) and sends it back to you as CSV.")
    template_df = build_template_df()
    st.dataframe(template_df, use_container_width=True)

    st.download_button(
        "Download client_template.csv",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="client_template.csv",
        mime="text/csv",
    )

# -------------------------
# Read + compute metrics if upload exists
# -------------------------
df_client = None
df_eval = None

if uploaded is not None:
    try:
        df_client = read_uploaded_csv(uploaded)
    except ValueError as e:
        # Show error in main area, not only sidebar
        st.error(str(e))
        df_client = None

    if df_client is not None:
        missing = validate_template(df_client)
        if missing:
            st.error(f"Template error. Missing columns: {', '.join(missing)}")
            df_client = None

if df_client is not None:
    df_eval = df_client.copy()
    df_eval = coerce_numeric(df_eval)

    # Basic sanity checks
    if df_eval[NUM_COLS].isna().any().any():
        st.warning("Some numeric fields could not be parsed (NaN). Check the CSV values (no € signs, no text).")

    # Metrics
    df_eval["co2_value_eur_per_year"] = df_eval["annual_co2_reduction_t"] * co2_price
    df_eval["annual_total_benefit_eur"] = df_eval["annual_opex_saving_eur"] + df_eval["co2_value_eur_per_year"]

    df_eval["npv_eur"] = [
        compute_npv(capex, benefit, years, discount_rate)
        for capex, benefit in zip(df_eval["capex_eur"], df_eval["annual_total_benefit_eur"])
    ]

    df_eval["payback_years"] = df_eval["capex_eur"] / df_eval["annual_total_benefit_eur"].replace({0: np.nan})
    df_eval["co2_total_t"] = df_eval["annual_co2_reduction_t"] * years
    df_eval["macc_eur_per_t"] = df_eval["capex_eur"] / df_eval["co2_total_t"].replace({0: np.nan})

# -------------------------
# TAB 2: EVALUATION
# -------------------------
with tab2:
    st.subheader("Initiative evaluation")
    if df_eval is None:
        st.info("Upload a client CSV in the sidebar to evaluate initiatives.")
    else:
        show_cols = [
            "id",
            "initiative",
            "capex_eur",
            "annual_opex_saving_eur",
            "annual_co2_reduction_t",
            "co2_value_eur_per_year",
            "annual_total_benefit_eur",
            "npv_eur",
            "payback_years",
            "macc_eur_per_t",
            "implementation_months",
            "strategic_score_1_5",
            "notes" if "notes" in df_eval.columns else None,
        ]
        show_cols = [c for c in show_cols if c is not None]

        st.dataframe(df_eval[show_cols].sort_values("npv_eur", ascending=False), use_container_width=True)

        st.caption(
            "MACC shown here is a simplified proxy: CAPEX / total CO₂ abated over the horizon. "
            "You can refine it later (net CAPEX vs savings, OPEX, etc.)."
        )

# -------------------------
# TAB 3: OPTIMIZATION
# -------------------------
with tab3:
    st.subheader("Optimal portfolio selection (MILP)")
    if df_eval is None:
        st.info("Upload a client CSV in the sidebar to run the optimization.")
    else:
        # Run optimization
        selected_idx = optimize_portfolio(
            df=df_eval,
            budget=budget,
            max_impl_months=max_impl_months if use_impl else None,
            min_annual_co2=min_annual_co2 if use_co2_target else None,
            weights=weights,
        )

        df_out = df_eval.copy()
        df_out["selected"] = 0
        df_out.loc[selected_idx, "selected"] = 1

        selected_df = df_out[df_out["selected"] == 1].copy()

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Selected initiatives", f"{len(selected_df)}/{len(df_out)}")
        col2.metric("CAPEX used (€)", f"{selected_df['capex_eur'].sum():,.0f}")
        col3.metric("Portfolio NPV (€)", f"{selected_df['npv_eur'].sum():,.0f}")
        col4.metric("Annual CO₂ reduction (t/yr)", f"{selected_df['annual_co2_reduction_t'].sum():,.0f}")

        st.markdown("### Selected portfolio")
        st.dataframe(
            selected_df[
                [
                    "id",
                    "initiative",
                    "capex_eur",
                    "annual_opex_saving_eur",
                    "annual_co2_reduction_t",
                    "implementation_months",
                    "strategic_score_1_5",
                    "npv_eur",
                    "payback_years",
                ]
            ].sort_values("npv_eur", ascending=False),
            use_container_width=True,
        )

        st.markdown("### Full list (with selection flag)")
        cols_full = [
            "id",
            "initiative",
            "capex_eur",
            "annual_opex_saving_eur",
            "annual_co2_reduction_t",
            "implementation_months",
            "strategic_score_1_5",
            "npv_eur",
            "selected",
        ]
        if "notes" in df_out.columns:
            cols_full.append("notes")

        st.dataframe(df_out[cols_full].sort_values(["selected", "npv_eur"], ascending=[False, False]), use_container_width=True)

        # Download results
        export = df_out.copy()
        export["horizon_years"] = years
        export["discount_rate"] = discount_rate
        export["co2_price_eur_per_t"] = co2_price
        export["budget_eur"] = budget
        export["max_impl_months_sum"] = max_impl_months if use_impl else np.nan
        export["min_annual_co2_target"] = min_annual_co2 if use_co2_target else np.nan
        export["w_npv"] = w_npv
        export["w_co2"] = w_co2
        export["w_strat"] = w_strat

        st.download_button(
            "Download portfolio_results.csv",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="portfolio_results.csv",
            mime="text/csv",
        )
