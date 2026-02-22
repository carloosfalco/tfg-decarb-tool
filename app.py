import streamlit as st
import pandas as pd
import numpy as np
import pulp

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Industrial Decarb Copilot", layout="wide")
st.title("Industrial Decarbonization Copilot (MVP)")
st.caption(
    "Upload plant data → auto-generate initiatives (rule-based with data-gating) → set assumptions → optimize portfolio."
)

# =========================================================
# HELPERS
# =========================================================
def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """
    Robust CSV reader for Streamlit uploads:
    - seek(0) enables multiple reads
    - sep=None autodetects ',' vs ';'
    - utf-8-sig handles BOM from Excel
    """
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8-sig")
    if df is None or df.empty:
        raise ValueError("Uploaded CSV is empty or unreadable.")
    df.columns = [c.strip() for c in df.columns]
    return df

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def safe_div(a, b):
    try:
        if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def kwh_to_mwh(kwh: float) -> float:
    return kwh / 1000.0

def compute_npv(capex: float, annual_benefit: float, years: int, r: float) -> float:
    return -capex + sum(annual_benefit / ((1 + r) ** t) for t in range(1, years + 1))

def normalize_yes_no_unknown(x):
    if x is None:
        return "unknown"
    s = str(x).strip().lower()
    if s in ["yes", "y", "true", "1"]:
        return "yes"
    if s in ["no", "n", "false", "0"]:
        return "no"
    return "unknown"

def missing_fields(plant: dict, fields: list[str]) -> list[str]:
    miss = []
    for f in fields:
        v = plant.get(f, None)
        if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
            miss.append(f)
    return miss

def confidence_from_missing(n_missing: int) -> str:
    if n_missing == 0:
        return "High"
    if n_missing <= 2:
        return "Medium"
    return "Low"

def confidence_factor(conf: str) -> float:
    return {"High": 1.0, "Medium": 0.8, "Low": 0.55}.get(conf, 0.7)

def fmt_missing(lst: list[str]) -> str:
    return ", ".join(lst) if lst else ""

# =========================================================
# DEFAULT EMISSION FACTORS (simple, for MVP)
# =========================================================
DEFAULT_EF_ELECTRICITY = 0.20  # tCO2 / MWh
DEFAULT_EF_GAS = 0.202         # tCO2 / MWh

# =========================================================
# PLANT TEMPLATE (SEND TO CLIENT)
# Added "gating" fields to avoid false certainty.
# =========================================================
PLANT_TEMPLATE_COLS = [
    # Identity / context
    "plant_name",
    "country",
    "industry_subsector",

    # Energy totals (annual)
    "annual_electricity_kwh",
    "annual_electricity_cost_eur",
    "annual_gas_kwh",
    "annual_gas_cost_eur",

    # Site + constraints (for PV, electrification, etc.)
    "pv_already_installed",          # yes/no/unknown
    "roof_area_available_m2",        # number or blank
    "site_has_free_land",            # yes/no/unknown
    "grid_connection_limit_kw",      # number or blank

    # Utilities/equipment (screening-level)
    "compressed_air_installed_kw",
    "compressed_air_hours_per_year",
    "compressed_air_has_leak_program",  # yes/no/unknown

    "hvac_installed_kw",
    "hvac_hours_per_year",
    "hvac_has_bms_controls",            # yes/no/unknown

    "percent_motors_with_vfd",          # 0-100
    "percent_led_lighting",             # 0-100

    # Operational hints (for process optimization screening)
    "scrap_rate_percent",               # optional
    "oee_percent",                      # optional
]

def build_plant_template() -> pd.DataFrame:
    return pd.DataFrame(
        [[
            "Example Plant",
            "Spain",
            "Industrial manufacturing",
            5_000_000,   # kWh
            750_000,     # €
            3_500_000,   # kWh
            210_000,     # €
            "unknown",
            "",          # roof_area_available_m2
            "unknown",
            "",          # grid_connection_limit_kw
            250,         # compressed_air_installed_kw
            4_500,       # compressed_air_hours_per_year
            "unknown",
            300,         # hvac_installed_kw
            2_500,       # hvac_hours_per_year
            "unknown",
            40,          # % motors with VFD
            60,          # % LED lighting
            "",          # scrap_rate_percent
            "",          # oee_percent
        ]],
        columns=PLANT_TEMPLATE_COLS,
    )

# =========================================================
# RULE-BASED INITIATIVE GENERATION WITH DATA GATING
# =========================================================
def generate_initiatives_from_plant(
    plant: dict,
    ef_el_t_per_mwh: float,
    ef_gas_t_per_mwh: float,
) -> pd.DataFrame:
    """
    Produces initiatives as:
      - Feasible: enough inputs to estimate reasonably
      - Screening: possible, but missing key inputs (needs validation)
      - Not feasible: contradicted by input (already implemented / not applicable)
    """
    # Normalize yes/no/unknown
    plant = dict(plant)
    plant["pv_already_installed"] = normalize_yes_no_unknown(plant.get("pv_already_installed"))
    plant["site_has_free_land"] = normalize_yes_no_unknown(plant.get("site_has_free_land"))
    plant["compressed_air_has_leak_program"] = normalize_yes_no_unknown(plant.get("compressed_air_has_leak_program"))
    plant["hvac_has_bms_controls"] = normalize_yes_no_unknown(plant.get("hvac_has_bms_controls"))

    # Extract numeric fields
    el_kwh = float(plant.get("annual_electricity_kwh") or 0)
    el_cost = float(plant.get("annual_electricity_cost_eur") or 0)
    gas_kwh = float(plant.get("annual_gas_kwh") or 0)
    gas_cost = float(plant.get("annual_gas_cost_eur") or 0)

    ca_kw = float(plant.get("compressed_air_installed_kw") or 0)
    ca_h = float(plant.get("compressed_air_hours_per_year") or 0)

    hvac_kw = float(plant.get("hvac_installed_kw") or 0)
    hvac_h = float(plant.get("hvac_hours_per_year") or 0)

    vfd_pct = float(plant.get("percent_motors_with_vfd") or 0) if plant.get("percent_motors_with_vfd") is not None else np.nan
    led_pct = float(plant.get("percent_led_lighting") or 0) if plant.get("percent_led_lighting") is not None else np.nan

    roof_area = plant.get("roof_area_available_m2", None)
    roof_area = float(roof_area) if roof_area is not None and str(roof_area).strip() != "" else np.nan
    grid_limit_kw = plant.get("grid_connection_limit_kw", None)
    grid_limit_kw = float(grid_limit_kw) if grid_limit_kw is not None and str(grid_limit_kw).strip() != "" else np.nan

    # Derived prices (may be missing)
    el_price = safe_div(el_cost, el_kwh)  # €/kWh
    gas_price = safe_div(gas_cost, gas_kwh)

    initiatives = []
    next_id = 1

    def add_initiative(
        name: str,
        capex: float,
        annual_opex_save: float,
        annual_co2_save: float,
        impl_months: float,
        strat: float,
        required_fields: list[str],
        hard_block_reason: str | None = None,
        assumptions: str = "",
    ):
        nonlocal next_id
        miss = missing_fields(plant, required_fields)

        if hard_block_reason:
            status = "Not feasible"
            conf = "High"
            miss_str = ""
        else:
            status = "Feasible" if len(miss) == 0 else "Screening"
            conf = confidence_from_missing(len(miss))
            miss_str = fmt_missing(miss)

        initiatives.append({
            "id": next_id,
            "initiative": name,
            "status": status,
            "confidence": conf,
            "confidence_factor": confidence_factor(conf),
            "missing_data": miss_str,
            "assumptions": assumptions,
            "block_reason": hard_block_reason or "",
            "capex_eur": float(capex),
            "annual_opex_saving_eur": float(annual_opex_save),
            "annual_co2_reduction_t": float(annual_co2_save),
            "implementation_months": float(impl_months),
            "strategic_score_1_5": float(strat),
        })
        next_id += 1

    # -------------------------
    # 1) EMS + submetering
    # Needs: electricity cost (or at least kWh + cost)
    # -------------------------
    req = ["annual_electricity_kwh", "annual_electricity_cost_eur"]
    if el_cost > 0:
        save = 0.02 * el_cost
        capex = max(30_000, min(120_000, 0.02 * el_cost))
        # CO2 savings estimated from electricity price (if missing, CO2=0 and will be Screening)
        co2 = 0.0
        if el_price and not np.isnan(el_price) and el_price > 0:
            save_kwh = save / el_price
            co2 = kwh_to_mwh(save_kwh) * ef_el_t_per_mwh
        add_initiative(
            "Energy Management System (EMS) + submetering",
            capex=capex,
            annual_opex_save=save,
            annual_co2_save=co2,
            impl_months=4,
            strat=5,
            required_fields=req,
            assumptions="Savings assumed ~2% of annual electricity spend; CO₂ from implied saved kWh using avg €/kWh."
        )
    else:
        add_initiative(
            "Energy Management System (EMS) + submetering",
            capex=50_000,
            annual_opex_save=20_000,
            annual_co2_save=0,
            impl_months=4,
            strat=5,
            required_fields=req,
            assumptions="Fallback estimates used due to missing electricity totals."
        )

    # -------------------------
    # 2) Compressed air leaks + pressure optimization
    # Gating: needs CA kW, hours and electricity price to compute €
    # Hard block: if explicitly already has leak program => still possible but smaller; mark Screening unless we know maturity
    # -------------------------
    req = ["compressed_air_installed_kw", "compressed_air_hours_per_year", "annual_electricity_kwh", "annual_electricity_cost_eur"]
    ca_kwh = ca_kw * ca_h
    if ca_kwh > 50_000:
        # adjust potential if leak program already exists
        leak_prog = plant.get("compressed_air_has_leak_program")
        potential = 0.15 if leak_prog != "yes" else 0.07  # reduced if already has program
        ca_save_kwh = potential * ca_kwh
        save = (ca_save_kwh * el_price) if el_price and not np.isnan(el_price) else 0.0
        capex = max(15_000, min(80_000, 0.5 * save if save > 0 else 40_000))
        co2 = kwh_to_mwh(ca_save_kwh) * ef_el_t_per_mwh
        add_initiative(
            "Compressed air leak program + pressure optimization + controls",
            capex=capex,
            annual_opex_save=save,
            annual_co2_save=co2,
            impl_months=2,
            strat=2,
            required_fields=req,
            assumptions=f"Compressed air savings assumed {int(potential*100)}% of CA energy; lower if leak program already exists."
        )

    # -------------------------
    # 3) HVAC optimization (controls, setpoints, scheduling)
    # Gating: needs HVAC kW, hours, electricity price
    # If already has BMS/controls = yes -> still possible but reduced potential
    # -------------------------
    req = ["hvac_installed_kw", "hvac_hours_per_year", "annual_electricity_kwh", "annual_electricity_cost_eur"]
    hvac_kwh = hvac_kw * hvac_h
    if hvac_kwh > 50_000:
        has_bms = plant.get("hvac_has_bms_controls")
        potential = 0.12 if has_bms != "yes" else 0.06
        hvac_save_kwh = potential * hvac_kwh
        save = (hvac_save_kwh * el_price) if el_price and not np.isnan(el_price) else 0.0
        capex = max(20_000, min(120_000, 0.6 * save if save > 0 else 60_000))
        co2 = kwh_to_mwh(hvac_save_kwh) * ef_el_t_per_mwh
        add_initiative(
            "HVAC optimization (controls, setpoints, scheduling)",
            capex=capex,
            annual_opex_save=save,
            annual_co2_save=co2,
            impl_months=3,
            strat=3,
            required_fields=req,
            assumptions=f"HVAC savings assumed {int(potential*100)}% of HVAC electricity; reduced if BMS/controls already in place."
        )

    # -------------------------
    # 4) LED lighting retrofit + smart controls
    # Gating: needs LED %
    # Hard block: if LED already high (>=90) -> Not feasible (or "minor upgrades", but for MVP block it)
    # -------------------------
    req = ["percent_led_lighting", "annual_electricity_kwh", "annual_electricity_cost_eur"]
    if not np.isnan(led_pct):
        if led_pct >= 90:
            add_initiative(
                "LED lighting retrofit + smart controls",
                capex=0,
                annual_opex_save=0,
                annual_co2_save=0,
                impl_months=0,
                strat=1,
                required_fields=req,
                hard_block_reason="LED penetration already >= 90% (retrofit not a priority).",
                assumptions="If needed, consider only minor control tuning / occupancy sensors."
            )
        else:
            lighting_kwh = 0.03 * el_kwh if el_kwh > 0 else 0
            save_kwh = 0.50 * lighting_kwh
            save = (save_kwh * el_price) if el_price and not np.isnan(el_price) else 0.0
            capex = max(25_000, min(150_000, 1.5 * save if save > 0 else 70_000))
            co2 = kwh_to_mwh(save_kwh) * ef_el_t_per_mwh
            add_initiative(
                "LED lighting retrofit + smart controls",
                capex=capex,
                annual_opex_save=save,
                annual_co2_save=co2,
                impl_months=2,
                strat=2,
                required_fields=req,
                assumptions="Lighting share assumed 3% of electricity; retrofit saves 50% of lighting energy."
            )

    # -------------------------
    # 5) VFD + high-efficiency motors (targeted)
    # Gating: needs VFD %
    # Hard block: if VFD >= 85 -> deprioritize (block)
    # -------------------------
    req = ["percent_motors_with_vfd", "annual_electricity_kwh", "annual_electricity_cost_eur"]
    if not np.isnan(vfd_pct):
        if vfd_pct >= 85:
            add_initiative(
                "Variable Frequency Drives (VFD) + high-efficiency motors (targeted)",
                capex=0,
                annual_opex_save=0,
                annual_co2_save=0,
                impl_months=0,
                strat=1,
                required_fields=req,
                hard_block_reason="Estimated VFD penetration already high (>= 85%).",
                assumptions="If needed, focus on niche loads (pumps/fans) or maintenance-driven replacements."
            )
        else:
            motor_kwh = 0.25 * el_kwh if el_kwh > 0 else 0
            save_kwh = 0.05 * motor_kwh
            save = (save_kwh * el_price) if el_price and not np.isnan(el_price) else 0.0
            capex = max(50_000, min(300_000, 2.0 * save if save > 0 else 140_000))
            co2 = kwh_to_mwh(save_kwh) * ef_el_t_per_mwh
            add_initiative(
                "Variable Frequency Drives (VFD) + high-efficiency motors (targeted)",
                capex=capex,
                annual_opex_save=save,
                annual_co2_save=co2,
                impl_months=4,
                strat=3,
                required_fields=req,
                assumptions="Motors share assumed 25% of electricity; targeted VFD/motor upgrades save 5% of motor energy."
            )

    # -------------------------
    # 6) Solar PV self-consumption (screening candidate)
    # Gating: needs PV already installed + roof area or free land, and electricity totals.
    # Hard block: if PV already installed = yes AND roof/land unknown -> propose expansion as Screening instead (not block).
    # -------------------------
    req = ["annual_electricity_kwh", "annual_electricity_cost_eur", "pv_already_installed", "roof_area_available_m2", "site_has_free_land"]
    if el_kwh > 500_000 and el_price and not np.isnan(el_price):
        pv_already = plant.get("pv_already_installed")
        # If PV exists, keep but mark as expansion screening unless we know remaining area
        pv_name = "Solar PV self-consumption (roof/ground) – feasibility screening"
        assumptions = "Assumes PV offsets 10% of annual electricity if site area allows; requires roof/land check and grid constraints."
        # Basic pre-estimate (will be Screening if missing roof/land info)
        pv_share = 0.10
        pv_kwh = pv_share * el_kwh
        save = pv_kwh * el_price
        capex = max(150_000, min(1_200_000, pv_kwh * 0.45))
        co2 = kwh_to_mwh(pv_kwh) * ef_el_t_per_mwh

        # If PV already installed = yes, reduce “new potential” unless we know remaining area
        if pv_already == "yes":
            pv_name = "Solar PV expansion (existing PV on site) – feasibility screening"
            pv_kwh *= 0.50
            save *= 0.50
            co2 *= 0.50
            assumptions = "PV already installed: assumed 50% of baseline PV potential remains, subject to available area and grid constraints."

        add_initiative(
            pv_name,
            capex=capex,
            annual_opex_save=save,
            annual_co2_save=co2,
            impl_months=5,
            strat=4,
            required_fields=req,
            assumptions=assumptions
        )

    # -------------------------
    # 7) Waste heat recovery / thermal optimization (gas)
    # Gating: needs gas totals and cost.
    # -------------------------
    req = ["annual_gas_kwh", "annual_gas_cost_eur"]
    if gas_kwh > 500_000:
        potential = 0.08
        save_kwh = potential * gas_kwh
        save = (save_kwh * gas_price) if gas_price and not np.isnan(gas_price) else 0.0
        capex = max(120_000, min(800_000, 3.0 * save if save > 0 else 250_000))
        co2 = kwh_to_mwh(save_kwh) * ef_gas_t_per_mwh
        add_initiative(
            "Waste heat recovery / process heat optimization (thermal)",
            capex=capex,
            annual_opex_save=save,
            annual_co2_save=co2,
            impl_months=6,
            strat=4,
            required_fields=req,
            assumptions="Thermal savings assumed 8% of annual gas use; requires process temperature/heat integration validation."
        )

    # -------------------------
    # 8) Electrification (partial) of thermal demand
    # Gating: needs gas + electricity totals/costs and grid constraints ideally.
    # It's very scenario-sensitive, so often Screening.
    # -------------------------
    req = ["annual_gas_kwh", "annual_gas_cost_eur", "annual_electricity_kwh", "annual_electricity_cost_eur", "grid_connection_limit_kw"]
    if gas_kwh > 1_000_000 and el_price and gas_price and not np.isnan(el_price) and not np.isnan(gas_price):
        sub_gas_kwh = 0.10 * gas_kwh
        extra_el_kwh = 0.60 * sub_gas_kwh  # efficiency gain assumption
        annual_cost_change = (extra_el_kwh * el_price) - (sub_gas_kwh * gas_price)
        annual_opex_save = -annual_cost_change  # may be negative

        # CO2: avoided gas - added electricity
        co2_avoided = kwh_to_mwh(sub_gas_kwh) * ef_gas_t_per_mwh - kwh_to_mwh(extra_el_kwh) * ef_el_t_per_mwh
        co2_avoided = max(co2_avoided, 0)

        capex = 800_000
        add_initiative(
            "Electrification (partial) of thermal demand – feasibility screening",
            capex=capex,
            annual_opex_save=annual_opex_save,
            annual_co2_save=co2_avoided,
            impl_months=7,
            strat=4,
            required_fields=req,
            assumptions="Assumes 10% gas substitution, electricity demand at 0.6 kWh/kWh_gas displaced; requires grid capacity + process constraints."
        )

    # -------------------------
    # 9) Process optimization (yield/scrap)
    # Highly dependent on production KPIs. If missing, Screening but still useful as a candidate.
    # -------------------------
    req = ["scrap_rate_percent", "annual_electricity_cost_eur"]
    if el_cost > 0:
        scrap = plant.get("scrap_rate_percent", np.nan)
        scrap = float(scrap) if scrap is not None and str(scrap).strip() != "" else np.nan
        # heuristic: if scrap high -> higher savings potential
        if np.isnan(scrap):
            potential_spend = 0.01  # conservative
            assumptions = "No scrap KPI provided: assumes 1% electricity spend equivalent savings via yield/scrap improvement."
        else:
            potential_spend = 0.015 if scrap >= 5 else 0.008
            assumptions = f"Scrap rate {scrap:.1f}%: assumes {potential_spend*100:.1f}% electricity spend equivalent savings via yield/scrap improvement."

        save = potential_spend * el_cost
        capex = max(50_000, min(250_000, 2.0 * save if save > 0 else 120_000))
        # CO2 via saved kWh if price known
        co2 = 0.0
        if el_price and not np.isnan(el_price) and el_price > 0:
            save_kwh = save / el_price
            co2 = kwh_to_mwh(save_kwh) * ef_el_t_per_mwh

        add_initiative(
            "Process optimization (yield / scrap reduction) – screening",
            capex=capex,
            annual_opex_save=save,
            annual_co2_save=co2,
            impl_months=5,
            strat=4,
            required_fields=req,
            assumptions=assumptions
        )

    df = pd.DataFrame(initiatives)

    # Ensure at least one row
    if df.empty:
        df = pd.DataFrame([{
            "id": 1,
            "initiative": "Energy Management System (EMS) + submetering",
            "status": "Screening",
            "confidence": "Low",
            "confidence_factor": 0.55,
            "missing_data": "annual_electricity_kwh, annual_electricity_cost_eur",
            "assumptions": "Fallback initiative when insufficient plant data is provided.",
            "block_reason": "",
            "capex_eur": 50_000,
            "annual_opex_saving_eur": 20_000,
            "annual_co2_reduction_t": 0,
            "implementation_months": 4,
            "strategic_score_1_5": 5,
        }])

    return df

# =========================================================
# OPTIMIZATION
# =========================================================
def optimize_portfolio(df: pd.DataFrame, budget: float, max_impl_months: float | None, min_annual_co2: float | None, weights: dict, apply_confidence_penalty: bool):
    """
    Maximize weighted objective with optional confidence penalty:
      score = (w_npv*npv + w_co2*co2_total + w_strat*strategic) * confidence_factor (optional)
    Constraints:
      CAPEX <= budget
      (optional) sum(implementation_months) <= max_impl_months
      (optional) annual_co2 >= min_annual_co2
    """
    model = pulp.LpProblem("Portfolio_Selection", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in df.index}

    def row_score(i):
        base = (
            weights["npv"] * df.loc[i, "npv_eur"] +
            weights["co2"] * df.loc[i, "co2_total_t"] +
            weights["strat"] * df.loc[i, "strategic_score_1_5"]
        )
        if apply_confidence_penalty:
            base *= float(df.loc[i, "confidence_factor"])
        return base

    model += pulp.lpSum(x[i] * row_score(i) for i in df.index)
    model += pulp.lpSum(x[i] * df.loc[i, "capex_eur"] for i in df.index) <= budget

    if max_impl_months is not None:
        model += pulp.lpSum(x[i] * df.loc[i, "implementation_months"] for i in df.index) <= max_impl_months

    if min_annual_co2 is not None:
        model += pulp.lpSum(x[i] * df.loc[i, "annual_co2_reduction_t"] for i in df.index) >= min_annual_co2

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    selected_idx = [i for i in df.index if pulp.value(x[i]) == 1]
    return selected_idx

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("A) Plant data input")
uploaded_plant = st.sidebar.file_uploader("Upload Plant_Data_Input.csv", type=["csv"])

st.sidebar.header("B) Emission factors (MVP)")
ef_el = st.sidebar.number_input("Electricity EF (tCO₂/MWh)", value=float(DEFAULT_EF_ELECTRICITY), step=0.01)
ef_gas = st.sidebar.number_input("Gas EF (tCO₂/MWh)", value=float(DEFAULT_EF_GAS), step=0.01)

st.sidebar.header("C) Financial assumptions")
years = st.sidebar.slider("Horizon (years)", 3, 7, 5)
discount_rate = st.sidebar.slider("Discount rate (%)", 0.0, 20.0, 8.0) / 100
co2_price = st.sidebar.number_input("CO₂ price (€/t)", value=80, step=10)

st.sidebar.header("D) Portfolio constraints")
budget = st.sidebar.number_input("CAPEX budget (€)", value=600_000, step=50_000)

use_impl = st.sidebar.checkbox("Add implementation capacity constraint", value=False)
max_impl_months = None
if use_impl:
    max_impl_months = st.sidebar.number_input("Max implementation months (sum)", value=18.0, step=1.0)

use_co2_target = st.sidebar.checkbox("Add minimum annual CO₂ target", value=False)
min_annual_co2 = None
if use_co2_target:
    min_annual_co2 = st.sidebar.number_input("Min annual CO₂ reduction (t/yr)", value=300.0, step=10.0)

st.sidebar.header("E) Objective weights")
st.sidebar.caption("Default prioritizes NPV; increase CO₂/Strategic to shift priorities.")
w_npv = st.sidebar.slider("Weight: NPV", 0.0, 1.0, 1.0, 0.05)
w_co2 = st.sidebar.slider("Weight: CO₂ (total over horizon)", 0.0, 1.0, 0.25, 0.05)
w_strat = st.sidebar.slider("Weight: Strategic score", 0.0, 1.0, 0.10, 0.05)
weights = {"npv": w_npv, "co2": w_co2, "strat": w_strat}

st.sidebar.header("F) Screening handling")
include_screening = st.sidebar.checkbox("Include Screening initiatives in optimization", value=False)
apply_conf_penalty = st.sidebar.checkbox("Penalize low-confidence initiatives in objective", value=True)

# =========================================================
# MAIN: TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1) Client Template",
    "2) Plant Baseline",
    "3) Generated Initiatives",
    "4) Optimization",
])

# -----------------------------
# TAB 1: template download
# -----------------------------
with tab1:
    st.subheader("Client template (send to plant / client)")
    st.write(
        "This is a **minimal, screening-friendly** dataset. Missing data will not break the tool—"
        "it will label initiatives as **Screening** and tell you what to validate."
    )
    tpl = build_plant_template()
    st.dataframe(tpl, use_container_width=True)

    st.download_button(
        "Download Plant_Data_Input.csv",
        data=tpl.to_csv(index=False).encode("utf-8"),
        file_name="Plant_Data_Input.csv",
        mime="text/csv",
    )

# -----------------------------
# Load plant data
# -----------------------------
plant_df = None
plant = None

if uploaded_plant is not None:
    try:
        plant_df = read_uploaded_csv(uploaded_plant)
    except Exception as e:
        st.error(f"Error reading plant CSV: {e}")
        plant_df = None

if plant_df is not None:
    missing_cols = [c for c in PLANT_TEMPLATE_COLS if c not in plant_df.columns]
    if missing_cols:
        st.error(f"Plant template error. Missing columns: {', '.join(missing_cols)}")
        plant_df = None
    else:
        row = plant_df.iloc[0].copy()
        for c in PLANT_TEMPLATE_COLS:
            if c not in ["plant_name", "country", "industry_subsector",
                         "pv_already_installed", "site_has_free_land",
                         "compressed_air_has_leak_program", "hvac_has_bms_controls"]:
                row[c] = to_num(row[c])
        plant = row.to_dict()

# -----------------------------
# TAB 2: baseline
# -----------------------------
with tab2:
    st.subheader("Plant baseline (estimated)")
    if plant is None:
        st.info("Upload the Plant_Data_Input.csv in the sidebar to compute baseline and generate initiatives.")
    else:
        el_kwh = float(plant.get("annual_electricity_kwh") or 0)
        el_cost = float(plant.get("annual_electricity_cost_eur") or 0)
        gas_kwh = float(plant.get("annual_gas_kwh") or 0)
        gas_cost = float(plant.get("annual_gas_cost_eur") or 0)

        el_price = safe_div(el_cost, el_kwh)
        gas_price = safe_div(gas_cost, gas_kwh)

        el_mwh = kwh_to_mwh(el_kwh)
        gas_mwh = kwh_to_mwh(gas_kwh)

        baseline_co2_el = el_mwh * ef_el
        baseline_co2_gas = gas_mwh * ef_gas
        baseline_co2_total = baseline_co2_el + baseline_co2_gas

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Electricity (MWh/yr)", f"{el_mwh:,.0f}")
        c2.metric("Gas (MWh/yr)", f"{gas_mwh:,.0f}")
        c3.metric("Electricity price (€/kWh)", f"{el_price:.3f}" if el_price and not np.isnan(el_price) else "n/a")
        c4.metric("Gas price (€/kWh)", f"{gas_price:.3f}" if gas_price and not np.isnan(gas_price) else "n/a")

        c5, c6, c7 = st.columns(3)
        c5.metric("CO₂ (electricity) t/yr", f"{baseline_co2_el:,.0f}")
        c6.metric("CO₂ (gas) t/yr", f"{baseline_co2_gas:,.0f}")
        c7.metric("CO₂ total t/yr", f"{baseline_co2_total:,.0f}")

        st.markdown("### Raw plant input (as used by the engine)")
        st.dataframe(pd.DataFrame([plant]), use_container_width=True)

# -----------------------------
# TAB 3: initiatives generation
# -----------------------------
initiatives_df = None
with tab3:
    st.subheader("Auto-generated initiatives (rule-based + data-gated)")
    if plant is None:
        st.info("Upload plant data to generate initiatives.")
    else:
        initiatives_df = generate_initiatives_from_plant(
            plant=plant,
            ef_el_t_per_mwh=ef_el,
            ef_gas_t_per_mwh=ef_gas,
        )

        # Add financial & CO2 valuation metrics
        initiatives_df["co2_value_eur_per_year"] = initiatives_df["annual_co2_reduction_t"] * co2_price
        initiatives_df["annual_total_benefit_eur"] = initiatives_df["annual_opex_saving_eur"] + initiatives_df["co2_value_eur_per_year"]
        initiatives_df["npv_eur"] = [
            compute_npv(capex, benefit, years, discount_rate)
            for capex, benefit in zip(initiatives_df["capex_eur"], initiatives_df["annual_total_benefit_eur"])
        ]
        initiatives_df["payback_years"] = initiatives_df["capex_eur"] / initiatives_df["annual_total_benefit_eur"].replace({0: np.nan})
        initiatives_df["co2_total_t"] = initiatives_df["annual_co2_reduction_t"] * years
        initiatives_df["macc_eur_per_t"] = initiatives_df["capex_eur"] / initiatives_df["co2_total_t"].replace({0: np.nan})

        st.write(
            "Key point: initiatives are **Feasible / Screening / Not feasible** depending on data availability and applicability. "
            "For Screening initiatives, check **Missing data** and validate with the plant."
        )

        show_cols = [
            "id", "initiative", "status", "confidence", "missing_data", "block_reason",
            "capex_eur", "annual_opex_saving_eur", "annual_co2_reduction_t",
            "co2_value_eur_per_year", "annual_total_benefit_eur", "npv_eur",
            "payback_years", "macc_eur_per_t", "implementation_months", "strategic_score_1_5",
            "assumptions"
        ]
        st.dataframe(
            initiatives_df[show_cols].sort_values(["status", "npv_eur"], ascending=[True, False]),
            use_container_width=True
        )

        st.download_button(
            "Download Generated_Initiatives.csv",
            data=initiatives_df.to_csv(index=False).encode("utf-8"),
            file_name="Generated_Initiatives.csv",
            mime="text/csv",
        )

# -----------------------------
# TAB 4: optimization
# -----------------------------
with tab4:
    st.subheader("Portfolio optimization (MILP)")
    if initiatives_df is None:
        st.info("Generate initiatives first (Tab 3).")
    else:
        # Filter what can be optimized
        eligible = initiatives_df.copy()

        # Never include "Not feasible"
        eligible = eligible[eligible["status"] != "Not feasible"].copy()

        if not include_screening:
            eligible = eligible[eligible["status"] == "Feasible"].copy()

        if eligible.empty:
            st.warning("No eligible initiatives for optimization with current filters. Try enabling Screening initiatives.")
        else:
            sel_idx = optimize_portfolio(
                df=eligible,
                budget=budget,
                max_impl_months=max_impl_months if use_impl else None,
                min_annual_co2=min_annual_co2 if use_co2_target else None,
                weights=weights,
                apply_confidence_penalty=apply_conf_penalty,
            )

            out = eligible.copy()
            out["selected"] = 0
            out.loc[sel_idx, "selected"] = 1
            selected = out[out["selected"] == 1].copy()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Selected initiatives", f"{len(selected)}/{len(out)}")
            col2.metric("CAPEX used (€)", f"{selected['capex_eur'].sum():,.0f}")
            col3.metric("Portfolio NPV (€)", f"{selected['npv_eur'].sum():,.0f}")
            col4.metric("Annual CO₂ reduction (t/yr)", f"{selected['annual_co2_reduction_t'].sum():,.0f}")

            st.markdown("### Selected portfolio")
            st.dataframe(
                selected[[
                    "id", "initiative", "status", "confidence", "missing_data",
                    "capex_eur", "annual_opex_saving_eur", "annual_co2_reduction_t",
                    "npv_eur", "payback_years", "macc_eur_per_t",
                    "implementation_months", "strategic_score_1_5", "assumptions"
                ]].sort_values("npv_eur", ascending=False),
                use_container_width=True
            )

            st.markdown("### Full eligible list (with selection flag)")
            st.dataframe(
                out[[
                    "id", "initiative", "status", "confidence",
                    "capex_eur", "annual_opex_saving_eur", "annual_co2_reduction_t",
                    "npv_eur", "selected"
                ]].sort_values(["selected", "npv_eur"], ascending=[False, False]),
                use_container_width=True
            )

            export = out.copy()
            export["assump_horizon_years"] = years
            export["assump_discount_rate"] = discount_rate
            export["assump_co2_price_eur_per_t"] = co2_price
            export["constraint_budget_eur"] = budget
            export["constraint_max_impl_months_sum"] = max_impl_months if use_impl else np.nan
            export["constraint_min_annual_co2_t"] = min_annual_co2 if use_co2_target else np.nan
            export["w_npv"] = w_npv
            export["w_co2"] = w_co2
            export["w_strat"] = w_strat
            export["include_screening"] = include_screening
            export["apply_confidence_penalty"] = apply_confidence_penalty

            st.download_button(
                "Download Portfolio_Results.csv",
                data=export.to_csv(index=False).encode("utf-8"),
                file_name="Portfolio_Results.csv",
                mime="text/csv",
            )
