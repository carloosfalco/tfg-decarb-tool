# app.py
# Industrial Decarbonization Decision Tool (TFG prototype) — with:
# 1) Two modes: (A) Auto-propose initiatives from company inputs, (B) Upload client CSV
# 2) PESTEL mini-analysis (rule-based)
# 3) Robust CSV validation, metrics, confidence penalty
# 4) Portfolio optimization with PuLP
#
# Author: Carlos Falcó

from __future__ import annotations

import io
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import pulp


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Industrial Decarbonization Decision Tool",
    page_icon="🌿",
    layout="wide",
)


# -----------------------------
# Schema
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
    "confidence_0_1",       # optional direct confidence (0..1)
    "required_info",        # semicolon list of required fields to be "high confidence"
    "provided_info",        # semicolon list of info the client actually provided
    "initiative_family",    # e.g., Solar, Motors, Heat, etc.
    "data_dependency",      # Low/Medium/High
]

NUMERIC_COLUMNS = [
    "capex_eur",
    "annual_opex_saving_eur",
    "annual_co2_reduction_t",
    "implementation_months",
    "strategic_score_1_5",
    "confidence_0_1",
]


# -----------------------------
# Utility helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_read_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No file uploaded")
    raw = uploaded_file.getvalue()
    if raw is None or len(raw) == 0:
        raise ValueError("Uploaded file is empty")

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

    raise ValueError("Could not parse CSV. Check separators (comma/semicolon) and headers.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
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

    if "id" in df.columns and df["id"].isna().any():
        errors.append("Column 'id' has missing values.")
    if "initiative" in df.columns and df["initiative"].astype(str).str.strip().eq("").any():
        errors.append("Column 'initiative' has blank values.")

    return (len(errors) == 0, errors)


def parse_semicolon_list(s: str) -> List[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip().lower() for x in s.split(";") if x.strip()]


def apply_confidence_penalty(value: float, confidence: float, floor: float = 0.4) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if confidence is None or (isinstance(confidence, float) and np.isnan(confidence)):
        confidence = 0.6
    confidence = clamp(float(confidence), 0.0, 1.0)
    floor = clamp(float(floor), 0.0, 1.0)
    multiplier = floor + (1.0 - floor) * confidence
    return float(value) * multiplier


def infer_confidence_row(row: pd.Series) -> float:
    if "confidence_0_1" in row.index and pd.notna(row["confidence_0_1"]):
        return clamp(float(row["confidence_0_1"]), 0.0, 1.0)

    req = parse_semicolon_list(row.get("required_info", ""))
    prv = parse_semicolon_list(row.get("provided_info", ""))

    # If it requires info and none is provided => low confidence
    if req:
        if not prv:
            return 0.45
        overlap = len(set(req).intersection(set(prv)))
        ratio = overlap / max(1, len(set(req)))
        return clamp(0.4 + 0.55 * ratio, 0.0, 1.0)

    return 0.65


# -----------------------------
# PESTEL (rule-based)
# -----------------------------
def generate_pestel(company: Dict) -> Dict[str, List[str]]:
    """
    Rule-based, short PESTEL. No hallucinated facts.
    Uses: country_region, sector, energy_intensity, electricity_price_level, fossil_fuel_use, eu_context
    """
    sector = (company.get("sector") or "").lower()
    country = (company.get("country_region") or "").lower()
    eu = company.get("eu_context", True)

    energy_intensity = company.get("energy_intensity", "Medium")
    fossil_heat = company.get("fossil_heat_use", "Some")  # None/Some/High
    grid_emissions = company.get("grid_emissions_level", "Medium")  # Low/Medium/High

    p: List[str] = []
    e: List[str] = []
    s: List[str] = []
    t: List[str] = []
    en: List[str] = []
    l: List[str] = []

    # Political / Policy
    if eu:
        p.append("Creciente presión regulatoria y de reporting ESG (CSRD/ESRS) que empuja a definir hojas de ruta de descarbonización.")
        p.append("Acceso potencial a ayudas públicas/financiación verde si el proyecto es ‘bankable’ y medible.")
    else:
        p.append("Entorno regulatorio puede ser más heterogéneo; conviene mapear incentivos locales y requisitos de reporte.")

    # Economic
    if energy_intensity in ["high", "alto", "very high"]:
        e.append("Alta exposición a volatilidad de precios de energía: medidas de eficiencia suelen tener ROI rápido.")
    else:
        e.append("Las medidas ‘no-regret’ (eficiencia, control, mantenimiento) suelen ser la base económica del plan.")

    if fossil_heat.lower() in ["high", "alto"]:
        e.append("Coste de combustibles y riesgo de ‘carbon cost’ hacen atractivas soluciones de electrificación y recuperación de calor.")
    if grid_emissions.lower() == "low":
        e.append("Red eléctrica relativamente limpia mejora el caso de electrificación (más CO₂ evitado por kWh).")
    elif grid_emissions.lower() == "high":
        e.append("Si la red es intensiva en carbono, prioriza eficiencia y calor residual antes de electrificar masivamente.")

    # Social
    s.append("Clientes y cadenas de suministro demandan evidencias: métricas verificables y trazabilidad de resultados.")
    s.append("En planta, la adopción depende de seguridad, formación y cambios operativos (O&M).")

    # Technological
    t.append("Madurez alta en tecnologías ‘quick wins’: VFD/motores eficientes, aire comprimido, EMS/submetering, BMS.")
    t.append("Digitalización (medición + analítica) mejora verificación, priorización y mantenimiento predictivo.")
    if "food" in sector or "aliment" in sector:
        t.append("Procesos térmicos son críticos: opciones de recuperación de calor y bombas de calor industrial suelen ser relevantes.")
    if "cement" in sector or "steel" in sector or "acero" in sector or "cemento" in sector:
        t.append("Sectores hard-to-abate: conviene separar ‘eficiencia’ de ‘transformación’ (combustibles alternativos/CCUS), por fases.")

    # Environmental
    en.append("Reducción de emisiones y consumo energético suele correlacionar con reducción de costes y riesgo regulatorio.")
    en.append("Priorizar ‘no-regret’ y proyectos escalables ayuda a construir un roadmap por fases.")
    if fossil_heat.lower() in ["high", "alto"]:
        en.append("Foco especial en calor de proceso: es donde suele estar gran parte del CO₂ en industria.")

    # Legal
    if eu:
        l.append("Necesidad de documentación, supuestos, y trazabilidad de datos para auditoría/aseguramiento limitado.")
        l.append("Contratos energéticos (PPA, autoconsumo, etc.) requieren revisión legal y de permisos si aplica.")
    else:
        l.append("Requisitos de cumplimiento y permisos varían; imprescindible check legal local para generación in situ y modificaciones de proceso.")

    return {"Political": p, "Economic": e, "Social": s, "Technological": t, "Environmental": en, "Legal": l}


# -----------------------------
# Initiative proposal engine (rule-based)
# -----------------------------
def propose_initiatives(company: Dict, n: int = 8) -> pd.DataFrame:
    """
    Generates up to n initiatives based on company inputs.
    Uses conservative heuristics. If critical data missing => marks required_info and lowers confidence.
    """
    # Read company inputs
    annual_electricity_mwh = company.get("annual_electricity_mwh")
    annual_fuel_mwh = company.get("annual_fuel_mwh")  # thermal fuels
    electricity_price = company.get("electricity_price_eur_mwh")
    fuel_price = company.get("fuel_price_eur_mwh")
    co2_factor_elec = company.get("co2_factor_elec_t_per_mwh")
    co2_factor_fuel = company.get("co2_factor_fuel_t_per_mwh")

    has_compressed_air = company.get("has_compressed_air", True)
    has_process_heat = company.get("has_process_heat", True)
    roof_area_m2 = company.get("roof_area_m2")  # optional
    heat_waste_potential = company.get("waste_heat_potential", "Unknown")  # Low/Medium/High/Unknown
    sector = company.get("sector", "General industry")

    # Defaults if missing
    # (We do not "invent" scale; we only estimate if enough info exists)
    def ok_num(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x))

    initiatives = []

    next_id = 1

    # 1) EMS + Submetering + Analytics (low dependency)
    # Estimate: 2-5% electricity savings if electricity MWh known; else leave placeholders with low confidence
    req = "annual_electricity_mwh;electricity_price_eur_mwh"
    prv = []
    if ok_num(annual_electricity_mwh):
        prv.append("annual_electricity_mwh")
    if ok_num(electricity_price):
        prv.append("electricity_price_eur_mwh")

    if ok_num(annual_electricity_mwh) and ok_num(electricity_price):
        saving_eur = 0.03 * annual_electricity_mwh * electricity_price
        co2_red = (0.03 * annual_electricity_mwh * (co2_factor_elec or 0.0)) if ok_num(co2_factor_elec) else 0.0
    else:
        saving_eur = np.nan
        co2_red = np.nan

    initiatives.append({
        "id": next_id,
        "initiative_family": "Digital / EMS",
        "initiative": "Energy Management System (EMS) + submetering + analytics",
        "capex_eur": 100000,
        "annual_opex_saving_eur": saving_eur,
        "annual_co2_reduction_t": co2_red,
        "implementation_months": 4,
        "strategic_score_1_5": 5,
        "notes": "Base de medición para priorizar y verificar ahorros; reduce ‘data gaps’ del plan.",
        "required_info": req,
        "provided_info": ";".join(prv),
        "data_dependency": "Medium",
        "confidence_0_1": np.nan,
    })
    next_id += 1

    # 2) High efficiency motors + VFD (medium dependency)
    req = "motor_inventory;operating_hours;electricity_price_eur_mwh"
    initiatives.append({
        "id": next_id,
        "initiative_family": "Electric efficiency",
        "initiative": "High-efficiency motors + VFD (targeted replacements)",
        "capex_eur": 120000,
        "annual_opex_saving_eur": (0.02 * annual_electricity_mwh * electricity_price) if (ok_num(annual_electricity_mwh) and ok_num(electricity_price)) else np.nan,
        "annual_co2_reduction_t": (0.02 * annual_electricity_mwh * (co2_factor_elec or 0.0)) if (ok_num(annual_electricity_mwh) and ok_num(co2_factor_elec)) else np.nan,
        "implementation_months": 3,
        "strategic_score_1_5": 3,
        "notes": "Suele ser ‘quick win’ si hay cargas variables (bombas, ventiladores, compresores).",
        "required_info": req,
        "provided_info": "",
        "data_dependency": "Medium",
        "confidence_0_1": np.nan,
    })
    next_id += 1

    # 3) Compressed air leak program (if applicable)
    if has_compressed_air:
        req = "compressed_air_kwh;leak_rate;electricity_price_eur_mwh"
        initiatives.append({
            "id": next_id,
            "initiative_family": "Utilities",
            "initiative": "Compressed air leak program + pressure optimization + controls",
            "capex_eur": 60000,
            "annual_opex_saving_eur": (0.01 * annual_electricity_mwh * electricity_price) if (ok_num(annual_electricity_mwh) and ok_num(electricity_price)) else np.nan,
            "annual_co2_reduction_t": (0.01 * annual_electricity_mwh * (co2_factor_elec or 0.0)) if (ok_num(annual_electricity_mwh) and ok_num(co2_factor_elec)) else np.nan,
            "implementation_months": 2,
            "strategic_score_1_5": 2,
            "notes": "Alta dependencia de inventario/medición; fácil de ejecutar con auditoría y mantenimiento.",
            "required_info": req,
            "provided_info": "",
            "data_dependency": "High",
            "confidence_0_1": np.nan,
        })
        next_id += 1

    # 4) HVAC optimization + BMS
    req = "building_area_m2;hvac_profile;electricity_price_eur_mwh"
    initiatives.append({
        "id": next_id,
        "initiative_family": "Buildings",
        "initiative": "HVAC optimization + controls (BMS tuning / schedules)",
        "capex_eur": 150000,
        "annual_opex_saving_eur": (0.01 * annual_electricity_mwh * electricity_price) if (ok_num(annual_electricity_mwh) and ok_num(electricity_price)) else np.nan,
        "annual_co2_reduction_t": (0.01 * annual_electricity_mwh * (co2_factor_elec or 0.0)) if (ok_num(annual_electricity_mwh) and ok_num(co2_factor_elec)) else np.nan,
        "implementation_months": 3,
        "strategic_score_1_5": 3,
        "notes": "Más relevante si hay oficinas/almacenes climatizados o condiciones de calidad/humedad.",
        "required_info": req,
        "provided_info": "",
        "data_dependency": "Medium",
        "confidence_0_1": np.nan,
    })
    next_id += 1

    # 5) Waste heat recovery / heat pump (if process heat)
    if has_process_heat:
        req = "waste_heat_sources;heat_demand_profile;fuel_type;fuel_price_eur_mwh"
        # Only estimate if we have thermal fuel MWh and price and a declared potential
        if ok_num(annual_fuel_mwh) and ok_num(fuel_price) and heat_waste_potential.lower() in ["medium", "high"]:
            # Conservative: 5% fuel displacement
            saving_eur = 0.05 * annual_fuel_mwh * fuel_price
            co2_red = (0.05 * annual_fuel_mwh * (co2_factor_fuel or 0.0)) if ok_num(co2_factor_fuel) else np.nan
        else:
            saving_eur = np.nan
            co2_red = np.nan

        initiatives.append({
            "id": next_id,
            "initiative_family": "Heat",
            "initiative": "Waste heat recovery / industrial heat pump (feasibility-based)",
            "capex_eur": 250000,
            "annual_opex_saving_eur": saving_eur,
            "annual_co2_reduction_t": co2_red,
            "implementation_months": 6,
            "strategic_score_1_5": 4,
            "notes": "Muy sensible a datos de calor residual y perfil de demanda; suele requerir estudio de ingeniería.",
            "required_info": req,
            "provided_info": f"annual_fuel_mwh;fuel_price_eur_mwh;waste_heat_potential" if (ok_num(annual_fuel_mwh) and ok_num(fuel_price)) else "waste_heat_potential",
            "data_dependency": "High",
            "confidence_0_1": np.nan,
        })
        next_id += 1

    # 6) Process optimization (yield/scrap)
    req = "baseline_scrap_rate;production_volume;unit_cost;energy_intensity"
    initiatives.append({
        "id": next_id,
        "initiative_family": "Process",
        "initiative": "Process optimization (yield / scrap reduction program)",
        "capex_eur": 90000,
        "annual_opex_saving_eur": np.nan,  # too company-specific unless scrap data provided
        "annual_co2_reduction_t": np.nan,
        "implementation_months": 5,
        "strategic_score_1_5": 4,
        "notes": "Potencial alto pero muy dependiente de datos de merma, producto y energía por unidad.",
        "required_info": req,
        "provided_info": "",
        "data_dependency": "High",
        "confidence_0_1": np.nan,
    })
    next_id += 1

    # 7) Solar PV (only if roof area or land stated)
    req = "roof_area_m2;location;orientation;annual_kwh;electricity_price_eur_mwh"
    prv = []
    if ok_num(roof_area_m2):
        prv.append("roof_area_m2")
    if ok_num(annual_electricity_mwh):
        prv.append("annual_kwh")
    if ok_num(electricity_price):
        prv.append("electricity_price_eur_mwh")

    # Only estimate if roof area AND electricity price known (very conservative)
    if ok_num(roof_area_m2) and ok_num(electricity_price):
        # Conservative: 0.18 kWp per m2, 1400 kWh/kWp-year => kWh = area*0.18*1400
        annual_kwh = roof_area_m2 * 0.18 * 1400.0
        annual_mwh = annual_kwh / 1000.0
        # Assume self-consumption limited: 70% used
        used_mwh = 0.70 * annual_mwh
        saving_eur = used_mwh * electricity_price
        co2_red = used_mwh * (co2_factor_elec or 0.0) if ok_num(co2_factor_elec) else np.nan
    else:
        saving_eur = np.nan
        co2_red = np.nan

    initiatives.append({
        "id": next_id,
        "initiative_family": "Renewables",
        "initiative": "Solar PV self-consumption (roof/ground, feasibility-based sizing)",
        "capex_eur": 225000,
        "annual_opex_saving_eur": saving_eur,
        "annual_co2_reduction_t": co2_red,
        "implementation_months": 5,
        "strategic_score_1_5": 4,
        "notes": "Si faltan datos (superficie, localización, perfil de carga) la herramienta NO asume: lo marca como low-confidence.",
        "required_info": req,
        "provided_info": ";".join(prv),
        "data_dependency": "High",
        "confidence_0_1": np.nan,
    })
    next_id += 1

    # 8) Partial electrification (depends on heat demand and grid)
    req = "heat_demand_profile;temperature_levels;grid_capacity;electricity_price_eur_mwh;fuel_price_eur_mwh"
    initiatives.append({
        "id": next_id,
        "initiative_family": "Heat",
        "initiative": "Electrification (partial) of low/medium-temperature heat demand",
        "capex_eur": 420000,
        "annual_opex_saving_eur": np.nan,
        "annual_co2_reduction_t": np.nan,
        "implementation_months": 7,
        "strategic_score_1_5": 4,
        "notes": "Depende del nivel de temperatura, perfil de demanda, capacidad de red y comparativa electricidad/combustible.",
        "required_info": req,
        "provided_info": "",
        "data_dependency": "High",
        "confidence_0_1": np.nan,
    })
    next_id += 1

    df = pd.DataFrame(initiatives)

    # Keep up to n initiatives
    df = df.head(n).copy()

    # Ensure required cols exist
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    # Standardize notes/strings
    df["notes"] = df["notes"].fillna("").astype(str)

    return df


# -----------------------------
# Metrics + Optimization
# -----------------------------
def compute_metrics(
    df: pd.DataFrame,
    horizon_years: int,
    discount_rate: float,
    co2_price: float,
    confidence_floor: float,
) -> pd.DataFrame:
    df = df.copy()

    # Ensure optional columns
    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    # Infer confidence
    df["confidence"] = df.apply(infer_confidence_row, axis=1)

    # CO2 value and benefit
    df["co2_value_eur_per_year"] = df["annual_co2_reduction_t"] * co2_price
    df["total_annual_benefit_eur"] = df["annual_opex_saving_eur"] + df["co2_value_eur_per_year"]

    # Implementation delay in years
    df["implementation_years"] = df["implementation_months"] / 12.0

    def npv_row(r: pd.Series) -> float:
        capex = r["capex_eur"]
        benefit = r["total_annual_benefit_eur"]
        if pd.isna(capex) or pd.isna(benefit):
            return np.nan
        if benefit <= 0:
            return -float(capex)

        delay = float(r["implementation_years"]) if pd.notna(r["implementation_years"]) else 0.0
        start_year = int(math.floor(delay)) + 1

        npv = -float(capex)
        for t in range(start_year, horizon_years + 1):
            npv += float(benefit) / ((1.0 + discount_rate) ** t)
        return npv

    df["npv_eur"] = df.apply(npv_row, axis=1)

    df["payback_years"] = np.where(
        df["total_annual_benefit_eur"] > 0,
        df["capex_eur"] / df["total_annual_benefit_eur"],
        np.nan,
    )

    df["npv_penalized_eur"] = df.apply(
        lambda r: apply_confidence_penalty(r["npv_eur"], r["confidence"], confidence_floor),
        axis=1,
    )

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
    df = df.copy()

    df["capex_eur"] = df["capex_eur"].fillna(0.0)
    df["annual_co2_reduction_t"] = df["annual_co2_reduction_t"].fillna(0.0)
    df["npv_penalized_eur"] = df["npv_penalized_eur"].fillna(-1e9)
    df["strategic_score_1_5"] = df["strategic_score_1_5"].fillna(3.0)

    model = pulp.LpProblem("DecarbPortfolio", pulp.LpMaximize)

    ids = df["id"].astype(str).tolist()
    x = pulp.LpVariable.dicts("x", ids, lowBound=0, upBound=1, cat=pulp.LpBinary)

    model += pulp.lpSum(
        x[i] * float(df.loc[df["id"].astype(str) == i, "capex_eur"].iloc[0]) for i in ids
    ) <= float(budget_eur)

    if min_co2_t and min_co2_t > 0:
        model += pulp.lpSum(
            x[i] * float(df.loc[df["id"].astype(str) == i, "annual_co2_reduction_t"].iloc[0]) for i in ids
        ) >= float(min_co2_t)

    if objective == "Maximize penalized NPV":
        model += pulp.lpSum(
            x[i] * float(df.loc[df["id"].astype(str) == i, "npv_penalized_eur"].iloc[0]) for i in ids
        )
    elif objective == "Maximize CO2 reduction":
        model += pulp.lpSum(
            x[i] * float(df.loc[df["id"].astype(str) == i, "annual_co2_reduction_t"].iloc[0]) for i in ids
        )
    else:
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
                "initiative": "Solar PV self-consumption (roof/ground, feasibility-based sizing)",
                "capex_eur": 400000,
                "annual_opex_saving_eur": 95000,
                "annual_co2_reduction_t": 250,
                "implementation_months": 4,
                "strategic_score_1_5": 3,
                "notes": "If applicable; depends on roof/land availability and load profile.",
                "required_info": "roof_area_m2;location;orientation;annual_kwh;electricity_price_eur_mwh",
                "provided_info": "",
                "confidence_0_1": 0.55,
                "initiative_family": "Renewables",
                "data_dependency": "High",
            },
            {
                "id": 2,
                "initiative": "High efficiency motors + VFD (targeted replacements)",
                "capex_eur": 120000,
                "annual_opex_saving_eur": 45000,
                "annual_co2_reduction_t": 90,
                "implementation_months": 3,
                "strategic_score_1_5": 2,
                "notes": "Usually easier to quantify with motor inventory and operating hours.",
                "required_info": "motor_inventory;operating_hours;electricity_price_eur_mwh",
                "provided_info": "",
                "confidence_0_1": 0.70,
                "initiative_family": "Electric efficiency",
                "data_dependency": "Medium",
            },
        ]
    )
    return example.to_csv(index=False).encode("utf-8")


# -----------------------------
# UI
# -----------------------------
st.title("Industrial Decarbonization Decision Tool")
st.caption("Prototype TFG – Carlos Falcó")

with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Choose workflow",
        ["A) Propose initiatives from company inputs", "B) Upload client initiatives CSV"],
        index=0,
    )

    st.divider()
    st.header("Financial assumptions")
    horizon_years = st.slider("Project horizon (years)", 1, 10, 5, 1)
    discount_rate_pct = st.slider("Discount rate (%)", 0.0, 25.0, 8.0, 0.25)
    discount_rate = discount_rate_pct / 100.0
    co2_price = st.number_input("CO₂ price (€/t)", min_value=0.0, value=80.0, step=5.0)

    st.divider()
    st.header("Optimization constraints")
    budget_eur = st.number_input("CAPEX budget (€)", min_value=0.0, value=600000.0, step=10000.0)
    min_co2_t = st.number_input("Minimum annual CO₂ target (t/year) [optional]", min_value=0.0, value=0.0, step=10.0)

    st.divider()
    st.header("Confidence handling")
    confidence_floor = st.slider("Minimum multiplier at low confidence", 0.0, 1.0, 0.4, 0.05)

    st.divider()
    st.header("Objective")
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
    st.header("Client template")
    st.download_button(
        "Download CSV template (example)",
        data=template_csv_bytes(),
        file_name="client_initiatives_template.csv",
        mime="text/csv",
        use_container_width=True,
    )


# -----------------------------
# Mode A — Propose from inputs
# -----------------------------
df_base = None
company_inputs = {}

if mode.startswith("A"):
    st.markdown("### 1) Company & sector inputs (to propose initiatives)")
    c1, c2, c3 = st.columns(3)

    with c1:
        company_inputs["company_name"] = st.text_input("Company name (optional)", value="")
        company_inputs["country_region"] = st.text_input("Country/region (e.g., Spain, EU)", value="Spain / EU")
        company_inputs["eu_context"] = st.checkbox("EU regulatory context (CSRD/ESRS relevant)", value=True)
        company_inputs["sector"] = st.selectbox(
            "Sector (approx.)",
            ["General industry", "Food & beverage", "Chemicals", "Metals/Steel", "Cement", "Paper", "Logistics/Warehousing", "Other"],
            index=0,
        )

    with c2:
        company_inputs["energy_intensity"] = st.selectbox("Energy intensity", ["Low", "Medium", "High"], index=1)
        company_inputs["grid_emissions_level"] = st.selectbox("Grid emissions level (rough)", ["Low", "Medium", "High"], index=1)
        company_inputs["fossil_heat_use"] = st.selectbox("Fossil fuel use for process heat", ["None", "Some", "High"], index=1)

        company_inputs["has_compressed_air"] = st.checkbox("Uses compressed air systems", value=True)
        company_inputs["has_process_heat"] = st.checkbox("Has process heat demand (steam/hot water/hot air)", value=True)

    with c3:
        st.write("Optional quantitative inputs (if you have them)")
        company_inputs["annual_electricity_mwh"] = st.number_input("Annual electricity (MWh)", min_value=0.0, value=0.0, step=100.0)
        company_inputs["annual_fuel_mwh"] = st.number_input("Annual thermal fuels for heat (MWh)", min_value=0.0, value=0.0, step=100.0)
        company_inputs["electricity_price_eur_mwh"] = st.number_input("Electricity price (€/MWh)", min_value=0.0, value=0.0, step=5.0)
        company_inputs["fuel_price_eur_mwh"] = st.number_input("Fuel price (€/MWh)", min_value=0.0, value=0.0, step=5.0)
        company_inputs["co2_factor_elec_t_per_mwh"] = st.number_input("CO₂ factor electricity (tCO₂/MWh) [optional]", min_value=0.0, value=0.0, step=0.01)
        company_inputs["co2_factor_fuel_t_per_mwh"] = st.number_input("CO₂ factor fuel (tCO₂/MWh) [optional]", min_value=0.0, value=0.0, step=0.01)

    st.markdown("### 2) Site constraints (only if known)")
    c4, c5 = st.columns(2)
    with c4:
        company_inputs["roof_area_m2"] = st.number_input("Available roof area (m²) [optional]", min_value=0.0, value=0.0, step=100.0)
    with c5:
        company_inputs["waste_heat_potential"] = st.selectbox("Waste heat potential (rough)", ["Unknown", "Low", "Medium", "High"], index=0)

    # Convert zeros to None for "unknown" semantics (avoid pretending it's real)
    for k in ["annual_electricity_mwh", "annual_fuel_mwh", "electricity_price_eur_mwh", "fuel_price_eur_mwh",
              "co2_factor_elec_t_per_mwh", "co2_factor_fuel_t_per_mwh", "roof_area_m2"]:
        if isinstance(company_inputs.get(k), (int, float)) and company_inputs.get(k) == 0.0:
            company_inputs[k] = None

    st.markdown("### 3) PESTEL (brief)")
    pestel = generate_pestel(company_inputs)
    pcols = st.columns(3)
    keys = list(pestel.keys())
    for i, k in enumerate(keys):
        with pcols[i % 3]:
            st.subheader(k)
            for bullet in pestel[k]:
                st.write(f"- {bullet}")

    st.markdown("### 4) Proposed initiatives (editable before optimization)")
    n_inits = st.slider("Number of proposed initiatives", 4, 10, 8, 1)

    df_base = propose_initiatives(company_inputs, n=n_inits)
    df_base = normalize_columns(df_base)
    df_base = coerce_numeric(df_base)

    # Allow user to edit the table (so consultant can adjust CAPEX, savings, etc.)
    st.info("Puedes editar valores antes de optimizar (CAPEX, ahorros, CO₂...). Si faltan datos, deja en blanco: se penaliza por confianza.")
    edited = st.data_editor(
        df_base,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
    )
    df_base = pd.DataFrame(edited)


# -----------------------------
# Mode B — Upload CSV
# -----------------------------
if mode.startswith("B"):
    st.markdown("### 1) Upload initiatives CSV")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV (you can start from the template).")
        st.stop()

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

    for c in OPTIONAL_COLUMNS:
        if c not in df_raw.columns:
            df_raw[c] = ""

    df_base = df_raw


# -----------------------------
# Compute metrics + Optimize
# -----------------------------
if df_base is None or len(df_base) == 0:
    st.stop()

# Ensure required columns exist even after edits
for c in REQUIRED_COLUMNS:
    if c not in df_base.columns:
        df_base[c] = np.nan

for c in OPTIONAL_COLUMNS:
    if c not in df_base.columns:
        df_base[c] = ""

df_base = coerce_numeric(df_base)

df = compute_metrics(
    df_base,
    horizon_years=horizon_years,
    discount_rate=discount_rate,
    co2_price=co2_price,
    confidence_floor=confidence_floor,
)

st.markdown("### 5) Initiative evaluation")
cols_to_show = [
    "id",
    "initiative_family",
    "initiative",
    "data_dependency",
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
    "required_info",
    "provided_info",
    "notes",
]
cols_to_show = [c for c in cols_to_show if c in df.columns]
st.dataframe(df[cols_to_show], use_container_width=True, hide_index=True)

# Optimization
st.markdown("### 6) Portfolio optimization")
df_opt, summary = optimize_portfolio(
    df=df,
    budget_eur=budget_eur,
    min_co2_t=min_co2_t,
    objective=objective,
    w_npv=w_npv,
    w_co2=w_co2,
    w_strategy=w_strategy,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Status", summary["status"])
c2.metric("Selected initiatives", summary["selected_count"])
c3.metric("CAPEX selected (€)", f"{summary['capex_selected']:,.0f}")
c4.metric("CO₂ reduction selected (t/y)", f"{summary['co2_selected']:,.1f}")
st.metric("Total penalized NPV selected (€)", f"{summary['npv_selected']:,.0f}")

st.markdown("#### Selected initiatives")
selected_df = df_opt[df_opt["selected"]].copy()
st.dataframe(selected_df[cols_to_show], use_container_width=True, hide_index=True)

# Visuals
st.markdown("### 7) Visuals")
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

# Export
st.markdown("### 8) Export results")
export_cols = [
    "id",
    "initiative_family",
    "initiative",
    "data_dependency",
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
    "required_info",
    "provided_info",
    "notes",
]
export_cols = [c for c in export_cols if c in df_opt.columns]
export = df_opt[export_cols].copy()

st.download_button(
    "Download portfolio results (CSV)",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="portfolio_results.csv",
    mime="text/csv",
    use_container_width=True,
)

st.success("Ready ✅ Cambia inputs / edita iniciativas / re-subir CSV y se recalcula automáticamente.")
