
# app.py
# Industrial Decarbonization Decision Tool (TFG prototype) — with:
# 1) Two modes: (A) Auto-propose initiatives from company inputs, (B) Upload client CSV
# 2) PESTEL mini-analysis (rule-based)
# 3) Robust CSV validation, metrics, confidence penalty
# 4) Portfolio optimization with PuLP
# 5) Plotly charts hardened against NaN/inf/negative sizes
#
# Author: Carlos Falcó

from __future__ import annotations

import io
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

import pulp

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Industrial Decarbonization Decision Tool",
    page_icon="",
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
    sector = (company.get("sector") or "").lower()
    eu = bool(company.get("eu_context", True))

    energy_intensity = (company.get("energy_intensity") or "Medium").lower()
    fossil_heat = (company.get("fossil_heat_use") or "Some").lower()
    grid_emissions = (company.get("grid_emissions_level") or "Medium").lower()

    p: List[str] = []
    e: List[str] = []
    s: List[str] = []
    t: List[str] = []
    en: List[str] = []
    l: List[str] = []

    if eu:
        p.append("Creciente presión regulatoria y de reporting ESG (CSRD/ESRS) que empuja a definir hojas de ruta de descarbonización.")
        p.append("Acceso potencial a ayudas públicas/financiación verde si el proyecto es medible y verificable.")
    else:
        p.append("Entorno regulatorio más heterogéneo; conviene mapear incentivos locales y requisitos de reporte.")

    if energy_intensity in ["high", "alto", "very high"]:
        e.append("Alta exposición a volatilidad de precios de energía: medidas de eficiencia suelen tener ROI rápido.")
    else:
        e.append("Medidas ‘no-regret’ (eficiencia, control, mantenimiento) suelen ser la base económica del plan.")

    if fossil_heat in ["high", "alto"]:
        e.append("Riesgo de coste de carbono y combustibles: atractivas soluciones de electrificación y recuperación de calor.")
    if grid_emissions == "low":
        e.append("Red eléctrica relativamente limpia mejora el caso de electrificación (más CO₂ evitado por kWh).")
    elif grid_emissions == "high":
        e.append("Si la red es intensiva en carbono, prioriza eficiencia y calor residual antes de electrificar masivamente.")

    s.append("Clientes y cadenas de suministro demandan evidencias: métricas verificables y trazabilidad de resultados.")
    s.append("En planta, la adopción depende de seguridad, formación y cambios operativos (O&M).")

    t.append("Madurez alta en tecnologías ‘quick wins’: VFD/motores, aire comprimido, EMS/submetering, control y automatización.")
    t.append("Digitalización (medición + analítica) mejora verificación, priorización y mantenimiento predictivo.")
    if "food" in sector or "aliment" in sector:
        t.append("Procesos térmicos críticos: recuperación de calor y bombas de calor industrial suelen ser relevantes.")
    if any(x in sector for x in ["cement", "cemento", "steel", "acero"]):
        t.append("Sector hard-to-abate: separar ‘eficiencia’ de ‘transformación’ (combustibles alternativos/CCUS), por fases.")

    en.append("Reducir emisiones y consumo energético suele correlacionar con reducción de costes y riesgo regulatorio.")
    en.append("Priorizar ‘no-regret’ y proyectos escalables ayuda a construir un roadmap por fases.")
    if fossil_heat in ["high", "alto"]:
        en.append("Foco especial en calor de proceso: suele concentrar gran parte del CO₂ en industria.")

    if eu:
        l.append("Documentación, supuestos y trazabilidad de datos para auditoría/aseguramiento limitado.")
        l.append("Permisos/contratos energéticos (autoconsumo, PPA, etc.) pueden requerir revisión legal.")
    else:
        l.append("Cumplimiento y permisos varían; imprescindible check legal local para generación in situ y cambios de proceso.")

    return {"Political": p, "Economic": e, "Social": s, "Technological": t, "Environmental": en, "Legal": l}


# -----------------------------
# Initiative proposal engine (rule-based)
# -----------------------------
def propose_initiatives(company: Dict, n: int = 8) -> pd.DataFrame:
    annual_electricity_mwh = company.get("annual_electricity_mwh")
    annual_fuel_mwh = company.get("annual_fuel_mwh")
    electricity_price = company.get("electricity_price_eur_mwh")
    fuel_price = company.get("fuel_price_eur_mwh")
    co2_factor_elec = company.get("co2_factor_elec_t_per_mwh")
    co2_factor_fuel = company.get("co2_factor_fuel_t_per_mwh")

    has_compressed_air = bool(company.get("has_compressed_air", True))
    has_process_heat = bool(company.get("has_process_heat", True))
    roof_area_m2 = company.get("roof_area_m2")
    heat_waste_potential = (company.get("waste_heat_potential") or "Unknown")

    def ok_num(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x))

    initiatives = []
    next_id = 1

    # EMS + submetering
    req = "annual_electricity_mwh;electricity_price_eur_mwh"
    prv = []
    if ok_num(annual_electricity_mwh):
        prv.append("annual_electricity_mwh")
    if ok_num(electricity_price):
        prv.append("electricity_price_eur_mwh")

    if ok_num(annual_electricity_mwh) and ok_num(electricity_price):
        saving_eur = 0.03 * annual_electricity_mwh * electricity_price
        co2_red = (0.03 * annual_electricity_mwh * (co2_factor_elec or 0.0)) if ok_num(co2_factor_elec) else np.nan
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

    # Motors + VFD
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
        "required_info": "motor_inventory;operating_hours;electricity_price_eur_mwh",
        "provided_info": "",
        "data_dependency": "Medium",
        "confidence_0_1": np.nan,
    })
    next_id += 1

    # Compressed air
    if has_compressed_air:
        initiatives.append({
            "id": next_id,
            "initiative_family": "Utilities",
            "initiative": "Compressed air leak program + pressure optimization + controls",
            "capex_eur": 60000,
            "annual_opex_saving_eur": (0.01 * annual_electricity_mwh * electricity_price) if (ok_num(annual_electricity_mwh) and ok_num(electricity_price)) else np.nan,
            "annual_co2_reduction_t": (0.01 * annual_electricity_mwh * (co2_factor_elec or 0.0)) if (ok_num(annual_electricity_mwh) and ok_num(co2_factor_elec)) else np.nan,
            "implementation_months": 2,
            "strategic_score_1_5": 2,
            "notes": "Alta dependencia de inventario/medición; se confirma con auditoría.",
            "required_info": "compressed_air_kwh;leak_rate;electricity_price_eur_mwh",
            "provided_info": "",
            "data_dependency": "High",
            "confidence_0_1": np.nan,
        })
        next_id += 1

    # HVAC / BMS
    initiatives.append({
        "id": next_id,
        "initiative_family": "Buildings",
        "initiative": "HVAC optimization + controls (BMS tuning / schedules)",
        "capex_eur": 150000,
        "annual_opex_saving_eur": (0.01 * annual_electricity_mwh * electricity_price) if (ok_num(annual_electricity_mwh) and ok_num(electricity_price)) else np.nan,
        "annual_co2_reduction_t": (0.01 * annual_electricity_mwh * (co2_factor_elec or 0.0)) if (ok_num(annual_electricity_mwh) and ok_num(co2_factor_elec)) else np.nan,
        "implementation_months": 3,
        "strategic_score_1_5": 3,
        "notes": "Más relevante si hay oficinas/almacenes climatizados o requisitos de humedad/calidad.",
        "required_info": "building_area_m2;hvac_profile;electricity_price_eur_mwh",
        "provided_info": "",
        "data_dependency": "Medium",
        "confidence_0_1": np.nan,
    })
    next_id += 1

    # Waste heat recovery / heat pump
    if has_process_heat:
        if ok_num(annual_fuel_mwh) and ok_num(fuel_price) and str(heat_waste_potential).lower() in ["medium", "high"]:
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
            "notes": "Muy sensible a datos de calor residual y perfil de demanda; requiere estudio de ingeniería.",
            "required_info": "waste_heat_sources;heat_demand_profile;fuel_type;fuel_price_eur_mwh",
            "provided_info": "waste_heat_potential" if heat_waste_potential else "",
            "data_dependency": "High",
            "confidence_0_1": np.nan,
        })
        next_id += 1

    # Process yield/scrap
    initiatives.append({
        "id": next_id,
        "initiative_family": "Process",
        "initiative": "Process optimization (yield / scrap reduction program)",
        "capex_eur": 90000,
        "annual_opex_saving_eur": np.nan,
        "annual_co2_reduction_t": np.nan,
        "implementation_months": 5,
        "strategic_score_1_5": 4,
        "notes": "Potencial alto pero muy dependiente de datos de merma, producto y energía por unidad.",
        "required_info": "baseline_scrap_rate;production_volume;unit_cost;energy_intensity",
        "provided_info": "",
        "data_dependency": "High",
        "confidence_0_1": np.nan,
    })
    next_id += 1

    # Solar PV
    prv = []
    if ok_num(roof_area_m2):
        prv.append("roof_area_m2")
    if ok_num(electricity_price):
        prv.append("electricity_price_eur_mwh")
    if ok_num(roof_area_m2) and ok_num(electricity_price):
        annual_kwh = roof_area_m2 * 0.18 * 1400.0
        annual_mwh = annual_kwh / 1000.0
        used_mwh = 0.70 * annual_mwh
        saving_eur = used_mwh * electricity_price
        co2_red = used_mwh * (co2_factor_elec or 0.0) if ok_num(co2_factor_elec) else np.nan
    else:
        saving_eur = np.nan
        co2_red = np.nan

    initiatives.append({
        "id": next_id,
        "initiative_family": "Renewables",
        "initiative": "Solar PV self-consumption (feasibility-based sizing)",
        "capex_eur": 225000,
        "annual_opex_saving_eur": saving_eur,
        "annual_co2_reduction_t": co2_red,
        "implementation_months": 5,
        "strategic_score_1_5": 4,
        "notes": "Si faltan datos (superficie, localización, perfil de carga) NO se asume: se marca low-confidence.",
        "required_info": "roof_area_m2;location;orientation;annual_kwh;electricity_price_eur_mwh",
        "provided_info": ";".join(prv),
        "data_dependency": "High",
        "confidence_0_1": np.nan,
    })
    next_id += 1

    # Electrification (partial)
    initiatives.append({
        "id": next_id,
        "initiative_family": "Heat",
        "initiative": "Electrification (partial) of low/medium-temperature heat demand",
        "capex_eur": 420000,
        "annual_opex_saving_eur": np.nan,
        "annual_co2_reduction_t": np.nan,
        "implementation_months": 7,
        "strategic_score_1_5": 4,
        "notes": "Depende de temperatura, perfil de demanda, capacidad de red y comparativa electricidad/combustible.",
        "required_info": "heat_demand_profile;temperature_levels;grid_capacity;electricity_price_eur_mwh;fuel_price_eur_mwh",
        "provided_info": "",
        "data_dependency": "High",
        "confidence_0_1": np.nan,
    })
    next_id += 1

    df = pd.DataFrame(initiatives).head(n).copy()

    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = ""

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

    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    df["confidence"] = df.apply(infer_confidence_row, axis=1)

    df["co2_value_eur_per_year"] = df["annual_co2_reduction_t"] * co2_price
    df["total_annual_benefit_eur"] = df["annual_opex_saving_eur"] + df["co2_value_eur_per_year"]

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
                "initiative": "Solar PV self-consumption (feasibility-based sizing)",
                "capex_eur": 400000,
                "annual_opex_saving_eur": 95000,
                "annual_co2_reduction_t": 250,
                "implementation_months": 4,
                "strategic_score_1_5": 3,
                "notes": "Depends on roof/land availability and load profile.",
                "required_info": "roof_area_m2;location;orientation;annual_kwh;electricity_price_eur_mwh",
                "provided_info": "",
                "confidence_0_1": 0.55,
                "initiative_family": "Renewables",
                "data_dependency": "High",
            },
            {
                "id": 2,
                "initiative": "High-efficiency motors + VFD (targeted replacements)",
                "capex_eur": 120000,
                "annual_opex_saving_eur": 45000,
                "annual_co2_reduction_t": 90,
                "implementation_months": 3,
                "strategic_score_1_5": 2,
                "notes": "Easier to quantify with motor inventory and operating hours.",
                "required_info": "motor_inventory;operating_hours;electricity_price_eur_mwh",
                "provided_info": "",
                "confidence_0_1": 0.70,
                "initiative_family": "Electric efficiency",
                "data_dependency": "Medium",
            },
        ]
    )
    return example.to_csv(index=False).encode("utf-8")


def df_for_ai(df: pd.DataFrame, limit: int = 25) -> List[dict]:
    if df is None or df.empty:
        return []
    cols = [
        "id",
        "initiative_family",
        "initiative",
        "selected",
        "capex_eur",
        "annual_opex_saving_eur",
        "annual_co2_reduction_t",
        "npv_penalized_eur",
        "payback_years",
        "confidence",
        "data_dependency",
    ]
    cols = [c for c in cols if c in df.columns]
    sample = df[cols].head(limit).copy()
    sample = sample.replace([np.inf, -np.inf], np.nan).where(pd.notna(sample), None)
    return sample.to_dict(orient="records")


def generate_ai_brief(
    provider: str,
    api_key: str,
    model: str,
    mode: str,
    summary: dict,
    df_all: pd.DataFrame,
    df_selected: pd.DataFrame,
    pestel: Dict[str, List[str]] | None,
    extra_prompt: str,
) -> str:
    if not api_key:
        raise RuntimeError(f"Missing {provider} API key.")

    context = {
        "mode": mode,
        "optimization_summary": summary,
        "selected_count": int(df_selected.shape[0]) if df_selected is not None else 0,
        "all_initiatives_preview": df_for_ai(df_all, limit=25),
        "selected_initiatives_preview": df_for_ai(df_selected, limit=25),
        "pestel": pestel or {},
    }

    system_prompt = (
        "You are an industrial decarbonization consultant. "
        "Provide concise, practical recommendations grounded only in the provided data. "
        "Do not invent numeric facts. If data gaps exist, call them out explicitly."
    )
    user_prompt = (
        "Create an executive brief in Spanish with:\n"
        "1) portfolio diagnosis,\n"
        "2) top 3 actions for next 90 days,\n"
        "3) main risks/data gaps,\n"
        "4) 3 scenario sensitivities to test.\n\n"
        f"Additional user instruction: {extra_prompt or 'None'}\n\n"
        f"DATA:\n{json.dumps(context, ensure_ascii=False)}"
    )

    provider_norm = (provider or "").strip().lower()
    if provider_norm == "openai":
        if OpenAI is None:
            raise RuntimeError("OpenAI package not installed. Add 'openai' to requirements and install dependencies.")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    if provider_norm == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
        }
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Gemini API error ({r.status_code}): {r.text[:400]}")
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join([str(p.get("text", "")) for p in parts]).strip()
        if not text:
            raise RuntimeError("Gemini returned empty text.")
        return text

    raise RuntimeError(f"Unsupported provider: {provider}")


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

    st.divider()
    st.header("AI Copilot (optional)")
    ai_enabled = st.checkbox("Enable AI assistant", value=False)
    ai_provider = st.selectbox("Provider", ["OpenAI", "Gemini"], index=0)
    default_api_key = ""
    try:
        if ai_provider == "Gemini":
            default_api_key = st.secrets.get("GEMINI_API_KEY", "")
        else:
            default_api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        default_api_key = ""
    ai_api_key = st.text_input(f"{ai_provider} API key", value=default_api_key, type="password")
    ai_model_default = "gpt-4o-mini" if ai_provider == "OpenAI" else "gemini-1.5-flash"
    ai_model = st.text_input("Model", value=ai_model_default)


# -----------------------------
# Data source selection
# -----------------------------
df_base = None
company_inputs: Dict = {}
pestel: Dict[str, List[str]] = {}

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

    # Convert zeros to None (unknown)
    for k in [
        "annual_electricity_mwh", "annual_fuel_mwh", "electricity_price_eur_mwh", "fuel_price_eur_mwh",
        "co2_factor_elec_t_per_mwh", "co2_factor_fuel_t_per_mwh", "roof_area_m2"
    ]:
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

    st.info("Puedes editar valores antes de optimizar. Si faltan datos, deja en blanco: se penaliza por confianza.")
    df_base = pd.DataFrame(
        st.data_editor(df_base, use_container_width=True, hide_index=True, num_rows="fixed")
    )

else:
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

# Ensure required cols exist
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

# -----------------------------
# Charts (hardened)
# -----------------------------
st.markdown("### 7) Visuals")

chart_df = df_opt.copy()
chart_df["selected_label"] = np.where(chart_df["selected"], "Selected", "Not selected")

# --- Plot-safe scatter ---
plot_df = chart_df.copy()
for col in ["capex_eur", "annual_co2_reduction_t", "total_annual_benefit_eur"]:
    if col in plot_df.columns:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

plot_df["size_for_plot"] = plot_df["total_annual_benefit_eur"].fillna(0.0)
plot_df["size_for_plot"] = plot_df["size_for_plot"].clip(lower=0.0)

if plot_df["size_for_plot"].max() <= 0:
    plot_df["size_for_plot"] = 1.0

plot_df = plot_df.dropna(subset=["capex_eur", "annual_co2_reduction_t"])

if plot_df.empty:
    st.warning("No hay suficientes datos numéricos para graficar CAPEX vs CO₂. Completa/edita esas columnas.")
else:
    fig1 = px.scatter(
        plot_df,
        x="capex_eur",
        y="annual_co2_reduction_t",
        size="size_for_plot",
        color="selected_label",
        hover_name="initiative",
        title="CAPEX vs CO₂ reduction (bubble size = annual benefit)",
        size_max=40,
    )
    st.plotly_chart(fig1, use_container_width=True)

# --- Plot-safe bar ---
bar_df = chart_df.copy()
bar_df["npv_penalized_eur"] = pd.to_numeric(bar_df.get("npv_penalized_eur", 0.0), errors="coerce").fillna(0.0)

fig2 = px.bar(
    bar_df.sort_values("npv_penalized_eur", ascending=False),
    x="initiative",
    y="npv_penalized_eur",
    color="selected_label",
    title="Penalized NPV by initiative",
)
fig2.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# AI Copilot
# -----------------------------
st.markdown("### 8) AI copilot")
ai_extra_prompt = st.text_area(
    "Optional instruction for AI (e.g., focus on cash flow, regulatory risk, or operations):",
    value="Prioriza recomendaciones accionables y con bajo riesgo de implementación.",
)
if st.button("Generate AI executive brief", use_container_width=True):
    if not ai_enabled:
        st.warning("Enable AI assistant in the sidebar first.")
    else:
        try:
            with st.spinner("Generating AI brief..."):
                ai_text = generate_ai_brief(
                    provider=ai_provider,
                    api_key=ai_api_key.strip(),
                    model=ai_model.strip(),
                    mode=mode,
                    summary=summary,
                    df_all=df_opt,
                    df_selected=selected_df,
                    pestel=pestel,
                    extra_prompt=ai_extra_prompt.strip(),
                )
            st.success("AI brief generated.")
            st.markdown(ai_text if ai_text else "_No content returned by model._")
        except Exception as e:
            st.error(f"AI generation failed: {e}")

# -----------------------------
# -----------------------------
# Export
# -----------------------------
st.markdown("### 9) Export results")
export_cols = [
    "capex_eur",
    "annual_opex_saving_eur",
    "annual_co2_reduction_t",
    "co2_value_eur_per_year",
    "implementation_months",
    "strategic_score_1_5",
    "required_info",
    "provided_info",
]
export_cols = [c for c in export_cols if c in df_opt.columns]
export = df_opt[export_cols].copy()

st.download_button(
    "Download portfolio results (CSV)",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="portfolio_results.csv",
    mime="text/csv",
)