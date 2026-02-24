# app.py
# Industrial Decarbonization Decision Tool (TFG prototype) — with:
# 1) Two modes: (A) Auto-propose initiatives from company inputs, (B) Upload client CSV
# 2) PESTEL mini-analysis (rule-based)
# 3) Robust CSV validation, metrics, confidence penalty
# 4) Portfolio optimization with PuLP
# 5) Plotly charts hardened against NaN/inf/negative sizes
# 6) Normative-aligned initiative proposals (Spain/MITECO Scope 1+2 framing)
# 7) Integrated AI Copilot (OpenAI / Gemini)
#
# Author: Carlos Falcó

from __future__ import annotations

import io
import json
import math
import os
try:
    import tomllib
except Exception:
    tomllib = None
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
    page_title="Herramienta de Decarbonización Industrial",
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

# Normative columns aligned with Scope 1+2 & MRV
NORMATIVE_COLUMNS = [
    "scope",                 # "Scope 1" / "Scope 2"
    "emission_source",        # e.g., "Fixed combustion", "Fleet fuels", "F-gases", "Purchased electricity"
    "activity_unit",          # kWh, L, kg, etc.
    "mrv_method",             # how to measure/verify (invoices, meters, logs)
    "normative_reference",    # e.g., "MITECO HC Alcance 1+2"
]
OPTIONAL_COLUMNS = OPTIONAL_COLUMNS + NORMATIVE_COLUMNS

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
        raise ValueError("No se subió ningún archivo")
    raw = uploaded_file.getvalue()
    if raw is None or len(raw) == 0:
        raise ValueError("El archivo subido está vacío")

    for sep in [",", ";"]:
        for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc)
                if df.shape[1] >= 2:
                    return df
            except pd.errors.EmptyDataError:
                raise ValueError("El CSV no tiene columnas (vacío o malformado).")
            except Exception:
                continue

    raise ValueError("No se pudo parsear el CSV. Revisa separadores (coma/punto y coma) y cabeceras.")


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


def column_config_es(cols: List[str]) -> dict:
    labels = {
        "id": "ID",
        "initiative_family": "Familia",
        "initiative": "Iniciativa",
        "capex_eur": "CAPEX (€)",
        "annual_opex_saving_eur": "Ahorro OPEX anual (€)",
        "annual_co2_reduction_t": "Reducción CO₂ anual (t)",
        "implementation_months": "Implementación (meses)",
        "strategic_score_1_5": "Puntuación estratégica (1-5)",
        "notes": "Notas",
        "confidence_0_1": "Confianza (0-1)",
        "required_info": "Info requerida",
        "provided_info": "Info proporcionada",
        "initiative_family": "Familia",
        "data_dependency": "Dependencia de datos",
        "scope": "Alcance",
        "emission_source": "Fuente de emisión",
        "activity_unit": "Unidad de actividad",
        "mrv_method": "Método MRV",
        "normative_reference": "Referencia normativa",
        "co2_value_eur_per_year": "Valor CO₂ €/año",
        "total_annual_benefit_eur": "Beneficio anual total (€)",
        "confidence": "Confianza",
        "npv_eur": "NPV (€)",
        "npv_penalized_eur": "NPV penalizado (€)",
        "payback_years": "Payback (años)",
        "selected": "Seleccionada",
    }
    return {c: st.column_config.Column(labels.get(c, c)) for c in cols}


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Faltan columnas obligatorias: {missing}")

    if "id" in df.columns and df["id"].isna().any():
        errors.append("La columna 'id' tiene valores faltantes.")
    if "initiative" in df.columns and df["initiative"].astype(str).str.strip().eq("").any():
        errors.append("La columna 'initiative' tiene valores en blanco.")

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

    energy_intensity = (company.get("energy_intensity") or "Media").lower()
    fossil_heat = (company.get("fossil_heat_use") or "Algo").lower()
    grid_emissions = (company.get("grid_emissions_level") or "Medio").lower()

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

    if energy_intensity in ["high", "alto", "very high", "alta"]:
        e.append("Alta exposición a volatilidad de precios de energía: medidas de eficiencia suelen tener ROI rápido.")
    else:
        e.append("Medidas ‘no-regret’ (eficiencia, control, mantenimiento) suelen ser la base económica del plan.")

    if fossil_heat in ["high", "alto"]:
        e.append("Riesgo de coste de carbono y combustibles: atractivas soluciones de electrificación y recuperación de calor.")
    if grid_emissions in ["low", "bajo"]:
        e.append("Red eléctrica relativamente limpia mejora el caso de electrificación (más CO₂ evitado por kWh).")
    elif grid_emissions in ["high", "alto"]:
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

    return {"Político": p, "Económico": e, "Social": s, "Tecnológico": t, "Ambiental": en, "Legal": l}


def generate_ai_pestel(
    provider: str,
    api_key: str,
    model: str,
    company_inputs: Dict,
) -> Dict[str, List[str]]:
    if not api_key:
        raise RuntimeError(f"Falta la API key de {provider}.")

    system_prompt = (
        "Eres un consultor senior de descarbonización industrial. "
        "Devuelve SOLO JSON válido, sin texto adicional. "
        "Responde en español. "
        "Estructura: claves Político, Económico, Social, Tecnológico, Ambiental, Legal "
        "y cada valor es una lista de bullets cortos."
    )

    user_prompt = (
        "Genera un PESTEL breve y accionable (2-3 bullets por categoría) "
        "basado únicamente en los inputs de la empresa.\n\n"
        f"COMPANY_INPUTS: {json.dumps(company_inputs, ensure_ascii=False)}"
    )

    provider_norm = (provider or "").strip().lower()
    if provider_norm == "openai":
        if OpenAI is None:
            raise RuntimeError("Paquete OpenAI no instalado. Añade 'openai' a requirements.txt.")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
    elif provider_norm == "gemini":
        model_name = (model or "").strip()
        if model_name.startswith("models/"):
            model_name = model_name.split("/", 1)[1]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
        }
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Error API Gemini ({r.status_code}): {r.text[:400]}")
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini no devolvió candidatos.")
        parts = candidates[0].get("content", {}).get("parts", [])
        content = "".join([str(p.get("text", "")) for p in parts]).strip()
    else:
        raise RuntimeError(f"Proveedor no soportado: {provider}")

    if not content:
        raise RuntimeError("La IA devolvió contenido vacío.")

    def _extract_json(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
        for start, end in [("{", "}"), ("[", "]")]:
            s = t.find(start)
            e = t.rfind(end)
            if s != -1 and e != -1 and e > s:
                return t[s : e + 1]
        return t

    try:
        data = json.loads(_extract_json(content))
    except Exception as e:
        raise RuntimeError(f"Salida IA no es JSON válido: {e}")

    required_keys = ["Político", "Económico", "Social", "Tecnológico", "Ambiental", "Legal"]
    if not isinstance(data, dict) or not all(k in data for k in required_keys):
        raise RuntimeError("La salida de la IA debe ser un diccionario con claves PESTEL.")

    out: Dict[str, List[str]] = {}
    for k in required_keys:
        vals = data.get(k, [])
        if not isinstance(vals, list):
            vals = [str(vals)]
        out[k] = [str(x) for x in vals if str(x).strip()]
    return out


# -----------------------------
# Initiative proposal engine (normative-aligned)
# -----------------------------
def propose_initiatives(company: Dict, n: int = 8) -> pd.DataFrame:
    """
    Normative-aligned initiative generator (Spain / Scope 1+2 framing + MRV):
    - Scope 1: fixed combustion, fleet fuels, F-gas leaks
    - Scope 2: purchased electricity (and offsets/contracting)
    Each initiative includes scope/source + MRV hints and required_info.
    """

    annual_electricity_mwh = company.get("annual_electricity_mwh")
    annual_fuel_mwh = company.get("annual_fuel_mwh")
    electricity_price = company.get("electricity_price_eur_mwh")
    fuel_price = company.get("fuel_price_eur_mwh")
    co2_factor_elec = company.get("co2_factor_elec_t_per_mwh")
    co2_factor_fuel = company.get("co2_factor_fuel_t_per_mwh")

    has_compressed_air = bool(company.get("has_compressed_air", True))
    has_process_heat = bool(company.get("has_process_heat", True))
    roof_area_m2 = company.get("roof_area_m2")
    waste_heat_potential = (company.get("waste_heat_potential") or "Unknown")

    # New toggles
    has_fleet = bool(company.get("has_fleet", False))
    has_refrigerants = bool(company.get("has_refrigerants", True))
    electricity_has_gdo = company.get("electricity_has_gdo")  # None/False/True
    electricity_gdo_type = company.get("electricity_gdo_type")  # "renewable"/"cogen"/None
    cnmc_supplier_known = company.get("cnmc_supplier_known")  # True/False/None

    def ok_num(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x))

    initiatives: List[dict] = []
    next_id = 1

    def add_init(**kwargs):
        nonlocal next_id, initiatives
        base = {
            "id": next_id,
            "initiative_family": kwargs.get("initiative_family", ""),
            "initiative": kwargs.get("initiative", ""),
            "capex_eur": kwargs.get("capex_eur", np.nan),
            "annual_opex_saving_eur": kwargs.get("annual_opex_saving_eur", np.nan),
            "annual_co2_reduction_t": kwargs.get("annual_co2_reduction_t", np.nan),
            "implementation_months": kwargs.get("implementation_months", np.nan),
            "strategic_score_1_5": kwargs.get("strategic_score_1_5", 3),
            "notes": kwargs.get("notes", ""),
            "required_info": kwargs.get("required_info", ""),
            "provided_info": kwargs.get("provided_info", ""),
            "data_dependency": kwargs.get("data_dependency", "Medium"),
            "confidence_0_1": np.nan,

            "scope": kwargs.get("scope", ""),
            "emission_source": kwargs.get("emission_source", ""),
            "activity_unit": kwargs.get("activity_unit", ""),
            "mrv_method": kwargs.get("mrv_method", ""),
            "normative_reference": kwargs.get("normative_reference", "Metodología Alcance 1+2 (España/MITECO MRV)"),
        }
        initiatives.append(base)
        next_id += 1

    # -----------------------------
    # Scope 2 — Electricity contracting / GdO / factor review
    # -----------------------------
    prv = []
    if ok_num(annual_electricity_mwh):
        prv.append("annual_electricity_mwh")
    if ok_num(electricity_price):
        prv.append("electricity_price_eur_mwh")
    if ok_num(co2_factor_elec):
        prv.append("co2_factor_elec_t_per_mwh")
    if cnmc_supplier_known is True:
        prv.append("cnmc_supplier_known")
    if electricity_has_gdo is True:
        prv.append("electricity_has_gdo")
    if electricity_gdo_type:
        prv.append("electricity_gdo_type")

    # If GdO renewable, Scope2 can drop strongly; we keep "rough" placeholder reduction (50% of current) unless user fills.
    add_init(
        initiative_family="Electricity (Scope 2)",
        initiative="Electricity supply decarbonization (supplier factor review + GdO / PPA / contract change)",
        scope="Scope 2",
        emission_source="Purchased electricity",
        activity_unit="kWh (invoices)",
        mrv_method="Invoices + supplier label/factor + GdO certificates + kWh covered by GdO",
        capex_eur=5000,
        annual_opex_saving_eur=np.nan,
        annual_co2_reduction_t=(annual_electricity_mwh * (co2_factor_elec or 0.0) * 0.50) if (ok_num(annual_electricity_mwh) and ok_num(co2_factor_elec)) else np.nan,
        implementation_months=2,
        strategic_score_1_5=5,
        notes=(
            "Alinear cálculo con evidencia documental: etiqueta/factor del proveedor, kWh cubiertos por GdO, "
            "y coherencia con facturas. Ajustar CO₂ con datos reales del proveedor/GdO."
        ),
        required_info="annual_electricity_mwh;electricity_invoices;provider_label_or_factor;gdo_status;kwh_with_gdo",
        provided_info=";".join(prv),
        data_dependency="High",
    )

    # EMS + MRV backbone
    add_init(
        initiative_family="Digital / EMS",
        initiative="Energy Management System (EMS) + submetering + analytics (MRV backbone)",
        scope="Scope 2",
        emission_source="Purchased electricity (cross-cutting)",
        activity_unit="kWh (meters + invoices)",
        mrv_method="Submetering + invoices; baseline vs post-implementation verification",
        capex_eur=100000,
        annual_opex_saving_eur=(0.03 * annual_electricity_mwh * electricity_price) if (ok_num(annual_electricity_mwh) and ok_num(electricity_price)) else np.nan,
        annual_co2_reduction_t=(0.03 * annual_electricity_mwh * (co2_factor_elec or 0.0)) if (ok_num(annual_electricity_mwh) and ok_num(co2_factor_elec)) else np.nan,
        implementation_months=4,
        strategic_score_1_5=5,
        notes="Base de medición para priorizar y verificar ahorros; reduce ‘data gaps’.",
        required_info="annual_electricity_mwh;electricity_invoices;main_meters;submetering_plan",
        provided_info=";".join(prv),
        data_dependency="Medium",
    )

    # Motors + VFD
    add_init(
        initiative_family="Electric efficiency",
        initiative="High-efficiency motors + VFD (targeted replacements)",
        scope="Scope 2",
        emission_source="Purchased electricity (motors/drives)",
        activity_unit="kWh (metered or estimated)",
        mrv_method="Motor inventory + operating hours + spot measurements / submetering",
        capex_eur=120000,
        annual_opex_saving_eur=(0.02 * annual_electricity_mwh * electricity_price) if (ok_num(annual_electricity_mwh) and ok_num(electricity_price)) else np.nan,
        annual_co2_reduction_t=(0.02 * annual_electricity_mwh * (co2_factor_elec or 0.0)) if (ok_num(annual_electricity_mwh) and ok_num(co2_factor_elec)) else np.nan,
        implementation_months=3,
        strategic_score_1_5=3,
        notes="‘Quick win’ si hay cargas variables (bombas, ventiladores, compresores).",
        required_info="motor_inventory;operating_hours;electricity_price_eur_mwh;provider_label_or_factor",
        provided_info="",
        data_dependency="Medium",
    )

    # Compressed air
    if has_compressed_air:
        add_init(
            initiative_family="Utilities",
            initiative="Compressed air leak program + pressure optimization + controls",
            scope="Scope 2",
            emission_source="Purchased electricity (compressed air)",
            activity_unit="kWh (metered) / leak rate",
            mrv_method="Audit + leak tagging + compressor kWh metering before/after",
            capex_eur=60000,
            annual_opex_saving_eur=(0.01 * annual_electricity_mwh * electricity_price) if (ok_num(annual_electricity_mwh) and ok_num(electricity_price)) else np.nan,
            annual_co2_reduction_t=(0.01 * annual_electricity_mwh * (co2_factor_elec or 0.0)) if (ok_num(annual_electricity_mwh) and ok_num(co2_factor_elec)) else np.nan,
            implementation_months=2,
            strategic_score_1_5=2,
            notes="Requiere auditoría/medición; se confirma con campaña de fugas y registro de kWh.",
            required_info="compressed_air_kwh;leak_rate;electricity_price_eur_mwh",
            provided_info="",
            data_dependency="High",
        )

    # HVAC / controls
    add_init(
        initiative_family="Buildings",
        initiative="HVAC optimization + controls (BMS tuning / schedules)",
        scope="Scope 2",
        emission_source="Purchased electricity (HVAC)",
        activity_unit="kWh (metered/estimated)",
        mrv_method="BMS logs + invoices/submetering; compare schedules and energy use",
        capex_eur=150000,
        annual_opex_saving_eur=(0.01 * annual_electricity_mwh * electricity_price) if (ok_num(annual_electricity_mwh) and ok_num(electricity_price)) else np.nan,
        annual_co2_reduction_t=(0.01 * annual_electricity_mwh * (co2_factor_elec or 0.0)) if (ok_num(annual_electricity_mwh) and ok_num(co2_factor_elec)) else np.nan,
        implementation_months=3,
        strategic_score_1_5=3,
        notes="Más relevante si hay edificios climatizados o requisitos de humedad/calidad.",
        required_info="building_area_m2;hvac_profile;electricity_price_eur_mwh",
        provided_info="",
        data_dependency="Medium",
    )

    # Solar PV
    prv = []
    if ok_num(roof_area_m2):
        prv.append("roof_area_m2")
    if ok_num(annual_electricity_mwh):
        prv.append("annual_electricity_mwh")
    if ok_num(electricity_price):
        prv.append("electricity_price_eur_mwh")
    if ok_num(co2_factor_elec):
        prv.append("co2_factor_elec_t_per_mwh")

    if ok_num(roof_area_m2) and ok_num(electricity_price):
        annual_kwh = roof_area_m2 * 0.18 * 1400.0
        annual_mwh = annual_kwh / 1000.0
        used_mwh = 0.70 * annual_mwh
        saving_eur = used_mwh * electricity_price
        co2_red = used_mwh * (co2_factor_elec or 0.0) if ok_num(co2_factor_elec) else np.nan
    else:
        saving_eur = np.nan
        co2_red = np.nan

    add_init(
        initiative_family="Renewables",
        initiative="Solar PV self-consumption (feasibility-based sizing)",
        scope="Scope 2",
        emission_source="Purchased electricity (offset via self-generation)",
        activity_unit="kWh generados + kWh consumidos",
        mrv_method="Inverter production logs + electricity invoices; quantify kWh offset and apply supplier factor",
        capex_eur=225000,
        annual_opex_saving_eur=saving_eur,
        annual_co2_reduction_t=co2_red,
        implementation_months=5,
        strategic_score_1_5=4,
        notes="Si faltan datos (superficie, perfil de carga), no asumir: mantener low-confidence.",
        required_info="roof_area_m2;location;orientation;annual_electricity_mwh;electricity_price_eur_mwh;provider_label_or_factor",
        provided_info=";".join(prv),
        data_dependency="High",
    )

    # -----------------------------
    # Scope 1 — Process heat / fuels
    # -----------------------------
    if has_process_heat:
        prv = []
        if ok_num(annual_fuel_mwh):
            prv.append("annual_fuel_mwh")
        if ok_num(fuel_price):
            prv.append("fuel_price_eur_mwh")
        if ok_num(co2_factor_fuel):
            prv.append("co2_factor_fuel_t_per_mwh")

        add_init(
            initiative_family="Heat (Scope 1)",
            initiative="Boiler/burner upgrade + combustion tuning (efficiency increase)",
            scope="Scope 1",
            emission_source="Fixed combustion (process heat fuels)",
            activity_unit="kWh fuel / m3 / kg (invoices)",
            mrv_method="Fuel invoices + boiler efficiency verification; baseline vs after",
            capex_eur=180000,
            annual_opex_saving_eur=(0.05 * annual_fuel_mwh * fuel_price) if (ok_num(annual_fuel_mwh) and ok_num(fuel_price)) else np.nan,
            annual_co2_reduction_t=(0.05 * annual_fuel_mwh * (co2_factor_fuel or 0.0)) if (ok_num(annual_fuel_mwh) and ok_num(co2_factor_fuel)) else np.nan,
            implementation_months=6,
            strategic_score_1_5=4,
            notes="Medida típica de eficiencia térmica: requiere trazabilidad de consumo de combustible.",
            required_info="annual_fuel_mwh;fuel_type;fuel_invoices;fuel_emission_factor",
            provided_info=";".join(prv),
            data_dependency="Medium",
        )

        add_init(
            initiative_family="Heat",
            initiative="Waste heat recovery / industrial heat pump (feasibility-based)",
            scope="Scope 1",
            emission_source="Fixed combustion reduction (plus possible electricity shift)",
            activity_unit="kWh fuel avoided + kWh electricity added",
            mrv_method="Engineering study + metering; quantify fuel reduction and electricity increase",
            capex_eur=250000,
            annual_opex_saving_eur=(0.05 * annual_fuel_mwh * fuel_price) if (ok_num(annual_fuel_mwh) and ok_num(fuel_price) and str(waste_heat_potential).lower() in ["medium", "high"]) else np.nan,
            annual_co2_reduction_t=(0.05 * annual_fuel_mwh * (co2_factor_fuel or 0.0)) if (ok_num(annual_fuel_mwh) and ok_num(co2_factor_fuel) and str(waste_heat_potential).lower() in ["medium", "high"]) else np.nan,
            implementation_months=8,
            strategic_score_1_5=4,
            notes="Muy sensible a perfil térmico; requiere estudio y MRV para confirmar.",
            required_info="waste_heat_sources;heat_demand_profile;annual_fuel_mwh;fuel_type;electricity_price_eur_mwh",
            provided_info="waste_heat_potential" if waste_heat_potential else "",
            data_dependency="High",
        )

        add_init(
            initiative_family="Heat",
            initiative="Electrification (partial) of low/medium-temperature heat demand",
            scope="Scope 1",
            emission_source="Fuel displacement (Scope 1) + increased electricity (Scope 2)",
            activity_unit="kWh fuel avoided + kWh electricity added",
            mrv_method="Engineering study + metering; compare fuel invoices vs electricity invoices",
            capex_eur=420000,
            annual_opex_saving_eur=np.nan,
            annual_co2_reduction_t=np.nan,
            implementation_months=7,
            strategic_score_1_5=4,
            notes="Depende de temperatura, capacidad eléctrica y comparativa de costes. Estimar con datos reales.",
            required_info="heat_demand_profile;temperature_levels;grid_capacity;electricity_price_eur_mwh;fuel_price_eur_mwh",
            provided_info="",
            data_dependency="High",
        )

    # -----------------------------
    # Scope 1 — Fleet fuels
    # -----------------------------
    if has_fleet:
        add_init(
            initiative_family="Fleet (Scope 1)",
            initiative="Fleet efficiency program (eco-driving + maintenance + route optimization)",
            scope="Scope 1",
            emission_source="Fleet fuels",
            activity_unit="liters fuel (invoices/cards) or km",
            mrv_method="Fuel cards/invoices + mileage logs; baseline vs after",
            capex_eur=25000,
            annual_opex_saving_eur=np.nan,
            annual_co2_reduction_t=np.nan,
            implementation_months=2,
            strategic_score_1_5=3,
            notes="Medida de bajo CAPEX; necesita datos de litros por tipo de combustible.",
            required_info="fleet_fuel_liters_by_type;vehicle_km;fuel_emission_factors",
            provided_info="",
            data_dependency="High",
        )

    # -----------------------------
    # Scope 1 — F-gases (refrigerants)
    # -----------------------------
    if has_refrigerants:
        add_init(
            initiative_family="Refrigeration (Scope 1)",
            initiative="Refrigerant leak reduction + switch to lower-GWP refrigerants (where feasible)",
            scope="Scope 1",
            emission_source="F-gas leaks",
            activity_unit="kg refrigerant top-ups",
            mrv_method="Maintenance logs + refrigerant charge records; quantify kg recharged and GWP",
            capex_eur=80000,
            annual_opex_saving_eur=np.nan,
            annual_co2_reduction_t=np.nan,
            implementation_months=6,
            strategic_score_1_5=4,
            notes="Requiere inventario de equipos y registros de recargas (kg) para cuantificar emisiones.",
            required_info="refrigerant_type;kg_recharged_per_year;equipment_inventory;maintenance_logs",
            provided_info="",
            data_dependency="High",
        )

    df = pd.DataFrame(initiatives).head(n).copy()

    # Ensure required/optional columns exist
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
        npv_vals = df["npv_eur"].values.astype(float)
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
                "initiative_family": "Renovables",
                "initiative": "Autoconsumo fotovoltaico (dimensionamiento por viabilidad)",
                "capex_eur": 400000,
                "annual_opex_saving_eur": 95000,
                "annual_co2_reduction_t": 250,
                "implementation_months": 5,
                "strategic_score_1_5": 3,
                "notes": "Depende de disponibilidad de cubierta/terreno y del perfil de carga.",
                "required_info": "roof_area_m2;location;orientation;annual_electricity_mwh;electricity_price_eur_mwh;provider_label_or_factor",
                "provided_info": "roof_area_m2;annual_electricity_mwh;electricity_price_eur_mwh",
                "confidence_0_1": 0.55,
                "data_dependency": "High",
                "scope": "Alcance 2",
                "emission_source": "Electricidad comprada (compensada por autogeneración)",
                "activity_unit": "kWh generados + kWh consumidos",
                "mrv_method": "Registros de inversor + facturas",
                "normative_reference": "Metodología Alcance 1+2 (España/MITECO MRV)",
            },
            {
                "id": 2,
                "initiative_family": "Eficiencia eléctrica",
                "initiative": "Motores de alta eficiencia + VFD (reemplazos focalizados)",
                "capex_eur": 120000,
                "annual_opex_saving_eur": 45000,
                "annual_co2_reduction_t": 90,
                "implementation_months": 3,
                "strategic_score_1_5": 2,
                "notes": "Más fácil de cuantificar con inventario de motores y horas de operación.",
                "required_info": "motor_inventory;operating_hours;electricity_price_eur_mwh;provider_label_or_factor",
                "provided_info": "",
                "confidence_0_1": 0.70,
                "data_dependency": "Medium",
                "scope": "Alcance 2",
                "emission_source": "Electricidad comprada (motores/variadores)",
                "activity_unit": "kWh",
                "mrv_method": "Inventario + horas + submedición",
                "normative_reference": "Metodología Alcance 1+2 (España/MITECO MRV)",
            },
        ]
    )
    return example.to_csv(index=False).encode("utf-8")


# -----------------------------
# AI helpers
# -----------------------------
def df_for_ai(df: pd.DataFrame, limit: int = 25) -> List[dict]:
    if df is None or df.empty:
        return []
    cols = [
        "id",
        "initiative_family",
        "initiative",
        "scope",
        "emission_source",
        "selected",
        "capex_eur",
        "annual_opex_saving_eur",
        "annual_co2_reduction_t",
        "npv_penalized_eur",
        "payback_years",
        "data_dependency",
        "required_info",
        "provided_info",
        "notes",
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
    company_inputs: Dict,
    assumptions: Dict,
    constraints: Dict,
    summary: dict,
    df_all: pd.DataFrame,
    df_selected: pd.DataFrame,
    pestel: Dict[str, List[str]] | None,
    extra_prompt: str,
) -> str:
    if not api_key:
        raise RuntimeError(f"Falta la API key de {provider}.")

    context = {
        "mode": mode,
        "company_inputs": company_inputs,
        "assumptions": assumptions,
        "constraints": constraints,
        "optimization_summary": summary,
        "selected_count": int(df_selected.shape[0]) if df_selected is not None else 0,
        "all_initiatives_preview": df_for_ai(df_all, limit=25),
        "selected_initiatives_preview": df_for_ai(df_selected, limit=25),
        "pestel": pestel or {},
        "normative_note": "Trata las iniciativas como plan de mejora de Alcance 1+2; destaca data gaps y necesidades MRV.",
    }

    system_prompt = (
        "Eres un consultor senior de descarbonización industrial (operaciones + finanzas). "
        "Responde SOLO usando los datos proporcionados. "
        "No inventes cifras. Si faltan datos, dilo explícitamente y pide los datos mínimos. "
        "Incluye enfoque MRV (cómo medir/verificar) y distingue Alcance 1 vs Alcance 2 cuando aplique."
    )

    user_prompt = (
        "Redacta un executive brief en español (máximo ~600-900 palabras) con:\n"
        "1) Diagnóstico del portfolio y lógica de selección.\n"
        "2) Top 3 acciones para los próximos 90 días (con responsables sugeridos y datos a pedir).\n"
        "3) Riesgos principales y ‘data gaps’ críticos.\n"
        "4) 3 sensibilidades/escenarios para testear (precio energía, factor electricidad, CAPEX/ROI, etc.).\n"
        "5) Checklist MRV: qué evidencias/documentos usar para justificar ahorro y CO₂.\n\n"
        f"Instrucción adicional del usuario: {extra_prompt or 'Ninguna'}\n\n"
        f"DATA (JSON):\n{json.dumps(context, ensure_ascii=False)}"
    )

    provider_norm = (provider or "").strip().lower()
    if provider_norm == "openai":
        if OpenAI is None:
            raise RuntimeError("Paquete OpenAI no instalado. Añade 'openai' a requirements.txt.")
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
        model_name = (model or "").strip()
        if model_name.startswith("models/"):
            model_name = model_name.split("/", 1)[1]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
        }
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Error API Gemini ({r.status_code}): {r.text[:400]}")
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini no devolvió candidatos.")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join([str(p.get("text", "")) for p in parts]).strip()
        if not text:
            raise RuntimeError("Gemini devolvió texto vacío.")
        return text

    raise RuntimeError(f"Proveedor no soportado: {provider}")


def generate_ai_initiatives(
    provider: str,
    api_key: str,
    model: str,
    company_inputs: Dict,
    n: int = 8,
) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError(f"Falta la API key de {provider}.")

    schema_note = {
        "required_columns": REQUIRED_COLUMNS,
        "optional_columns": OPTIONAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "n_rows": n,
    }

    system_prompt = (
        "Eres un consultor senior de descarbonización industrial (operaciones + finanzas). "
        "Devuelve SOLO JSON válido, sin texto adicional. "
        "No inventes cifras si no hay datos: usa null. "
        "Responde en español. "
        "Incluye campos normativos (scope, emission_source, activity_unit, mrv_method, normative_reference) "
        "y data gaps (required_info, provided_info, data_dependency)."
    )

    user_prompt = (
        "Genera exactamente N iniciativas en JSON (lista de objetos) siguiendo este esquema. "
        "Usa los inputs de la empresa y supuestos conservadores. "
        "Cada iniciativa debe incluir TODAS las columnas requeridas y opcionales. "
        "Para 'scope' usa 'Alcance 1' o 'Alcance 2'.\n\n"
        f"N = {n}\n"
        f"SCHEMA: {json.dumps(schema_note, ensure_ascii=False)}\n"
        f"COMPANY_INPUTS: {json.dumps(company_inputs, ensure_ascii=False)}\n"
    )

    provider_norm = (provider or "").strip().lower()
    if provider_norm == "openai":
        if OpenAI is None:
            raise RuntimeError("Paquete OpenAI no instalado. Añade 'openai' a requirements.txt.")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
    elif provider_norm == "gemini":
        model_name = (model or "").strip()
        if model_name.startswith("models/"):
            model_name = model_name.split("/", 1)[1]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
        }
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Error API Gemini ({r.status_code}): {r.text[:400]}")
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini no devolvió candidatos.")
        parts = candidates[0].get("content", {}).get("parts", [])
        content = "".join([str(p.get("text", "")) for p in parts]).strip()
    else:
        raise RuntimeError(f"Proveedor no soportado: {provider}")

    if not content:
        raise RuntimeError("La IA devolvió contenido vacío.")

    def _extract_json(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
        # Try to find the first JSON list or object
        for start, end in [("[", "]"), ("{", "}")]:
            s = t.find(start)
            e = t.rfind(end)
            if s != -1 and e != -1 and e > s:
                return t[s : e + 1]
        return t

    try:
        data = json.loads(_extract_json(content))
    except Exception as e:
        raise RuntimeError(f"Salida IA no es JSON válido: {e}")

    if not isinstance(data, list):
        raise RuntimeError("La salida de la IA debe ser una lista JSON de iniciativas.")

    df = pd.DataFrame(data)
    if df.empty:
        raise RuntimeError("La IA devolvió una lista de iniciativas vacía.")

    return df

# -----------------------------
# UI
# -----------------------------
st.title("Herramienta de Decarbonización Industrial")
st.caption("Prototipo TFG – Carlos Falcó")

with st.sidebar:
    st.header("Modo")
    mode = "A"

    st.divider()
    st.header("Supuestos financieros")
    horizon_years = st.slider("Horizonte del proyecto (años)", 1, 10, 5, 1)
    discount_rate_pct = st.slider("Tasa de descuento (%)", 0.0, 25.0, 8.0, 0.25)
    discount_rate = discount_rate_pct / 100.0
    co2_price = st.number_input("Precio CO₂ (€/t)", min_value=0.0, value=80.0, step=5.0)

    st.divider()
    st.header("Restricciones de optimización")
    budget_eur = st.number_input("Presupuesto CAPEX (€)", min_value=0.0, value=10000.0, step=10000.0)
    min_co2_t = st.number_input("Objetivo mínimo anual de CO₂ (t/año) [opcional]", min_value=0.0, value=0.0, step=10.0)

    confidence_floor = 1.0

    st.divider()
    st.header("Objetivo")
    st.write("Optimización ponderada: máxima reducción de CO₂ y máximo NPV total con el presupuesto.")
    w_co2 = st.slider("Peso CO₂", 0.0, 1.0, 0.70, 0.05)
    w_npv = st.slider("Peso NPV", 0.0, 1.0, 0.30, 0.05)
    s = max(1e-9, w_npv + w_co2)
    w_npv, w_co2 = w_npv / s, w_co2 / s
    w_strategy = 0.0
    objective = "Balanced score (NPV + CO2 + strategy)"

    st.divider()
    st.header("Plantilla cliente")
    st.download_button(
        "Descargar plantilla CSV (ejemplo)",
        data=template_csv_bytes(),
        file_name="client_initiatives_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    st.header("Copiloto IA")
    ai_provider = "Gemini"
    ai_api_key = ""
    diag = {
        "cwd": os.getcwd(),
        "secrets_exists_rel": os.path.exists(".streamlit/secrets.toml"),
        "from_st_secrets": "NO",
        "from_env": "NO",
        "from_file": "NO",
        "file_parse_error": "",
    }

    try:
        ai_api_key = st.secrets.get("GEMINI_API_KEY", "")
        diag["from_st_secrets"] = "SÍ" if ai_api_key else "NO"
    except Exception:
        ai_api_key = ""

    if not ai_api_key:
        env_key = os.getenv("GEMINI_API_KEY", "") or ""
        if env_key:
            ai_api_key = env_key
            diag["from_env"] = "SÍ"

    if not ai_api_key and tomllib is not None:
        try:
            path = ".streamlit/secrets.toml"
            raw = ""
            if os.path.exists(path):
                raw = open(path, "rb").read().decode("utf-8-sig", errors="ignore")
            if raw:
                data = tomllib.loads(raw)
                ai_api_key = data.get("GEMINI_API_KEY", "") or ""
                if ai_api_key:
                    diag["from_file"] = "SÍ"
        except Exception as e:
            diag["file_parse_error"] = str(e)[:120]

    key_status = "SÍ" if ai_api_key else "NO"
    st.caption(f"Clave Gemini cargada: {key_status}")
    ai_model = "gemini-2.5-flash"


# -----------------------------
# Data source selection
# -----------------------------
df_base = None
company_inputs: Dict = {}
pestel: Dict[str, List[str]] = {}

if mode.startswith("A"):
    st.markdown("### 1) Inputs para huella de carbono y plan de mejora (MITECO)")
    c1, c2, c3 = st.columns(3)

    with c1:
        company_inputs["company_name"] = st.text_input("Nombre de la organización (opcional)", value="")
        company_inputs["cnae_sector"] = st.text_input("CNAE / Sector", value="")
        company_inputs["country_region"] = st.text_input("País / Comunidad / Provincia", value="España")
        company_inputs["inventory_year"] = st.number_input("Año de inventario (cálculo)", min_value=2000, max_value=2100, value=2024, step=1)
        company_inputs["sector"] = company_inputs["cnae_sector"]

    with c2:
        st.markdown("**Alcance 1 — Combustión fija**")
        company_inputs["fuel_ng_mwh"] = st.number_input("Gas natural (MWh/año)", min_value=0.0, value=0.0, step=100.0)
        company_inputs["fuel_diesel_mwh"] = st.number_input("Gasóleo (MWh/año)", min_value=0.0, value=0.0, step=100.0)
        company_inputs["fuel_fuel_oil_mwh"] = st.number_input("Fuelóleo (MWh/año)", min_value=0.0, value=0.0, step=100.0)
        company_inputs["fuel_lpg_mwh"] = st.number_input("GLP (MWh/año)", min_value=0.0, value=0.0, step=50.0)
        company_inputs["fuel_biomass_mwh"] = st.number_input("Biomasa (MWh/año)", min_value=0.0, value=0.0, step=50.0)

        st.markdown("**Alcance 1 — Combustión móvil (flota)**")
        company_inputs["fleet_diesel_l"] = st.number_input("Diésel flota (litros/año)", min_value=0.0, value=0.0, step=1000.0)
        company_inputs["fleet_gasoline_l"] = st.number_input("Gasolina flota (litros/año)", min_value=0.0, value=0.0, step=1000.0)
        company_inputs["fleet_km"] = st.number_input("Km recorridos (opcional)", min_value=0.0, value=0.0, step=1000.0)

        st.markdown("**Alcance 1 — Emisiones fugitivas**")
        company_inputs["refrigerant_type"] = st.text_input("Tipo de refrigerante principal (p. ej., R-410A)", value="")
        company_inputs["refrigerant_kg"] = st.number_input("Kg recargados/año", min_value=0.0, value=0.0, step=10.0)

    with c3:
        st.markdown("**Alcance 2 — Electricidad comprada**")
        company_inputs["annual_electricity_mwh"] = st.number_input("Electricidad comprada (MWh/año)", min_value=0.0, value=0.0, step=100.0)
        company_inputs["electricity_price_eur_mwh"] = st.number_input("Precio electricidad (€/MWh)", min_value=0.0, value=0.0, step=5.0)
        company_inputs["co2_factor_elec_t_per_mwh"] = st.number_input("Factor CO₂ electricidad (tCO₂/MWh) [opcional]", min_value=0.0, value=0.0, step=0.01)

        st.markdown("**Alcance 2 — Calor/vapor comprado**")
        company_inputs["annual_purchased_heat_mwh"] = st.number_input("Calor/vapor comprado (MWh/año)", min_value=0.0, value=0.0, step=50.0)

        st.markdown("**Precios y evidencias**")
        company_inputs["fuel_price_eur_mwh"] = st.number_input("Precio combustible (€/MWh)", min_value=0.0, value=0.0, step=5.0)
        company_inputs["co2_factor_fuel_t_per_mwh"] = st.number_input("Factor CO₂ combustibles (tCO₂/MWh) [opcional]", min_value=0.0, value=0.0, step=0.01)
        company_inputs["has_invoices"] = st.checkbox("Facturas/lecturas disponibles", value=True)
        company_inputs["has_meters"] = st.checkbox("Contadores/medición disponibles", value=False)
        company_inputs["has_submetering"] = st.checkbox("Submetering disponible", value=False)

        st.markdown("**Documentación Alcance 2**")
        company_inputs["cnmc_supplier_known"] = st.checkbox("Etiqueta/factor del proveedor disponible (factura/contrato)", value=False)
        company_inputs["electricity_has_gdo"] = st.checkbox("El proveedor canjea GdO para tu consumo", value=False)
        company_inputs["electricity_gdo_type"] = st.selectbox("Tipo de GdO (si aplica)", ["Ninguno/Desconocido", "Renovable", "Cogeneración"], index=0)

    st.markdown("### 2) Restricciones técnicas para plan de mejora (opcional)")
    c4, c5 = st.columns(2)
    with c4:
        company_inputs["roof_area_m2"] = st.number_input("Área disponible de cubierta (m²) [opcional]", min_value=0.0, value=0.0, step=100.0)
        company_inputs["has_compressed_air"] = st.checkbox("Usa sistemas de aire comprimido", value=True)
    with c5:
        company_inputs["waste_heat_potential"] = st.selectbox("Potencial de calor residual (aprox.)", ["Desconocido", "Bajo", "Medio", "Alto"], index=0)
        company_inputs["has_process_heat"] = st.checkbox("Tiene calor de proceso", value=True)
        company_inputs["heat_temp_level"] = st.selectbox("Nivel de temperatura de proceso", ["Baja", "Media", "Alta", "Desconocida"], index=3)
        company_inputs["load_profile_known"] = st.checkbox("Perfil de carga conocido", value=False)

    # Normalizar tipo de GdO
    if company_inputs.get("electricity_gdo_type") == "Ninguno/Desconocido":
        company_inputs["electricity_gdo_type"] = None
    elif company_inputs.get("electricity_gdo_type") == "Renovable":
        company_inputs["electricity_gdo_type"] = "renewable"
    elif company_inputs.get("electricity_gdo_type") == "Cogeneración":
        company_inputs["electricity_gdo_type"] = "cogen"

    # Derivar agregados para el motor de iniciativas
    stationary_fuel_mwh = sum([
        company_inputs.get("fuel_ng_mwh", 0.0),
        company_inputs.get("fuel_diesel_mwh", 0.0),
        company_inputs.get("fuel_fuel_oil_mwh", 0.0),
        company_inputs.get("fuel_lpg_mwh", 0.0),
        company_inputs.get("fuel_biomass_mwh", 0.0),
    ])
    company_inputs["annual_fuel_mwh"] = stationary_fuel_mwh if stationary_fuel_mwh > 0 else 0.0

    # Señales para PESTEL (derivadas)
    total_energy_mwh = (company_inputs.get("annual_electricity_mwh", 0.0) or 0.0) + (company_inputs.get("annual_fuel_mwh", 0.0) or 0.0)
    if total_energy_mwh >= 20000:
        company_inputs["energy_intensity"] = "Alta"
    elif total_energy_mwh >= 5000:
        company_inputs["energy_intensity"] = "Media"
    else:
        company_inputs["energy_intensity"] = "Baja"

    elec_factor = company_inputs.get("co2_factor_elec_t_per_mwh") or 0.0
    if elec_factor >= 0.3:
        company_inputs["grid_emissions_level"] = "Alto"
    elif 0 < elec_factor <= 0.1:
        company_inputs["grid_emissions_level"] = "Bajo"
    else:
        company_inputs["grid_emissions_level"] = "Medio"

    company_inputs["fossil_heat_use"] = "Alto" if stationary_fuel_mwh > 0 else "Ninguno"

    country_text = (company_inputs.get("country_region") or "").lower()
    company_inputs["eu_context"] = True if any(x in country_text for x in ["espa", "spain", "ue", "eu"]) else False

    company_inputs["has_fleet"] = (company_inputs.get("fleet_diesel_l", 0.0) or 0.0) > 0 or (company_inputs.get("fleet_gasoline_l", 0.0) or 0.0) > 0
    company_inputs["has_refrigerants"] = (company_inputs.get("refrigerant_kg", 0.0) or 0.0) > 0 or bool(company_inputs.get("refrigerant_type"))

    # Convertir ceros a None (desconocido)
    for k in [
        "annual_electricity_mwh", "annual_fuel_mwh", "electricity_price_eur_mwh", "fuel_price_eur_mwh",
        "co2_factor_elec_t_per_mwh", "co2_factor_fuel_t_per_mwh", "roof_area_m2"
    ]:
        if isinstance(company_inputs.get(k), (int, float)) and company_inputs.get(k) == 0.0:
            company_inputs[k] = None

    st.markdown("### 3) PESTEL (breve)")
    if "ai_pestel" not in st.session_state:
        st.session_state["ai_pestel"] = None
    if "ai_pestel_key" not in st.session_state:
        st.session_state["ai_pestel_key"] = None

    col_p_a, col_p_b = st.columns([1, 3])
    with col_p_a:
        gen_pestel = st.button("Generar PESTEL con IA", use_container_width=True)
    with col_p_b:
        st.caption("PESTEL generado por IA usando los inputs actuales.")

    pestel_key = json.dumps(company_inputs, sort_keys=True, ensure_ascii=False)
    if st.session_state["ai_pestel_key"] != pestel_key:
        st.session_state["ai_pestel_key"] = pestel_key
        st.session_state["ai_pestel"] = None

    if gen_pestel:
        try:
            with st.spinner("Generando PESTEL con IA..."):
                ai_p = generate_ai_pestel(
                    provider=ai_provider,
                    api_key=ai_api_key.strip(),
                    model=ai_model.strip(),
                    company_inputs=company_inputs,
                )
            st.session_state["ai_pestel"] = ai_p
            st.success("PESTEL generado.")
        except Exception as e:
            st.error(f"Fallo al generar PESTEL con IA: {e}")

    pestel = st.session_state["ai_pestel"]
    if pestel is None:
        st.info("PESTEL no generado. Puedes continuar con las iniciativas si quieres.")
    if pestel:
        pcols = st.columns(3)
        keys = list(pestel.keys())
        for i, k in enumerate(keys):
            with pcols[i % 3]:
                st.subheader(k)
                for bullet in pestel[k]:
                    st.write(f"- {bullet}")

    st.markdown("### 4) Iniciativas propuestas (editables antes de optimizar)")
    n_inits = 8

    if "ai_initiatives" not in st.session_state:
        st.session_state["ai_initiatives"] = None

    col_ai_a, col_ai_b = st.columns([1, 3])
    with col_ai_a:
        generate_ai = st.button("Generar 8 iniciativas con IA", use_container_width=True)
    with col_ai_b:
        st.caption("Usa todos los inputs de empresa para generar la tabla normativa completa.")

    if generate_ai:
        try:
            with st.spinner("Generando iniciativas con IA..."):
                ai_df = generate_ai_initiatives(
                    provider=ai_provider,
                    api_key=ai_api_key.strip(),
                    model=ai_model.strip(),
                    company_inputs=company_inputs,
                    n=n_inits,
                )
            st.session_state["ai_initiatives"] = ai_df.copy()
            st.success("Iniciativas generadas.")
        except Exception as e:
            st.error(f"Fallo al generar iniciativas con IA: {e}")

    if st.session_state["ai_initiatives"] is None:
        st.info("Pulsa 'Generar 8 iniciativas con IA' para crear la lista.")
        st.stop()

    df_base = st.session_state["ai_initiatives"].copy()
    df_base = normalize_columns(df_base)
    df_base = coerce_numeric(df_base)

    # Ensure required/optional columns exist
    for c in REQUIRED_COLUMNS:
        if c not in df_base.columns:
            df_base[c] = np.nan
    for c in OPTIONAL_COLUMNS:
        if c not in df_base.columns:
            df_base[c] = ""

    st.info("Puedes editar valores antes de optimizar. Si faltan datos, deja en blanco: se penaliza por confianza.")
    df_base = pd.DataFrame(
        st.data_editor(
            df_base,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config=column_config_es(list(df_base.columns)),
        )
    )

else:
    st.stop()

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

st.markdown("### 5) Evaluación de iniciativas")
cols_to_show = [
    "id",
    "initiative_family",
    "initiative",
    "scope",
    "emission_source",
    "activity_unit",
    "mrv_method",
    "data_dependency",
    "capex_eur",
    "annual_opex_saving_eur",
    "annual_co2_reduction_t",
    "co2_value_eur_per_year",
    "total_annual_benefit_eur",
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
st.dataframe(
    df[cols_to_show],
    use_container_width=True,
    hide_index=True,
    column_config=column_config_es(cols_to_show),
)

st.markdown("### 6) Optimización del portafolio")
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
c1.metric("Estado", summary["status"])
c2.metric("Iniciativas seleccionadas", summary["selected_count"])
c3.metric("CAPEX seleccionado (€)", f"{summary['capex_selected']:,.0f}")
c4.metric("Reducción CO₂ seleccionada (t/año)", f"{summary['co2_selected']:,.1f}")
st.metric("NPV penalizado total seleccionado (€)", f"{summary['npv_selected']:,.0f}")

st.markdown("#### Iniciativas seleccionadas")
selected_df = df_opt[df_opt["selected"]].copy()
st.dataframe(
    selected_df[cols_to_show],
    use_container_width=True,
    hide_index=True,
    column_config=column_config_es(cols_to_show),
)

# -----------------------------
# Charts (hardened)
# -----------------------------
st.markdown("### 7) Visuales")

chart_df = df_opt.copy()
chart_df["selected_label"] = np.where(chart_df["selected"], "Seleccionada", "No seleccionada")

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
        title="CAPEX vs reducción CO₂ (tamaño burbuja = beneficio anual)",
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
    title="NPV penalizado por iniciativa",
)
fig2.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# AI Copilot
# -----------------------------
st.markdown("### 8) Copiloto IA")

ai_extra_prompt = st.text_area(
    "Instrucción opcional para la IA (p. ej., flujo de caja, riesgo regulatorio u operaciones):",
    value="Prioriza recomendaciones accionables, con bajo riesgo de implementación y con enfoque MRV.",
)

assumptions = {
    "horizon_years": horizon_years,
    "discount_rate": discount_rate,
    "co2_price_eur_per_t": co2_price,
    "objective": objective,
    "weights": {"w_npv": w_npv, "w_co2": w_co2, "w_strategy": w_strategy},
}
constraints = {"budget_eur": budget_eur, "min_co2_t_per_year": min_co2_t}

if st.button("Generar informe ejecutivo IA", use_container_width=True):
    try:
        with st.spinner("Generando informe IA..."):
            ai_text = generate_ai_brief(
                provider=ai_provider,
                api_key=ai_api_key.strip(),
                model=ai_model.strip(),
                mode=mode,
                company_inputs=company_inputs if mode.startswith("A") else {},
                assumptions=assumptions,
                constraints=constraints,
                summary=summary,
                df_all=df_opt,
                df_selected=selected_df,
                pestel=pestel if mode.startswith("A") else {},
                extra_prompt=ai_extra_prompt.strip(),
            )
        st.success("Informe IA generado.")
        st.markdown(ai_text if ai_text else "_El modelo no devolvió contenido._")
    except Exception as e:
        st.error(f"Fallo en generación IA: {e}")

# -----------------------------
# Export
# -----------------------------
st.markdown("### 9) Exportar resultados")
export_cols = [
    "id",
    "initiative_family",
    "initiative",
    "scope",
    "emission_source",
    "activity_unit",
    "mrv_method",
    "capex_eur",
    "annual_opex_saving_eur",
    "annual_co2_reduction_t",
    "co2_value_eur_per_year",
    "implementation_months",
    "strategic_score_1_5",
    "npv_eur",
    "npv_penalized_eur",
    "payback_years",
    "required_info",
    "provided_info",
    "notes",
    "selected",
]
export_cols = [c for c in export_cols if c in df_opt.columns]
export = df_opt[export_cols].copy()

st.download_button(
    "Descargar resultados del portafolio (CSV)",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="portfolio_results.csv",
    mime="text/csv",
)
