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
import unicodedata
from contextlib import contextmanager
try:
    import tomllib
except Exception:
    tomllib = None
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import pulp
from scope2_electricity import build_scope2_ui, get_inventory_year as get_scope2_inventory_year

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Herramienta de Descarbonización Industrial",
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
    "categoria",            # quick_win / estrategica
    "priority_weight",      # weighting factor for prioritization
    "co2_adjusted_t",       # adjusted CO2 reduction for ranking
    "nombre",               # display alias
    "tiempo_implementacion",# display alias
    "thematic_bucket",      # diversity bucket
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
    "co2_adjusted_t",
    "implementation_months",
    "strategic_score_1_5",
    "confidence_0_1",
    "priority_weight",
]

# -----------------------------
# Methodology constants (placeholders; update with official sources)
# -----------------------------
# NOTE: These values are aligned to official sources where available.
# Update with official factors from MITECO/CNMC/REE and F-gas regulation when new years are published.

DEFAULT_INVENTORY_YEAR = 2025

CNMC_ELECTRICITY_FACTORS = {
    # Supplier Remaining Mix (tCO2/MWh) — CNMC published electricity mix (gCO2/kWh converted to tCO2/MWh)
    "supplier_remaining_mix_t_per_mwh": {
        2020: 0.250,
        2021: 0.259,
        2022: 0.273,
        2023: 0.260,
        2024: 0.283,
        2025: 0.258,
    },
    # Generic supplier factor when no specific supplier data available
    "generic_supplier_t_per_mwh": {
        2024: 0.283,
        2025: 0.258,
    },
    # GdO cogeneration factor (kg CO2e/kWh -> tCO2/MWh) from MITECO calculator instructions
    "gdo_cogen_t_per_mwh": {
        2024: 0.302,
        2025: 0.302,
    },
}

REE_LOCATION_BASED_FACTORS = {
    # REE system average (tCO2/MWh). Update with REE official location-based factors.
    "Peninsular": {
        2020: 0.250,
        2021: 0.259,
        2022: 0.273,
        2023: 0.260,
        2024: 0.283,
        2025: 0.258,
    },
    "No peninsular": {
        # Placeholder until official factors are available
    },
}

STATIONARY_FUEL_FACTORS_BY_KWH = {
    # tCO2 per kWh derived from MITECO NIR emission factors (t/TJ) -> t/MWh -> t/kWh
    "gas_natural": 0.00020196,
    "gasoleo": 0.00026676,
    "fueloil": 0.00028066,
    "glp": 0.00023040,
    "biomasa": 0.0,
}

STATIONARY_FUEL_FACTORS_BY_LITER = {
    # tCO2 per liter (approx). Update with MITECO official emission factors.
    "gasoleo": 0.00264,
    "fueloil": 0.00297,
    "glp": 0.00152,
}

FUEL_CONVERSION_KWH_PER_L = {
    # Approx kWh per liter. Used when only kWh factor exists.
    "gasoleo": 9.9,
    "fueloil": 10.6,
    "glp": 6.6,
    "gasolina": 8.6,
    "diesel": 9.9,
    "gasoline": 8.6,
}

MOBILE_FUEL_FACTORS_T_PER_L = {
    "car": {
        "diesel": 0.00264,
        "gasoline": 0.00231,
        "glp": 0.00152,
    },
    "truck": {
        "diesel": 0.00264,
        "gasoline": 0.00231,
        "glp": 0.00152,
    },
}

ADBLUE_FACTOR_T_PER_L = 0.0000005  # Placeholder; update if you choose to include AdBlue impact

SPAIN_PROVINCES = [
    "A Coruna", "Alava", "Albacete", "Alicante", "Almeria", "Asturias", "Avila", "Badajoz",
    "Barcelona", "Burgos", "Caceres", "Cadiz", "Cantabria", "Castellon", "Ceuta", "Ciudad Real",
    "Cordoba", "Cuenca", "Girona", "Granada", "Guadalajara", "Guipuzcoa", "Huelva", "Huesca",
    "Illes Balears", "Jaen", "La Rioja", "Las Palmas", "Leon", "Lleida", "Lugo", "Madrid",
    "Malaga", "Melilla", "Murcia", "Navarra", "Ourense", "Palencia", "Pontevedra", "Salamanca",
    "Santa Cruz de Tenerife", "Segovia", "Sevilla", "Soria", "Tarragona", "Teruel", "Toledo",
    "Valencia", "Valladolid", "Vizcaya", "Zamora", "Zaragoza",
]

STATIONARY_FUELS_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "stationary_fuel_factors_es.csv")
MOBILE_FUELS_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "mobile_fuel_factors_es.csv")
REFRIGERANTS_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "refrigerants_pca_es.csv")
COMMON_STATIONARY_FUEL_LABELS = {"Gasóleo C", "Gasóleo B", "Gas natural", "LPG"}
COMMON_MOBILE_FUEL_LABELS = {"B7", "E10"}
STATIONARY_FUEL_MWH_PER_UNIT = {
    "Gasóleo C": 0.0099,
    "Gasóleo B": 0.0099,
    "Gas natural": 0.001,
    "Fuelóleo": 0.0106,
    "LPG": 0.0066,
    "Queroseno": 0.0096,
    "Gas propano": 0.0138,
    "Gas butano": 0.0137,
    "Biomasa madera": 0.0042,
    "Biomasa pellets": 0.0048,
    "Biomasa astillas": 0.0035,
    "Biomasa serrines virutas": 0.0041,
    "Biomasa cáscara f. secos": 0.0046,
    "Biomasa hueso aceituna": 0.0050,
    "B7": 0.0099,
    "B10": 0.0098,
    "B20": 0.0097,
    "B30": 0.0095,
    "B100": 0.0092,
    "E5": 0.0085,
    "E10": 0.0083,
    "E85": 0.0067,
    "E100": 0.0059,
}

MOBILE_FUEL_MWH_PER_UNIT = {
    "B7": 0.0099,
    "B10": 0.0098,
    "B20": 0.0097,
    "B30": 0.0095,
    "B100": 0.0092,
    "E5": 0.0085,
    "E10": 0.0083,
    "E85": 0.0067,
    "E100": 0.0059,
    "LPG": 0.0066,
    "CNG": 0.0133,
    "LNG": 0.0139,
    "AdBlue": 0.0,
}


def _year_factor_map(row: pd.Series) -> Dict[int, float]:
    factors: Dict[int, float] = {}
    for col in row.index:
        if str(col).isdigit() and not pd.isna(row[col]):
            factors[int(col)] = float(row[col])
    return factors


def load_stationary_fuels_catalog() -> List[Dict]:
    df = pd.read_csv(STATIONARY_FUELS_DB_PATH)
    required_cols = {"Combustible", "Unidad"}
    if not required_cols.issubset(df.columns):
        raise ValueError("El catálogo de combustibles estacionarios no tiene el formato esperado.")
    fuels = []
    for _, row in df.iterrows():
        label = str(row["Combustible"]).strip()
        key = (
            unicodedata.normalize("NFKD", label)
            .encode("ascii", "ignore")
            .decode("ascii")
            .lower()
            .replace(".", "")
            .replace(" ", "_")
        )
        fuels.append(
            {
                "key": key,
                "label": label,
                "unit": str(row["Unidad"]).strip(),
                "common": label in COMMON_STATIONARY_FUEL_LABELS,
                "mwh_per_unit": STATIONARY_FUEL_MWH_PER_UNIT.get(label),
                "factors_kg_per_unit": _year_factor_map(row),
            }
        )
    return fuels


STATIONARY_FUELS_CATALOG = load_stationary_fuels_catalog()
STATIONARY_FUELS_BY_KEY = {f["key"]: f for f in STATIONARY_FUELS_CATALOG}
COMMON_STATIONARY_FUELS = [f for f in STATIONARY_FUELS_CATALOG if f.get("common")]


def load_mobile_fuels_catalog() -> List[Dict]:
    df = pd.read_csv(MOBILE_FUELS_DB_PATH)
    required_cols = {"Combustible", "Tipo", "Unidad"}
    if not required_cols.issubset(df.columns):
        raise ValueError("El catálogo de combustibles móviles no tiene el formato esperado.")
    fuels = []
    for _, row in df.iterrows():
        fuel_label = str(row["Combustible"]).strip()
        vehicle_type = str(row["Tipo"]).strip()
        key = (
            unicodedata.normalize("NFKD", f"{fuel_label}_{vehicle_type}")
            .encode("ascii", "ignore")
            .decode("ascii")
            .lower()
            .replace(".", "")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("-", "_")
        )
        fuels.append(
            {
                "key": key,
                "fuel_label": fuel_label,
                "vehicle_type": vehicle_type,
                "unit": str(row["Unidad"]).strip(),
                "common": fuel_label in COMMON_MOBILE_FUEL_LABELS,
                "mwh_per_unit": MOBILE_FUEL_MWH_PER_UNIT.get(fuel_label),
                "factors_kg_per_unit": _year_factor_map(row),
            }
        )
    return fuels


MOBILE_FUELS_CATALOG = load_mobile_fuels_catalog()
MOBILE_FUELS_BY_KEY = {f["key"]: f for f in MOBILE_FUELS_CATALOG}
COMMON_MOBILE_FUELS = [f for f in MOBILE_FUELS_CATALOG if f.get("common")]


def load_refrigerants_catalog() -> List[Dict]:
    df = pd.read_csv(REFRIGERANTS_DB_PATH)
    required_cols = {"Nombre", "Formula_quimica", "PCA_6AR"}
    if not required_cols.issubset(df.columns):
        raise ValueError("El catálogo de refrigerantes no tiene el formato esperado.")
    out = []
    for _, row in df.iterrows():
        name = str(row["Nombre"]).strip()
        out.append({"name": name, "gwp": float(row["PCA_6AR"])})
    return out


REFRIGERANTS_CATALOG = load_refrigerants_catalog()
REFRIGERANTS_BY_NAME = {r["name"].upper(): r for r in REFRIGERANTS_CATALOG}


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

# -----------------------------
# Home page
# -----------------------------
def render_global_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
        :root {
            --inputs-strong: #2563eb;
            --inputs-soft: #e3f2fd;
            --inputs-border: rgba(37, 99, 235, 0.18);
            --carbon-strong: #0ea5a7;
            --carbon-soft: #e0f7fa;
            --carbon-border: rgba(14, 165, 167, 0.18);
            --pestel-strong: #2f855a;
            --pestel-soft: #e8f5e9;
            --pestel-border: rgba(47, 133, 90, 0.18);
            --portfolio-strong: #ea580c;
            --portfolio-soft: #fff3e0;
            --portfolio-border: rgba(234, 88, 12, 0.20);
            --phase-text: #1f2937;
        }
        div[class*="st-key-phase-shell-"],
        div[class*="st-key-phase-shell-"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            width: 100%;
            border-radius: 22px;
            padding: 0.35rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        div[class*="st-key-phase-shell-inputs"],
        div[class*="st-key-phase-shell-inputs"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--inputs-soft);
            border-color: var(--inputs-border);
        }
        div[class*="st-key-phase-shell-carbon"],
        div[class*="st-key-phase-shell-carbon"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--carbon-soft);
            border-color: var(--carbon-border);
        }
        div[class*="st-key-phase-shell-pestel"],
        div[class*="st-key-phase-shell-pestel"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--pestel-soft);
            border-color: var(--pestel-border);
        }
        div[class*="st-key-phase-shell-portfolio"],
        div[class*="st-key-phase-shell-portfolio"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--portfolio-soft);
            border-color: var(--portfolio-border);
        }
        .phase-block {
            margin: 0.2rem 0 1.5rem;
        }
        .phase-block-header {
            padding: 1.1rem 1.25rem;
            border-radius: 18px;
            color: #ffffff;
            margin-bottom: 1rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
        }
        .phase-block-kicker {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.88;
            margin-bottom: 0.35rem;
        }
        .phase-block-title {
            font-size: 1.55rem;
            line-height: 1.1;
            font-weight: 800;
            margin: 0;
        }
        .phase-block-subtitle {
            margin-top: 0.45rem;
            font-size: 0.95rem;
            line-height: 1.5;
            color: rgba(255, 255, 255, 0.92);
        }
        .block-inputs .phase-block-header {
            background: var(--inputs-strong);
        }
        .block-carbon .phase-block-header {
            background: var(--carbon-strong);
        }
        .block-pestel .phase-block-header {
            background: var(--pestel-strong);
        }
        .block-portfolio .phase-block-header {
            background: var(--portfolio-strong);
        }
        .phase-callout {
            padding: 0.95rem 1rem;
            border-radius: 14px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: rgba(255, 255, 255, 0.55);
            margin-bottom: 1rem;
        }
        .phase-callout strong {
            display: block;
            margin-bottom: 0.35rem;
            color: var(--phase-text);
        }
        .phase-callout p {
            margin: 0;
            color: #475569;
        }
        div[data-testid="stVerticalBlock"]:has(.phase-button-pestel) .stButton > button,
        div[data-testid="stVerticalBlock"]:has(.phase-button-pestel) .stButton > button[kind="primary"] {
            background: var(--pestel-strong) !important;
            border-color: var(--pestel-strong) !important;
            color: #ffffff !important;
        }
        div[data-testid="stVerticalBlock"]:has(.phase-button-pestel) .stButton > button:hover,
        div[data-testid="stVerticalBlock"]:has(.phase-button-pestel) .stButton > button[kind="primary"]:hover {
            background: #276c49 !important;
            border-color: #276c49 !important;
            color: #ffffff !important;
        }
        div[data-testid="stVerticalBlock"]:has(.phase-button-portfolio) .stButton > button,
        div[data-testid="stVerticalBlock"]:has(.phase-button-portfolio) .stButton > button[kind="secondary"] {
            background: var(--portfolio-strong) !important;
            border-color: var(--portfolio-strong) !important;
            color: #ffffff !important;
        }
        div[data-testid="stVerticalBlock"]:has(.phase-button-portfolio) .stButton > button:hover,
        div[data-testid="stVerticalBlock"]:has(.phase-button-portfolio) .stButton > button[kind="secondary"]:hover {
            background: #c2410c !important;
            border-color: #c2410c !important;
            color: #ffffff !important;
        }
        .hero-panel {
            padding: 2.4rem 2.2rem;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 20px;
            background: linear-gradient(135deg, #f7faf8 0%, #eef5f1 100%);
            margin-bottom: 1.4rem;
        }
        .eyebrow {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            background: #e6f0ea;
            color: #274c3c;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.9rem;
        }
        .hero-title {
            font-size: 2.35rem;
            line-height: 1.1;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.8rem;
        }
        .hero-subtitle { font-size: 1.1rem; color: #1f2937; margin-bottom: 0.8rem; }
        .hero-text { font-size: 1rem; color: #475569; max-width: 900px; }
        .section-card, .method-card, .step-card, .mini-card {
            border-radius: 16px;
            border: 1px solid rgba(15, 23, 42, 0.10);
            background: #ffffff;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
        }
        .section-card { padding: 1.1rem 1rem; min-height: 180px; }
        .mini-card {
            padding: 1rem 0.95rem;
            min-height: 160px;
            background: #f8fbf9;
            border-left: 4px solid #436850;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.7rem;
            margin-bottom: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.28rem;
            font-weight: 800;
            letter-spacing: 0.01em;
            padding: 1rem 1.35rem;
            border-radius: 16px 16px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background: #eef5f1;
            color: #1f4d3a;
        }
        .result-hero {
            padding: 1.35rem 1.4rem;
            border-radius: 20px;
            border: 1px solid rgba(15, 23, 42, 0.10);
            background: linear-gradient(135deg, #f7faf8 0%, #eef5f1 100%);
            margin-bottom: 1rem;
        }
        .result-overline {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #436850;
            margin-bottom: 0.45rem;
        }
        .result-total {
            font-size: 2.4rem;
            line-height: 1;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }
        .result-subtle {
            font-size: 0.95rem;
            color: #475569;
        }
        .result-card {
            padding: 1rem 1rem 0.9rem;
            border-radius: 16px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: #ffffff;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }
        .result-card h4 {
            margin: 0 0 0.35rem;
            font-size: 1rem;
            color: #0f172a;
        }
        .method-card { padding: 1rem; min-height: 175px; background: #fcfdfc; }
        .step-card { padding: 1rem; min-height: 150px; background: #f7f9fb; }
        .step-number {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: #1f4d3a;
            color: white;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }
        .cta-panel {
            padding: 1.8rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #eef5f1 0%, #f8fafc 100%);
            border: 1px solid rgba(15, 23, 42, 0.08);
            margin-top: 1rem;
        }
        .stButton > button[kind="primary"] {
            background: #2563eb;
            border-color: #2563eb;
            color: #ffffff;
        }
        .stButton > button[kind="secondary"] {
            background: #2f855a;
            border-color: #2f855a;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def render_phase_block(title: str, theme: str, subtitle: str = ""):
    phase_idx = st.session_state.get("_phase_marker_idx", 0)
    st.session_state["_phase_marker_idx"] = phase_idx + 1
    with st.container(border=True, key=f"phase-shell-{theme}-{phase_idx}"):
        st.markdown(
            f"""
            <div class="block-{theme} phase-block">
                <div class="phase-block-header">
                    <div class="phase-block-kicker">Fase metodológica</div>
                    <div class="phase-block-title">{title}</div>
                    {"<div class='phase-block-subtitle'>" + subtitle + "</div>" if subtitle else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        yield


def render_phase_action_button(label: str, button_theme: str, *, type: str = "secondary", key: Optional[str] = None) -> bool:
    with st.container():
        st.markdown(f'<div class="phase-button-{button_theme}"></div>', unsafe_allow_html=True)
        return st.button(label, use_container_width=True, type=type, key=key)


def render_home_page() -> None:
    render_global_styles()

    st.markdown(
        """
        <div class="hero-panel">
            <div class="eyebrow">TFG · Descarbonización industrial</div>
            <div class="hero-title">Herramienta de apoyo para la descarbonización industrial</div>
            <div class="hero-subtitle">
                Calcula la huella de carbono de Alcance 1 y 2, genera contexto estratégico y prioriza iniciativas de reducción con criterios técnicos y financieros.
            </div>
            <div class="hero-text">
                Esta herramienta ha sido desarrollada como prototipo de TFG para ayudar a empresas industriales a estructurar su plan de descarbonización de forma trazable, comprensible y accionable.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Qué ofrece esta herramienta")
    benefit_cols = st.columns(4)
    benefits = [
        ("Trazabilidad del cálculo", "Integra una lógica explícita para cuantificar emisiones energéticas y documentar supuestos de cálculo."),
        ("Priorización de iniciativas", "Ordena opciones de reducción según impacto ambiental, viabilidad económica y criterios de decisión."),
        ("Apoyo a inversión", "Conecta la huella con CAPEX, ahorro operativo, NPV y restricciones presupuestarias."),
        ("Visión para roadmap", "Permite construir una primera base de trabajo para un plan de descarbonización industrial."),
    ]
    for col, (title, text) in zip(benefit_cols, benefits):
        with col:
            st.markdown(f'<div class="mini-card"><strong>{title}</strong><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown("### ¿Por qué necesita una empresa un plan de descarbonización?")
    reason_cols = st.columns(4)
    reasons = [
        ("Cumplimiento y reporting", "La presión regulatoria y los requerimientos de reporte avanzan hacia una mayor exigencia de medición, trazabilidad y seguimiento."),
        ("Competitividad y costes", "La energía y el carbono afectan al margen industrial. Medir permite identificar oportunidades de ahorro y reducción de exposición."),
        ("Cadena de suministro y reputación", "Clientes, financiadores y grupos de interés exigen mayor transparencia climática y capacidad de respuesta."),
        ("Priorización de inversiones", "Un plan ordenado ayuda a comparar medidas y secuenciar decisiones con un criterio técnico, económico y operativo."),
    ]
    for col, (title, text) in zip(reason_cols, reasons):
        with col:
            st.markdown(f'<div class="section-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown("### ¿Qué significan los Alcances 1, 2 y 3?")
    scope_cols = st.columns(3)
    scopes = [
        ("Alcance 1", "Emisiones directas generadas por fuentes que controla la empresa.", "Combustibles en planta, flota propia y fugas de refrigerantes."),
        ("Alcance 2", "Emisiones indirectas asociadas a la energía comprada y consumida por la organización.", "Electricidad adquirida y calor o vapor comprados."),
        ("Alcance 3", "Resto de emisiones indirectas a lo largo de la cadena de valor.", "Compras, transporte, viajes, residuos, uso de producto o fin de vida."),
    ]
    for col, (title, desc, examples) in zip(scope_cols, scopes):
        with col:
            st.markdown(
                f'<div class="section-card"><h4>{title}</h4><p>{desc}</p><p style="margin-top:0.7rem;"><strong>Ejemplos:</strong> {examples}</p></div>',
                unsafe_allow_html=True,
            )
    st.info("Esta versión de la herramienta calcula Alcance 1 y 2. El Alcance 3 se incorpora a nivel conceptual, pero no se estima automáticamente en esta versión.")

    st.markdown("### Fuentes metodológicas y factores de emisión")
    source_cols = st.columns(4)
    sources = [
        ("Alcance 1", "Factores y metodología alineados con MITECO y con referencias regulatorias españolas aplicables a combustibles y cálculo de inventarios."),
        ("Electricidad market-based", "Uso de información de comercializadora, GdO, etiquetado o supplier mix con enfoque alineado con CNMC."),
        ("Electricidad location-based", "Factor medio del sistema eléctrico de referencia, alineado con el enfoque de REE."),
        ("Refrigerantes", "Potenciales de calentamiento global basados en referencias regulatorias de gases fluorados y catálogos GWP/PCA aplicables."),
    ]
    for col, (title, text) in zip(source_cols, sources):
        with col:
            st.markdown(f'<div class="method-card"><strong>{title}</strong><p style="margin-top:0.55rem; color:#475569;">{text}</p></div>', unsafe_allow_html=True)

    st.markdown("### ¿Cómo funciona la herramienta?")
    steps = [
        ("1", "Introducción de datos", "Se recogen datos básicos de la empresa, consumos energéticos, flota, refrigerantes y condicionantes operativos."),
        ("2", "Cálculo de huella", "La herramienta estima la huella de carbono de Alcance 1 y 2 a partir de factores de emisión y supuestos explícitos."),
        ("3", "Contexto PESTEL", "De forma opcional, se genera un contexto estratégico con IA para situar riesgos, impulsores y condicionantes externos."),
        ("4", "Iniciativas", "Se proponen medidas de descarbonización editables antes de pasar a la evaluación económica y ambiental."),
        ("5", "Evaluación", "Cada iniciativa se analiza en términos de reducción de CO2, ahorro, CAPEX, payback y valor financiero."),
        ("6", "Optimización y exportación", "Se construye un portafolio priorizado según presupuesto y objetivo, con visuales y descarga de resultados."),
    ]
    for col, (num, title, text) in zip(st.columns(3) + st.columns(3), steps):
        with col:
            st.markdown(
                f'<div class="step-card"><div class="step-number">{num}</div><strong>{title}</strong><p style="margin-top:0.55rem; color:#475569;">{text}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="cta-panel">
            <strong>Acceso al módulo operativo</strong>
            <p style="margin-top:0.6rem; color:#475569;">
                Cuando quieras, puedes pasar al módulo de cálculo para estimar tu huella y construir un primer plan de descarbonización.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Empezar cálculo de huella", type="primary", use_container_width=True):
        st.session_state["current_page"] = "tool"
        st.session_state["scroll_to_top"] = True
        st.rerun()


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def column_config_es(cols: List[str]) -> dict:
    labels = {
        "id": "ID",
        "nombre": "Nombre",
        "initiative_family": "Familia",
        "initiative": "Iniciativa",
        "categoria": "Categoría",
        "capex_eur": "CAPEX (€)",
        "annual_opex_saving_eur": "Ahorro OPEX anual (€)",
        "annual_co2_reduction_t": "Reducción CO₂ anual (t)",
        "co2_adjusted_t": "CO₂ ajustado (t)",
        "tiempo_implementacion": "Tiempo implementación",
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


def _to_float_or_zero(value) -> float:
    if value is None:
        return 0.0
    try:
        if pd.isna(value):
            return 0.0
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return 0.0


def _latest_factor(factors_by_year: Dict[int, float], year: int) -> Tuple[float, int]:
    if not factors_by_year:
        return 0.0, year
    if year in factors_by_year:
        return float(factors_by_year[year]), year
    latest_year = max(factors_by_year.keys())
    return float(factors_by_year[latest_year]), latest_year


def get_stationary_fuel_factor(fuel_key: str, year: int) -> Tuple[float, int]:
    fuel = STATIONARY_FUELS_BY_KEY.get(fuel_key)
    if not fuel:
        return 0.0, year
    raw_factors = fuel.get("factors_kg_per_unit", {}) or {}
    factors = {int(k): float(v) for k, v in raw_factors.items()}
    return _latest_factor(factors, year)


def get_stationary_fuel_entries(company: Dict) -> List[Dict]:
    entries = []
    for row in company.get("stationary_fuels", []) or []:
        fuel_key = str(row.get("fuel_key") or "").strip()
        quantity = _to_float_or_zero(row.get("quantity"))
        if fuel_key and quantity > 0:
            entries.append({"fuel_key": fuel_key, "quantity": quantity})
    return entries


def get_mobile_fuel_factor(fuel_key: str, year: int) -> Tuple[float, int]:
    fuel = MOBILE_FUELS_BY_KEY.get(fuel_key)
    if not fuel:
        return 0.0, year
    return _latest_factor(fuel.get("factors_kg_per_unit", {}), year)


def get_mobile_fuel_entries(company: Dict) -> List[Dict]:
    entries = []
    for row in company.get("mobile_fuels", []) or []:
        fuel_key = str(row.get("fuel_key") or "").strip()
        quantity = _to_float_or_zero(row.get("quantity"))
        if fuel_key and quantity > 0:
            entries.append({"fuel_key": fuel_key, "quantity": quantity})
    return entries


def get_refrigerant_entries(company: Dict) -> List[Dict]:
    entries = []
    for row in company.get("refrigerants", []) or []:
        name = str(row.get("name") or "").strip()
        quantity = _to_float_or_zero(row.get("quantity"))
        if name and quantity > 0:
            entries.append({"name": name, "quantity": quantity})
    return entries


def get_market_based_electricity_factor(
    year: int,
    supplier_factor_t_per_mwh: float | None,
    supplier_known: bool,
) -> Tuple[float, str]:
    if supplier_factor_t_per_mwh and supplier_factor_t_per_mwh > 0:
        return float(supplier_factor_t_per_mwh), "comercializadora (factor aportado)"

    if supplier_known:
        f, y = _latest_factor(CNMC_ELECTRICITY_FACTORS.get("supplier_remaining_mix_t_per_mwh", {}), year)
        if f > 0:
            suffix = " (fallback año disponible)" if y != year else ""
            return f, f"Supplier Remaining Mix CNMC ({y}){suffix}"

    f, y = _latest_factor(CNMC_ELECTRICITY_FACTORS.get("generic_supplier_t_per_mwh", {}), year)
    suffix = " (fallback año disponible)" if y != year else ""
    return f, f"fallback interno (mix genérico {y}){suffix}"


def get_location_based_electricity_factor(
    year: int,
    system: str,
) -> Tuple[float, str]:
    system_key = system if system in REE_LOCATION_BASED_FACTORS else "Peninsular"
    factors = REE_LOCATION_BASED_FACTORS.get(system_key, {})
    if not factors:
        system_key = "Peninsular"
        factors = REE_LOCATION_BASED_FACTORS.get(system_key, {})
        f, y = _latest_factor(factors, year)
        suffix = " (fallback sistema Peninsular)" if system != "Peninsular" else ""
        return f, f"REE location-based ({system_key}, {y}){suffix}"
    f, y = _latest_factor(factors, year)
    suffix = " (fallback año disponible)" if y != year else ""
    return f, f"REE location-based ({system_key}, {y}){suffix}"


def calculate_scope2_electricity(company: Dict) -> Dict[str, float | str]:
    elec_mwh = _to_float_or_zero(company.get("annual_electricity_mwh"))
    year = get_scope2_inventory_year(company)
    method = (company.get("electricity_method") or "location").lower()

    if elec_mwh <= 0:
        return {
            "emissions_t": 0.0,
            "used_factor": 0.0,
            "source": "sin consumo eléctrico",
            "method": method,
        }

    if method == "location":
        factor = _to_float_or_zero(company.get("scope2_location_factor_kg_kwh"))
        emissions_kg = _to_float_or_zero(company.get("scope2_location_emissions_kg"))
        return {
            "emissions_t": emissions_kg / 1000.0,
            "used_factor": factor,
            "source": f"factor REE España ({year})",
            "method": "location-based",
        }

    factor = _to_float_or_zero(company.get("scope2_market_factor_kg_kwh"))
    emissions_kg = _to_float_or_zero(company.get("scope2_market_emissions_kg"))
    return {
        "emissions_t": emissions_kg / 1000.0,
        "used_factor": factor,
        "source": f"comercializadoras + GdO ({year})",
        "method": "market-based",
    }


def calculate_scope2_heat(company: Dict) -> Dict[str, float | str]:
    heat_mwh = _to_float_or_zero(company.get("annual_purchased_heat_mwh"))
    heat_factor = _to_float_or_zero(company.get("co2_factor_heat_t_per_mwh"))
    used_heat_factor = heat_factor if heat_factor > 0 else 0.20
    scope2_heat_t = heat_mwh * used_heat_factor
    source = "factor calor aportado por la empresa" if heat_factor > 0 else "fallback interno (calor/vapor)"
    return {"emissions_t": scope2_heat_t, "used_factor": used_heat_factor, "source": source}


def convert_fuel_to_emissions(fuel_key: str, quantity: float, unit: str, by_kwh: Dict, by_liter: Dict) -> float:
    unit_norm = (unit or "").lower()
    if unit_norm in ["kwh", "mwh"]:
        kwh = quantity * (1000.0 if unit_norm == "mwh" else 1.0)
        factor = by_kwh.get(fuel_key, 0.0)
        return kwh * factor
    if unit_norm in ["l", "litros", "liters"]:
        if fuel_key in by_liter:
            return quantity * by_liter[fuel_key]
        kwh_per_l = FUEL_CONVERSION_KWH_PER_L.get(fuel_key, 0.0)
        factor = by_kwh.get(fuel_key, 0.0)
        return quantity * kwh_per_l * factor
    return 0.0


def calculate_scope1_stationary(company: Dict) -> Dict[str, float | Dict[str, float] | str]:
    year = int(company.get("inventory_year") or DEFAULT_INVENTORY_YEAR)
    breakdown = {}
    total = 0.0
    years_used = set()
    for entry in get_stationary_fuel_entries(company):
        fuel = STATIONARY_FUELS_BY_KEY.get(entry["fuel_key"])
        if not fuel:
            continue
        factor_kg_per_unit, factor_year = get_stationary_fuel_factor(entry["fuel_key"], year)
        emissions_t = entry["quantity"] * factor_kg_per_unit / 1000.0
        breakdown[fuel["label"]] = breakdown.get(fuel["label"], 0.0) + emissions_t
        total += emissions_t
        years_used.add(factor_year)

    if not years_used:
        source = "sin combustibles estacionarios reportados"
    elif len(years_used) == 1:
        source = f"catálogo local de factores para instalaciones fijas ({list(years_used)[0]})"
    else:
        source = f"catálogo local de factores para instalaciones fijas ({min(years_used)}-{max(years_used)})"
    return {"emissions_t": total, "breakdown": breakdown, "source": source}


def estimate_stationary_fuel_mwh(company: Dict) -> float:
    total_mwh = 0.0
    for entry in get_stationary_fuel_entries(company):
        fuel = STATIONARY_FUELS_BY_KEY.get(entry["fuel_key"])
        if not fuel:
            continue
        mwh_per_unit = fuel.get("mwh_per_unit")
        if isinstance(mwh_per_unit, (int, float)) and mwh_per_unit > 0:
            total_mwh += entry["quantity"] * float(mwh_per_unit)
    return total_mwh


def calculate_scope1_mobile(company: Dict) -> Dict[str, float | List[dict]]:
    year = int(company.get("inventory_year") or DEFAULT_INVENTORY_YEAR)
    rows = get_mobile_fuel_entries(company)
    total = 0.0
    details = []
    for r in rows:
        fuel = MOBILE_FUELS_BY_KEY.get(r["fuel_key"])
        if not fuel:
            continue
        qty = r["quantity"]
        factor_kg_per_unit, factor_year = get_mobile_fuel_factor(r["fuel_key"], year)
        t = qty * factor_kg_per_unit / 1000.0
        total += t
        details.append(
            {
                "fuel": fuel["fuel_label"],
                "vehicle": fuel["vehicle_type"],
                "t": t,
                "adblue_t": 0.0,
                "note": f"factor {factor_kg_per_unit:.3f} kgCO2/{fuel['unit']} ({factor_year})",
            }
        )

    return {"emissions_t": total, "details": details}


def calculate_fugitive_emissions(company: Dict) -> Dict[str, float | str | bool]:
    total = 0.0
    found = True
    details = []
    for entry in get_refrigerant_entries(company):
        name = entry["name"]
        kg = entry["quantity"]
        ref = REFRIGERANTS_BY_NAME.get(name.upper())
        if not ref:
            found = False
            continue
        gwp = ref["gwp"]
        emissions_t = kg * gwp / 1000.0
        total += emissions_t
        details.append({"name": name, "gwp": gwp, "kg": kg, "emissions_t": emissions_t})

    primary = details[0] if details else {"name": "", "gwp": 0.0}
    return {
        "emissions_t": total,
        "gwp": primary["gwp"],
        "source": "catálogo local PCA 6AR",
        "found": found,
        "details": details,
        "primary_name": primary["name"],
    }


def get_data_quality_score(company: Dict, footprint_meta: Dict) -> Tuple[str, str]:
    score = 0
    if company.get("has_invoices"):
        score += 2
    if company.get("has_meters"):
        score += 2
    if company.get("has_submetering"):
        score += 1
    if company.get("cnmc_supplier_known"):
        score += 1
    if company.get("electricity_has_gdo"):
        score += 1
    if company.get("has_energy_audit"):
        score += 2
    if footprint_meta.get("used_elec_factor", 0) > 0:
        score += 1

    if score >= 7:
        return "Alta", "Datos con alta trazabilidad (facturas/medición/auditoría)."
    if score >= 4:
        return "Media", "Datos razonables; mejorar trazabilidad con facturas y medición."
    return "Baja", "Datos incompletos; incorporar facturas, factores oficiales y auditoría."


def calculate_company_footprint(company: Dict) -> Dict[str, float | str | bool]:
    # Scope 1
    scope1_stationary = calculate_scope1_stationary(company)
    scope1_mobile = calculate_scope1_mobile(company)
    scope1_fugitive = calculate_fugitive_emissions(company)

    # Scope 2
    scope2_electricity = calculate_scope2_electricity(company)
    scope2_heat = calculate_scope2_heat(company)

    scope1_t = scope1_stationary["emissions_t"] + scope1_mobile["emissions_t"] + scope1_fugitive["emissions_t"]
    scope2_t = scope2_electricity["emissions_t"] + scope2_heat["emissions_t"]
    total_t = scope1_t + scope2_t

    return {
        "scope1_t": scope1_t,
        "scope2_t": scope2_t,
        "scope3_t": 0.0,
        "total_t": total_t,
        "scope1_stationary_t": scope1_stationary["emissions_t"],
        "scope1_fleet_t": scope1_mobile["emissions_t"],
        "scope1_fugitive_t": scope1_fugitive["emissions_t"],
        "scope2_elec_t": scope2_electricity["emissions_t"],
        "scope2_heat_t": scope2_heat["emissions_t"],
        "scope1_factor_source": scope1_stationary["source"],
        "used_elec_factor": scope2_electricity["used_factor"],
        "used_heat_factor": scope2_heat["used_factor"],
        "scope2_elec_source": scope2_electricity["source"],
        "scope2_elec_method": scope2_electricity["method"],
        "refrigerant_factor_found": scope1_fugitive["found"],
        "refrigerant_key": scope1_fugitive.get("primary_name", ""),
        "refrigerant_gwp": scope1_fugitive["gwp"],
        "refrigerant_source": scope1_fugitive["source"],
    }

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


def build_ai_company_context(company_inputs: Dict) -> Dict[str, Any]:
    stationary = get_stationary_fuel_entries(company_inputs)
    mobile = get_mobile_fuel_entries(company_inputs)
    refrigerants = get_refrigerant_entries(company_inputs)
    annual_electricity_mwh = _to_float_or_zero(company_inputs.get("annual_electricity_mwh"))
    annual_fuel_mwh = _to_float_or_zero(company_inputs.get("annual_fuel_mwh"))
    total_energy_mwh = annual_electricity_mwh + annual_fuel_mwh
    if total_energy_mwh >= 20000:
        size_band = "muy alta intensidad energética"
    elif total_energy_mwh >= 5000:
        size_band = "intensidad energética media"
    elif total_energy_mwh > 0:
        size_band = "intensidad energética moderada o baja"
    else:
        size_band = "sin suficiente dato energético"
    return {
        "company_name": (company_inputs.get("company_name") or "").strip(),
        "sector": company_inputs.get("sector") or company_inputs.get("cnae_sector") or "",
        "country": company_inputs.get("country") or "",
        "province": company_inputs.get("province") or "",
        "postal_code": company_inputs.get("postal_code") or "",
        "country_region": company_inputs.get("country_region") or "",
        "inventory_year": company_inputs.get("inventory_year") or DEFAULT_INVENTORY_YEAR,
        "electricity_method": company_inputs.get("electricity_method") or "location",
        "annual_electricity_mwh": annual_electricity_mwh,
        "annual_purchased_heat_mwh": company_inputs.get("annual_purchased_heat_mwh"),
        "annual_fuel_mwh": annual_fuel_mwh,
        "total_energy_mwh": total_energy_mwh,
        "size_signal": size_band,
        "has_fleet": bool(company_inputs.get("has_fleet", False)),
        "has_refrigerants": bool(company_inputs.get("has_refrigerants", False)),
        "roof_area_m2": company_inputs.get("roof_area_m2"),
        "stationary_fuels_reported": [STATIONARY_FUELS_BY_KEY[r["fuel_key"]]["label"] for r in stationary if r.get("fuel_key") in STATIONARY_FUELS_BY_KEY],
        "mobile_fuels_reported": [
            {
                "fuel": MOBILE_FUELS_BY_KEY[r["fuel_key"]]["fuel_label"],
                "vehicle": MOBILE_FUELS_BY_KEY[r["fuel_key"]]["vehicle_type"],
            }
            for r in mobile
            if r.get("fuel_key") in MOBILE_FUELS_BY_KEY
        ],
        "refrigerants_reported": [r.get("name") for r in refrigerants if r.get("name")],
        "implemented_measures": company_inputs.get("implemented_measures") or {},
        "financial_context": {
            "horizon_years": company_inputs.get("horizon_years"),
            "discount_rate": company_inputs.get("discount_rate"),
            "carbon_price_eur_t": company_inputs.get("carbon_price_eur_t"),
            "capex_budget_eur": company_inputs.get("capex_budget_eur"),
            "max_payback_years": company_inputs.get("max_payback_years"),
        },
    }


def gemini_should_use_web_research(company_inputs: Dict) -> bool:
    return bool((company_inputs.get("company_name") or "").strip())


def generate_ai_pestel(
    provider: str,
    api_key: str,
    model: str,
    company_inputs: Dict,
) -> Dict[str, List[str]]:
    if not api_key:
        raise RuntimeError(f"Falta la API key de {provider}.")

    company_context = build_ai_company_context(company_inputs)
    use_web_research = gemini_should_use_web_research(company_inputs)

    system_prompt = (
        "Eres un consultor senior de descarbonización industrial. "
        "Devuelve SOLO JSON válido, sin texto adicional. "
        "Responde en español. "
        "Estructura: claves Político, Económico, Social, Tecnológico, Ambiental, Legal "
        "y cada valor es una lista de bullets cortos. "
        "El análisis debe estar adaptado al sector, a la localización de la empresa en España y a su perfil operativo/energético. "
        "Evita bullets genéricos que podrían aplicarse a cualquier empresa. "
        "Si se proporciona nombre de empresa y dispones de búsqueda web, úsala para identificar qué vende la empresa, qué productos/servicios ofrece, "
        "qué procesos productivos o logísticos parecen relevantes y qué rasgos de su cadena de suministro pueden afectar a la descarbonización. "
        "Si algo no se puede verificar, indícalo con prudencia y apóyate en sector + localización + datos introducidos."
    )

    user_prompt = (
        "Genera un PESTEL breve y accionable (2-3 bullets por categoría) "
        "basado en los datos de la empresa y, si hay nombre de empresa y la herramienta lo permite, en información pública verificable.\n\n"
        "Reglas:\n"
        "- Ten en cuenta expresamente el sector y la provincia/ubicación.\n"
        "- Da prioridad a implicaciones reales para consumo energético, combustibles, calor de proceso, logística, regulación e inversión.\n"
        "- No repitas obviedades vacías.\n"
        "- Si haces una inferencia, que sea razonable y ligada al contexto.\n\n"
        f"COMPANY_CONTEXT: {json.dumps(company_context, ensure_ascii=False)}\n"
        f"RAW_INPUTS: {json.dumps(company_inputs, ensure_ascii=False)}"
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
        if use_web_research:
            payload["tools"] = [{"google_search": {}}]
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


def classify_initiative_category(row: pd.Series) -> str:
    name = str(row.get("initiative") or row.get("nombre") or "").lower()
    family = str(row.get("initiative_family") or "").lower()
    capex = _to_float_or_zero(row.get("capex_eur"))
    implementation = _to_float_or_zero(row.get("implementation_months"))
    quick_keywords = [
        "gdo", "garantía de origen", "garantia de origen", "led", "aire comprimido",
        "eco-driving", "ecodriving", "mantenimiento", "route optimization", "optimización"
    ]
    if any(k in name for k in quick_keywords):
        return "quick_win"
    if capex <= 50000 and implementation > 0 and implementation <= 3:
        return "quick_win"
    if "supply" in family and capex == 0:
        return "quick_win"
    return "estrategica"


def classify_thematic_bucket(row: pd.Series) -> str:
    text = f"{row.get('initiative_family', '')} {row.get('initiative', '')}".lower()
    if any(k in text for k in ["solar", "fotovolta", "autoconsumo", "pv", "renewable"]):
        return "renovables"
    if any(k in text for k in ["electrification", "electrificación", "heat pump", "bomba de calor", "flota eléctrica", "fleet electr"]):
        return "electrificacion"
    if any(k in text for k in ["led", "motor", "vfd", "efficiency", "hvac", "boiler", "burner"]):
        return "eficiencia"
    if any(k in text for k in ["ems", "submetering", "compressed air", "aire comprimido", "eco-driving", "route", "maintenance"]):
        return "operativa"
    if any(k in text for k in ["gdo", "ppa", "supplier", "supply decarbonization", "electricity supply", "contract"]):
        return "suministro"
    return "otras"


def apply_initiative_business_rules(df: pd.DataFrame, company_inputs: Dict) -> pd.DataFrame:
    df = df.copy()
    annual_electricity_mwh = _to_float_or_zero(company_inputs.get("annual_electricity_mwh"))
    elec_factor = _to_float_or_zero(company_inputs.get("co2_factor_elec_t_per_mwh"))
    electricity_method = (company_inputs.get("electricity_method") or "location").lower()

    for idx in df.index:
        name = str(df.at[idx, "initiative"]).lower()
        category = classify_initiative_category(df.loc[idx])
        df.at[idx, "categoria"] = category
        df.at[idx, "priority_weight"] = 1.0 if category == "estrategica" else 0.4
        df.at[idx, "thematic_bucket"] = classify_thematic_bucket(df.loc[idx])

        if "gdo" in name or "garantía de origen" in name or "garantia de origen" in name:
            df.at[idx, "capex_eur"] = 0.0
            df.at[idx, "categoria"] = "quick_win"
            df.at[idx, "priority_weight"] = 0.4
            df.at[idx, "strategic_score_1_5"] = min(2.0, _to_float_or_zero(df.at[idx, "strategic_score_1_5"]) or 2.0)
            df.at[idx, "implementation_months"] = 1.0 if _to_float_or_zero(df.at[idx, "implementation_months"]) <= 0 else min(2.0, _to_float_or_zero(df.at[idx, "implementation_months"]))
            if pd.isna(df.at[idx, "annual_opex_saving_eur"]):
                df.at[idx, "annual_opex_saving_eur"] = 0.0
            if electricity_method == "market" and annual_electricity_mwh > 0 and elec_factor > 0:
                df.at[idx, "annual_co2_reduction_t"] = annual_electricity_mwh * elec_factor
            else:
                df.at[idx, "annual_co2_reduction_t"] = 0.0

        if ("refrigerant" in name or "refrigerante" in name or "fugas" in name) and pd.isna(df.at[idx, "annual_opex_saving_eur"]):
            df.at[idx, "annual_opex_saving_eur"] = -3000.0

    return df


def finalize_initiatives(
    df: pd.DataFrame,
    company_inputs: Dict,
    n: int = 8,
    fallback_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = df.copy()
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan if c in NUMERIC_COLUMNS else ""

    df = coerce_numeric(df)
    df = apply_initiative_business_rules(df, company_inputs)
    df["nombre"] = df["initiative"].fillna("").astype(str)
    df["tiempo_implementacion"] = df["implementation_months"]
    df["annual_co2_reduction_t"] = pd.to_numeric(df["annual_co2_reduction_t"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["priority_weight"] = pd.to_numeric(df["priority_weight"], errors="coerce").fillna(1.0)
    df["co2_adjusted_t"] = df["annual_co2_reduction_t"] * df["priority_weight"]
    df["confidence_0_1"] = pd.to_numeric(df.get("confidence_0_1", np.nan), errors="coerce").clip(0.0, 1.0)
    df["strategic_score_1_5"] = pd.to_numeric(df["strategic_score_1_5"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(3.0)
    df["capex_eur"] = pd.to_numeric(df["capex_eur"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["implementation_months"] = pd.to_numeric(df["implementation_months"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(3.0)
    df["initiative"] = df["initiative"].fillna("").astype(str)
    df = df[df["initiative"].str.strip() != ""].copy()
    df["_dedupe_key"] = df["initiative"].str.strip().str.lower()
    df = df.drop_duplicates(subset=["_dedupe_key"], keep="first")

    if fallback_df is not None and len(df) < n:
        fb = fallback_df.copy()
        for c in df.columns:
            if c not in fb.columns:
                fb[c] = np.nan if c in NUMERIC_COLUMNS else ""
        fb = coerce_numeric(fb)
        fb = apply_initiative_business_rules(fb, company_inputs)
        fb["nombre"] = fb["initiative"].fillna("").astype(str)
        fb["tiempo_implementacion"] = fb["implementation_months"]
        fb["annual_co2_reduction_t"] = pd.to_numeric(fb["annual_co2_reduction_t"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        fb["priority_weight"] = pd.to_numeric(fb["priority_weight"], errors="coerce").fillna(1.0)
        fb["co2_adjusted_t"] = fb["annual_co2_reduction_t"] * fb["priority_weight"]
        fb["_dedupe_key"] = fb["initiative"].fillna("").astype(str).str.strip().str.lower()
        merged = pd.concat([df, fb], ignore_index=True)
        merged = merged.drop_duplicates(subset=["_dedupe_key"], keep="first")
        df = merged

    confidence = pd.to_numeric(df.get("confidence_0_1", np.nan), errors="coerce").fillna(0.55)
    df["_relevance"] = (
        pd.to_numeric(df["co2_adjusted_t"], errors="coerce").fillna(0.0)
        + 5.0 * (df["categoria"].eq("estrategica").astype(float))
        + 0.8 * pd.to_numeric(df["strategic_score_1_5"], errors="coerce").fillna(3.0)
        + confidence
        - 0.02 * pd.to_numeric(df["implementation_months"], errors="coerce").fillna(0.0)
    )
    df = df.sort_values("_relevance", ascending=False).reset_index(drop=True)

    selected_indices: List[int] = []
    diversity_buckets = ["eficiencia", "electrificacion", "renovables", "operativa", "suministro"]
    for bucket in diversity_buckets:
        bucket_candidates = df[df["thematic_bucket"] == bucket]
        if not bucket_candidates.empty:
            idx = int(bucket_candidates.index[0])
            if idx not in selected_indices:
                selected_indices.append(idx)

    strategic_candidates = [int(i) for i in df[df["categoria"] == "estrategica"].index.tolist() if int(i) not in selected_indices]
    for idx in strategic_candidates:
        if len(selected_indices) >= n:
            break
        selected_indices.append(idx)

    for idx in [int(i) for i in df.index.tolist() if int(i) not in selected_indices]:
        if len(selected_indices) >= n:
            break
        selected_indices.append(idx)

    df = df.loc[selected_indices[:n]].copy().reset_index(drop=True)
    if len(df) > n:
        df = df.head(n).copy()

    if len(df) < n:
        df = df.head(n).copy()

    df["id"] = range(1, len(df) + 1)
    df = df.drop(columns=[c for c in ["_dedupe_key", "_relevance"] if c in df.columns])
    df["notes"] = df["notes"].fillna("").astype(str)
    return df


# -----------------------------
# Initiative proposal engine (normative-aligned)
# -----------------------------
def propose_initiatives(company: Dict, n: int = 8, finalize_output: bool = True) -> pd.DataFrame:
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

    implemented = company.get("implemented_measures") or {}

    def ok_num(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x))

    initiatives: List[dict] = []
    next_id = 1

    def _measure_status(label: str) -> str:
        val = str(implemented.get(label, "No")).strip().lower()
        if val in ["sí", "si", "yes", "implantada", "implantado"]:
            return "yes"
        if val in ["parcial", "partial"]:
            return "partial"
        return "no"

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

    def add_if_allowed(measure_label: str, **kwargs):
        status = _measure_status(measure_label)
        if status == "yes":
            return
        if status == "partial":
            kwargs["initiative"] = f"Escalar: {kwargs.get('initiative', '')}"
            kwargs["notes"] = (kwargs.get("notes", "") + " (Medida parcialmente implantada: enfocar en ampliación.)").strip()
        add_init(**kwargs)

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
    add_if_allowed(
        "GdO",
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
    add_if_allowed(
        "EMS / submetering",
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

    add_if_allowed(
        "LED",
        initiative_family="Electric efficiency",
        initiative="LED lighting retrofit + controls",
        scope="Scope 2",
        emission_source="Purchased electricity (lighting)",
        activity_unit="kWh (meters/invoices)",
        mrv_method="Lighting inventory + kWh baseline vs after (submetering or invoices)",
        capex_eur=60000,
        annual_opex_saving_eur=(0.01 * annual_electricity_mwh * electricity_price) if (ok_num(annual_electricity_mwh) and ok_num(electricity_price)) else np.nan,
        annual_co2_reduction_t=(0.01 * annual_electricity_mwh * (co2_factor_elec or 0.0)) if (ok_num(annual_electricity_mwh) and ok_num(co2_factor_elec)) else np.nan,
        implementation_months=2,
        strategic_score_1_5=3,
        notes="Sustitución de luminarias y control de horarios/presencia.",
        required_info="lighting_inventory;operating_hours;electricity_price_eur_mwh",
        provided_info="",
        data_dependency="Medium",
    )

    # Motors + VFD
    add_if_allowed(
        "Variadores de frecuencia",
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
        add_if_allowed(
            "Programa de fugas de aire comprimido",
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

    add_if_allowed(
        "Paneles solares",
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

        add_if_allowed(
            "Recuperación de calor",
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
            activity_unit="liters fuel (invoices/cards)",
            mrv_method="Fuel cards/invoices + mileage logs; baseline vs after",
            capex_eur=25000,
            annual_opex_saving_eur=np.nan,
            annual_co2_reduction_t=np.nan,
            implementation_months=2,
            strategic_score_1_5=3,
            notes="Medida de bajo CAPEX; necesita datos de litros por tipo de combustible.",
            required_info="fleet_fuel_liters_by_type;fuel_emission_factors;fleet_inventory",
            provided_info="",
            data_dependency="High",
        )

        add_if_allowed(
            "Flota eléctrica",
            initiative_family="Fleet (Scope 1/2)",
            initiative="Fleet electrification (phased replacement of vehicles)",
            scope="Scope 1",
            emission_source="Fleet fuels displacement (Scope 1) + electricity increase (Scope 2)",
            activity_unit="liters fuel avoided + kWh electricity added",
            mrv_method="Fleet inventory + fuel invoices + charging kWh logs",
            capex_eur=300000,
            annual_opex_saving_eur=np.nan,
            annual_co2_reduction_t=np.nan,
            implementation_months=12,
            strategic_score_1_5=4,
            notes="Depende de perfiles de ruta, autonomía y disponibilidad de infraestructura de recarga.",
            required_info="fleet_inventory;fuel_invoices;charging_infrastructure;route_profiles",
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
            required_info="refrigerants;kg_recharged_per_year;equipment_inventory;maintenance_logs",
            provided_info="",
            data_dependency="High",
        )

    df = pd.DataFrame(initiatives).copy()

    # Ensure required/optional columns exist
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    df["notes"] = df["notes"].fillna("").astype(str)
    if finalize_output:
        return finalize_initiatives(df, company, n=n)
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
    numeric_defaults = {
        "capex_eur": 0.0,
        "annual_co2_reduction_t": 0.0,
        "co2_adjusted_t": 0.0,
        "npv_penalized_eur": -1e9,
        "npv_eur": -1e9,
        "strategic_score_1_5": 3.0,
    }
    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(default)

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
        co2_vals = df["co2_adjusted_t"].values.astype(float)
        strat_vals = df["strategic_score_1_5"].values.astype(float)

        def norm(arr: np.ndarray) -> np.ndarray:
            a = np.array(arr, dtype=float)
            a = np.where(np.isfinite(a), a, np.nan)
            lo, hi = np.nanmin(a), np.nanmax(a)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
                return np.zeros_like(a)
            out = (a - lo) / (hi - lo)
            return np.where(np.isfinite(out), out, 0.0)

        df["npv_norm"] = pd.Series(norm(npv_vals)).fillna(0.0)
        df["co2_norm"] = pd.Series(norm(co2_vals)).fillna(0.0)
        df["strat_norm"] = pd.Series(norm(strat_vals)).fillna(0.0)

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


def render_tab_intro(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="mini-card" style="min-height:unset; margin-bottom:1rem;">
            <strong>{title}</strong>
            <p style="margin:0.35rem 0 0;">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_optional_summary(value: Optional[float], unit: str, decimals: int = 1) -> str:
    if value is None:
        return "No definido"
    return f"{value:,.{decimals}f} {unit}".replace(",", "X").replace(".", ",").replace("X", ".")


def _parse_optional_nonnegative_number(raw_value: str, field_label: str) -> Optional[float]:
    text = (raw_value or "").strip()
    if not text:
        return None
    normalized = text.replace(".", "").replace(",", ".")
    try:
        value = float(normalized)
    except ValueError:
        st.warning(f"{field_label}: introduce un número válido.")
        return None
    if value < 0:
        st.warning(f"{field_label}: no se permiten valores negativos.")
        return None
    return value


def ensure_financial_state_defaults() -> None:
    defaults = {
        "horizon_years": 5,
        "discount_rate_pct": 8.0,
        "carbon_price_eur_t": 80.0,
        "capex_budget_eur": 10000.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    widget_defaults = {
        "fin_horizon_years_input": int(st.session_state["horizon_years"]),
        "fin_discount_rate_pct_input": float(st.session_state["discount_rate_pct"]),
        "fin_carbon_price_eur_t_input": float(st.session_state["carbon_price_eur_t"]),
        "fin_capex_budget_eur_input": float(st.session_state["capex_budget_eur"]),
        "sidebar_discount_rate_pct_input": float(st.session_state["discount_rate_pct"]),
        "sidebar_capex_budget_eur_input": float(st.session_state["capex_budget_eur"]),
    }
    for key, value in widget_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _sync_discount_rate_from_sidebar() -> None:
    value = float(st.session_state.get("sidebar_discount_rate_pct_input", 0.0))
    st.session_state["discount_rate_pct"] = value
    st.session_state["fin_discount_rate_pct_input"] = value


def _sync_discount_rate_from_main() -> None:
    value = float(st.session_state.get("fin_discount_rate_pct_input", 0.0))
    st.session_state["discount_rate_pct"] = value
    st.session_state["sidebar_discount_rate_pct_input"] = value


def _sync_capex_from_sidebar() -> None:
    value = float(st.session_state.get("sidebar_capex_budget_eur_input", 0.0))
    st.session_state["capex_budget_eur"] = value
    st.session_state["fin_capex_budget_eur_input"] = value


def _sync_capex_from_main() -> None:
    value = float(st.session_state.get("fin_capex_budget_eur_input", 0.0))
    st.session_state["capex_budget_eur"] = value
    st.session_state["sidebar_capex_budget_eur_input"] = value


def build_financial_assumptions_ui() -> Dict[str, float]:
    st.markdown("**Supuestos financieros**")
    ensure_financial_state_defaults()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        horizon_years = int(
            st.number_input(
                "Horizonte del proyecto (años)",
                min_value=1,
                max_value=20,
                step=1,
                key="fin_horizon_years_input",
                help="Determina durante cuántos años se acumulan ahorros y flujos del portfolio.",
            )
        )
    with col_b:
        discount_rate_pct = float(
            st.number_input(
                "Tasa de descuento (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.25,
                format="%.2f",
                key="fin_discount_rate_pct_input",
                on_change=_sync_discount_rate_from_main,
                help="Representa la rentabilidad mínima o coste de capital exigido por la empresa.",
            )
        )
    with col_c:
        carbon_price_eur_t = float(
            st.number_input(
                "Precio CO2 (€/t)",
                min_value=0.0,
                step=5.0,
                key="fin_carbon_price_eur_t_input",
                help="Se usa para monetizar la reducción anual de emisiones cuando aplique.",
            )
        )

    st.session_state["horizon_years"] = horizon_years
    st.session_state["discount_rate_pct"] = discount_rate_pct
    st.session_state["carbon_price_eur_t"] = carbon_price_eur_t
    return {
        "horizon_years": horizon_years,
        "discount_rate_pct": discount_rate_pct,
        "carbon_price_eur_t": carbon_price_eur_t,
    }


def build_investment_criteria_ui() -> Dict[str, Optional[float]]:
    st.markdown("**Criterios de inversión**")
    ensure_financial_state_defaults()
    if "fin_min_co2_target_tpy_input" not in st.session_state:
        st.session_state["fin_min_co2_target_tpy_input"] = str(st.session_state.get("min_co2_target_tpy_input", ""))
    if "fin_max_payback_years_input" not in st.session_state:
        st.session_state["fin_max_payback_years_input"] = str(st.session_state.get("max_payback_years_input", ""))

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        capex_budget_eur = float(
            st.number_input(
                "Presupuesto CAPEX (€)",
                min_value=0.0,
                step=10000.0,
                key="fin_capex_budget_eur_input",
                on_change=_sync_capex_from_main,
                help="Actúa como restricción presupuestaria del portfolio de medidas.",
            )
        )
    with col_b:
        min_co2_raw = st.text_input(
            "Objetivo mínimo anual de CO2 (t/año) [opcional]",
            key="fin_min_co2_target_tpy_input",
            placeholder="Ej. 50",
            help="Déjalo vacío si no quieres imponer un mínimo anual de reducción en la optimización.",
        )
    with col_c:
        max_payback_raw = st.text_input(
            "Payback máximo aceptable (años) [opcional]",
            key="fin_max_payback_years_input",
            placeholder="Ej. 4",
            help="Se guarda para usarlo como criterio de filtrado o priorización del portfolio.",
        )

    min_co2_target_tpy = _parse_optional_nonnegative_number(
        min_co2_raw, "Objetivo mínimo anual de CO2"
    )
    max_payback_years = _parse_optional_nonnegative_number(
        max_payback_raw, "Payback máximo aceptable"
    )

    st.session_state["capex_budget_eur"] = capex_budget_eur
    st.session_state["min_co2_target_tpy_input"] = min_co2_raw
    st.session_state["max_payback_years_input"] = max_payback_raw
    st.session_state["min_co2_target_tpy"] = min_co2_target_tpy
    st.session_state["max_payback_years"] = max_payback_years

    return {
        "capex_budget_eur": capex_budget_eur,
        "min_co2_target_tpy": min_co2_target_tpy,
        "max_payback_years": max_payback_years,
    }


def build_financial_summary_ui(financials: Dict[str, Optional[float]]) -> None:
    st.markdown("**Resumen financiero**")
    st.markdown(
        f"""
        <div class="mini-card" style="min-height:unset;">
            <p style="margin:0 0 0.35rem;"><strong>Horizonte:</strong> {int(financials["horizon_years"])} años</p>
            <p style="margin:0 0 0.35rem;"><strong>Tasa de descuento:</strong> {financials["discount_rate_pct"]:.2f} %</p>
            <p style="margin:0 0 0.35rem;"><strong>Precio CO2:</strong> {financials["carbon_price_eur_t"]:,.0f} €/t</p>
            <p style="margin:0 0 0.35rem;"><strong>Presupuesto CAPEX:</strong> {financials["capex_budget_eur"]:,.0f} €</p>
            <p style="margin:0 0 0.35rem;"><strong>Objetivo CO2:</strong> {_format_optional_summary(financials["min_co2_target_tpy"], "t/año")}</p>
            <p style="margin:0;"><strong>Payback máximo:</strong> {_format_optional_summary(financials["max_payback_years"], "años")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footprint_results(footprint: Dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="result-hero">
            <div class="result-overline">Resultado de huella de carbono</div>
            <div class="result-total">{footprint['total_t']:.1f} tCO₂e/año</div>
            <div class="result-subtle">
                Alcance 1: {footprint['scope1_t']:.1f} tCO₂e/año ·
                Alcance 2: {footprint['scope2_t']:.1f} tCO₂e/año ·
                Alcance 3: no considerado
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        min(1.0, footprint["total_t"] / 1000.0),
        text=f"Huella total estimada: {footprint['total_t']:.1f} tCO₂e/año",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="result-card">
                <h4>Alcance 1</h4>
                <div class="result-subtle">Emisiones directas generadas por la actividad y los equipos propios.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        s1, s2, s3 = st.columns(3)
        s1.metric("Combustión fija", f"{footprint['scope1_stationary_t']:.1f} t")
        s2.metric("Flota", f"{footprint['scope1_fleet_t']:.1f} t")
        s3.metric("Fugitivas", f"{footprint['scope1_fugitive_t']:.1f} t")

    with col_b:
        st.markdown(
            """
            <div class="result-card">
                <h4>Alcance 2</h4>
                <div class="result-subtle">Emisiones indirectas ligadas a la energía comprada por la organización.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        e1, e2 = st.columns(2)
        e1.metric("Electricidad", f"{footprint['scope2_elec_t']:.1f} t")
        e2.metric("Calor/vapor", f"{footprint['scope2_heat_t']:.1f} t")


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

    company_context = build_ai_company_context(company_inputs)
    use_web_research = gemini_should_use_web_research(company_inputs)

    schema_note = {
        "required_columns": REQUIRED_COLUMNS,
        "optional_columns": OPTIONAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "n_rows": n,
        "business_output_fields": [
            "nombre",
            "categoria",
            "capex_eur",
            "annual_opex_saving_eur",
            "annual_co2_reduction_t",
            "co2_adjusted_t",
            "implementation_months",
        ],
    }

    system_prompt = (
        "Eres un consultor senior de descarbonización industrial (operaciones + finanzas). "
        "Devuelve SOLO JSON válido, sin texto adicional. "
        "No inventes cifras si no hay datos: usa null. "
        "Responde en español. "
        "Incluye campos normativos (scope, emission_source, activity_unit, mrv_method, normative_reference) "
        "y data gaps (required_info, provided_info, data_dependency). "
        "Las iniciativas deben estar adaptadas al sector, localización y realidad operativa de la empresa; evita propuestas genéricas. "
        "Cuando estimes CAPEX, ahorro OPEX, reducción CO2 e implementación, usa lógica dimensional y de orden de magnitud realista para ese tipo de empresa. "
        "Ten en cuenta tamaño energético, consumos, combustibles, flota, cubierta disponible, calor comprado, calor de proceso si se deduce, restricciones de implantación y presupuesto. "
        "Si la base cuantitativa es insuficiente, reduce la confianza y usa null antes que inventar."
    )

    user_prompt = (
        "Genera exactamente N iniciativas en JSON (lista de objetos) siguiendo este esquema. "
        "Usa los inputs de la empresa, el contexto estructurado y supuestos conservadores. "
        "Cada iniciativa debe incluir TODAS las columnas requeridas y opcionales. "
        "Para 'scope' usa 'Alcance 1' o 'Alcance 2'. "
        "No propongas iniciativas ya implantadas (salvo si se pide explícitamente ampliación cuando son parciales). "
        "Si hay nombre de empresa y la herramienta permite búsqueda web, úsala para entender mejor actividad, productos/servicios, procesos o cadena de suministro, "
        "pero no inventes detalles no verificables ni cifras específicas.\n"
        "Reglas de rigor:\n"
        "- Debe haber exactamente 8 iniciativas finales.\n"
        "- Clasifica cada iniciativa como quick_win o estrategica.\n"
        "- Favorece estrategica frente a quick_win: usa peso 1.0 para estrategica y 0.4 para quick_win.\n"
        "- Calcula CO2 ajustado como reduccion_CO2 * peso.\n"
        "- Asegura diversidad entre eficiencia energética, electrificación, renovables/autoconsumo, mejoras operativas y suministro energético.\n"
        "- Las cifras deben ser plausibles para la escala de la empresa y consistentes entre sí.\n"
        "- El OPEX puede ser positivo, cero o negativo según el caso; no supongas siempre ahorro.\n"
        "- El ahorro OPEX debe derivar de consumos/energía/precios o quedar en null si no hay base.\n"
        "- La reducción de CO2 debe guardar relación con los consumos y factores de emisión disponibles.\n"
        "- Los meses de implementación deben reflejar complejidad real: quick wins, proyectos de ingeniería, permisos, obra e integración.\n"
        "- Usa confidence_0_1 para reflejar cuánta base real hay detrás de cada estimación.\n"
        "- En notes explica brevemente el driver principal del CAPEX/OPEX/CO2 estimado.\n\n"
        f"N = {n}\n"
        f"SCHEMA: {json.dumps(schema_note, ensure_ascii=False)}\n"
        f"COMPANY_CONTEXT: {json.dumps(company_context, ensure_ascii=False)}\n"
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
        if use_web_research:
            payload["tools"] = [{"google_search": {}}]
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

    implemented = company_inputs.get("implemented_measures") or {}
    def _status(label: str) -> str:
        val = str(implemented.get(label, "No")).strip().lower()
        if val in ["sí", "si", "yes"]:
            return "yes"
        if val in ["parcial", "partial"]:
            return "partial"
        return "no"

    keyword_map = {
        "LED": ["led", "iluminación led"],
        "GdO": ["gdo", "garantía de origen"],
        "Paneles solares": ["solar", "fotovoltaica", "pv"],
        "Flota eléctrica": ["flota eléctrica", "electrificación", "vehículo eléctrico", "vehicle electr"],
        "Variadores de frecuencia": ["variador", "vfd"],
        "EMS / submetering": ["ems", "submetering", "gestión energética"],
        "Recuperación de calor": ["recuperación de calor", "heat recovery"],
        "Programa de fugas de aire comprimido": ["aire comprimido", "fugas"],
    }

    if "initiative" in df.columns:
        updated_rows = []
        for _, row in df.iterrows():
            name = str(row.get("initiative", "")).lower()
            blocked = False
            for label, kws in keyword_map.items():
                status = _status(label)
                if status == "yes" and any(k in name for k in kws):
                    blocked = True
                    break
                if status == "partial" and any(k in name for k in kws):
                    row["initiative"] = f"Escalar: {row.get('initiative')}"
                    row["notes"] = (str(row.get("notes", "")) + " (Medida parcialmente implantada: enfoque ampliación.)").strip()
            if not blocked:
                updated_rows.append(row)
        df = pd.DataFrame(updated_rows).reset_index(drop=True)

    fallback_df = propose_initiatives(company_inputs, n=max(12, n), finalize_output=False)
    return finalize_initiatives(df, company_inputs, n=n, fallback_df=fallback_df)

# -----------------------------
# UI
# -----------------------------
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"

if st.session_state["current_page"] == "home":
    render_home_page()
    st.stop()

def render_tool_page() -> None:
    render_global_styles()
    ensure_financial_state_defaults()
    if st.session_state.get("scroll_to_top"):
        st.markdown(
            """
            <script>
            window.scrollTo(0, 0);
            document.documentElement.scrollTop = 0;
            document.body.scrollTop = 0;
            setTimeout(() => {
                window.scrollTo(0, 0);
                document.documentElement.scrollTop = 0;
                document.body.scrollTop = 0;
                const main = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                if (main) { main.scrollTop = 0; }
            }, 50);
            </script>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["scroll_to_top"] = False
    st.title("Herramienta de Descarbonización Industrial")
    
    with st.expander("Cómo funciona la herramienta", expanded=True):
        st.markdown(
            "1. Completa el cuestionario inicial (empresa + Alcance 1 + Alcance 2).\n"
            "2. La app estima tu huella de carbono anual.\n"
            "3. Puedes generar un PESTEL con IA para contexto estratégico.\n"
            "4. Se generan iniciativas de descarbonización y puedes editarlas.\n"
            "5. Se optimiza la cartera según presupuesto, CO₂ y NPV."
        )
        st.caption(
            "GHG Protocol: Alcance 1 (emisiones directas), Alcance 2 (electricidad/calor comprados), "
            "Alcance 3 (cadena de valor, explicado pero no calculado automáticamente aquí)."
        )
    
    with st.sidebar:
        if st.button("Volver a inicio", use_container_width=True):
            st.session_state["current_page"] = "home"
            st.rerun()

        st.divider()
        st.markdown("**Ajustes financieros rápidos**")
        st.number_input(
            "Tasa de descuento (%)",
            min_value=0.0,
            max_value=100.0,
            step=0.25,
            format="%.2f",
            key="sidebar_discount_rate_pct_input",
            on_change=_sync_discount_rate_from_sidebar,
        )
        st.number_input(
            "Presupuesto CAPEX (€)",
            min_value=0.0,
            step=10000.0,
            key="sidebar_capex_budget_eur_input",
            on_change=_sync_capex_from_sidebar,
        )
    
        st.divider()
        st.markdown("**TFG**")
        st.caption("Carlos Falcó Caravajal")
        st.caption("IGE EDEM")
        st.caption("Enterprise Risk Deloitte")
        mode = "A"
    
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
        ai_model = "gemini-3.1-flash-lite-preview"
    
    
    # -----------------------------
    # Data source selection
    # -----------------------------
    df_base = None
    company_inputs: Dict = {}
    pestel: Dict[str, List[str]] = {}
    
    if mode == "A":
        with render_phase_block(
            "INPUTS",
            "inputs",
            "Completa la información base de la empresa, las fuentes de emisión y los supuestos financieros antes de lanzar el diagnóstico.",
        ):
            tab_company, tab_scope1, tab_scope2, tab_finance = st.tabs(
                ["Sobre la empresa", "Alcance 1", "Alcance 2", "Supuestos financieros"]
            )

            with tab_company:
                render_tab_intro(
                    "Sobre la empresa",
                    "Introduce los datos base de la organización y marca las medidas ya implantadas para personalizar el diagnóstico.",
                )
                st.markdown("**Datos base de la organización**")
                row_1a, row_1b, row_1c = st.columns([1.4, 1.1, 0.8])
                with row_1a:
                    company_inputs["company_name"] = st.text_input("Nombre de la organización (opcional)", value="")
                with row_1b:
                    company_inputs["cnae_sector"] = st.text_input("CNAE / Sector", value="")
                with row_1c:
                    company_inputs["inventory_year"] = st.number_input("Año de inventario (cálculo)", min_value=2000, max_value=2100, value=DEFAULT_INVENTORY_YEAR, step=1)
                    st.session_state["inventory_year"] = int(company_inputs["inventory_year"])

                company_inputs["country"] = "España"
                row_2a, row_2b, row_2c = st.columns([0.9, 1.1, 0.8])
                with row_2a:
                    st.text_input("País", value="España", disabled=True)
                with row_2b:
                    company_inputs["province"] = st.selectbox("Provincia", SPAIN_PROVINCES, index=48)
                with row_2c:
                    company_inputs["postal_code"] = st.text_input("Código postal", value="", max_chars=5)
                company_inputs["country_region"] = f"España - {company_inputs['province']}"
                company_inputs["sector"] = company_inputs["cnae_sector"]
                company_inputs["has_invoices"] = True
                company_inputs["has_meters"] = False
                company_inputs["has_submetering"] = False
                company_inputs["fuel_price_eur_mwh"] = 0.0
                company_inputs["electricity_price_eur_mwh"] = 0.0
                company_inputs["has_energy_audit"] = False

                st.markdown("**Medidas ya implantadas**")
                st.caption("Marca solo las soluciones que ya estén implantadas o funcionando en la empresa para evitar propuestas redundantes.")
                implemented_catalog = [
                    ("LED", "Iluminación LED instalada en naves, oficinas o zonas auxiliares para reducir consumo eléctrico en alumbrado."),
                    ("GdO", "Contrato eléctrico con Garantías de Origen para acreditar suministro renovable en la electricidad comprada."),
                    ("Paneles solares", "Instalación fotovoltaica de autoconsumo ya operativa, en cubierta o suelo, que reduce compra de red."),
                    ("Flota eléctrica", "Vehículos eléctricos o híbridos enchufables incorporados en la flota propia de la empresa."),
                    ("Variadores de frecuencia", "Variadores en motores, bombas o ventiladores para adaptar la velocidad y evitar consumos innecesarios."),
                    ("EMS / submetering", "Sistema de gestión energética o submedición que permite seguir consumos por línea, área o equipo."),
                    ("Recuperación de calor", "Aprovechamiento de calor residual de procesos o equipos para precalentar, climatizar o cubrir otras demandas."),
                    ("Programa de fugas de aire comprimido", "Rutina activa de detección y corrección de fugas en la red de aire comprimido."),
                ]
                implemented_state = company_inputs.get("implemented_measures") or {}
                company_inputs["implemented_measures"] = {}
                measure_cols = st.columns(2)
                for idx, (label, description) in enumerate(implemented_catalog):
                    with measure_cols[idx % 2]:
                        checked = st.checkbox(
                            label,
                            value=str(implemented_state.get(label, "No")).strip().lower() in ["sí", "si", "yes"],
                            key=f"implemented_measure_{label}",
                        )
                        st.caption(description)
                        company_inputs["implemented_measures"][label] = "Sí" if checked else "No"

                company_inputs["roof_area_m2"] = st.number_input(
                    "Área disponible de cubierta (m²) [opcional]",
                    min_value=0.0,
                    value=float(st.session_state.get("roof_area_m2", 0.0)),
                    step=100.0,
                )
                st.session_state["roof_area_m2"] = company_inputs["roof_area_m2"]

            with tab_scope1:
                render_tab_intro(
                    "Alcance 1",
                    "Registra consumos de combustión fija, flota propia y gases refrigerantes con sus unidades oficiales para estimar las emisiones directas.",
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Combustión fija**")
                    st.caption("Introduce el consumo anual de los combustibles utilizados en instalaciones fijas con la unidad oficial indicada.")
                    stationary_entries = []

                    common_cols = st.columns(2)
                    for idx, fuel in enumerate(COMMON_STATIONARY_FUELS):
                        with common_cols[idx % 2]:
                            qty = st.number_input(
                                f"{fuel['label']} ({fuel['unit']})",
                                min_value=0.0,
                                value=0.0,
                                step=100.0,
                                key=f"stationary_common_{fuel['key']}",
                            )
                            factor_kg, factor_year = get_stationary_fuel_factor(fuel["key"], int(company_inputs.get("inventory_year") or DEFAULT_INVENTORY_YEAR))
                            st.caption(f"Factor {factor_year}: {factor_kg:.3f} kgCO2/{fuel['unit']}")
                            if qty > 0:
                                stationary_entries.append({"fuel_key": fuel["key"], "quantity": qty})

                    other_fuels = [f for f in STATIONARY_FUELS_CATALOG if not f.get("common")]
                    selected_other_fuels = st.multiselect(
                        "Otros combustibles: buscar y seleccionar",
                        options=[f["key"] for f in other_fuels],
                        default=[],
                        format_func=lambda key: f"{STATIONARY_FUELS_BY_KEY[key]['label']} ({STATIONARY_FUELS_BY_KEY[key]['unit']})",
                    )
                    for fuel_key in selected_other_fuels:
                        fuel = STATIONARY_FUELS_BY_KEY[fuel_key]
                        qty = st.number_input(
                            f"{fuel['label']} ({fuel['unit']})",
                            min_value=0.0,
                            value=0.0,
                            step=100.0,
                            key=f"stationary_other_{fuel_key}",
                        )
                        factor_kg, factor_year = get_stationary_fuel_factor(fuel_key, int(company_inputs.get("inventory_year") or DEFAULT_INVENTORY_YEAR))
                        st.caption(f"Factor {factor_year}: {factor_kg:.3f} kgCO2/{fuel['unit']}")
                        if qty > 0:
                            stationary_entries.append({"fuel_key": fuel_key, "quantity": qty})

                    company_inputs["stationary_fuels"] = stationary_entries
                    company_inputs["fuel_type"] = ", ".join(
                        STATIONARY_FUELS_BY_KEY[row["fuel_key"]]["label"] for row in stationary_entries
                    )
                with c2:
                    st.markdown("**Combustión móvil (flota)**")
                    st.caption("Selecciona combustible y tipo de vehículo, e introduce el consumo anual con su unidad oficial.")
                    mobile_entries = []
                    common_mobile_labels = []
                    for fuel in COMMON_MOBILE_FUELS:
                        if fuel["fuel_label"] not in common_mobile_labels:
                            common_mobile_labels.append(fuel["fuel_label"])

                    for fuel_label in common_mobile_labels:
                        options = [f for f in COMMON_MOBILE_FUELS if f["fuel_label"] == fuel_label]
                        selected_vehicles = st.multiselect(
                            f"Tipo de vehículo para {fuel_label}",
                            options=[f["key"] for f in options],
                            default=[],
                            format_func=lambda key: MOBILE_FUELS_BY_KEY[key]["vehicle_type"],
                            key=f"mobile_common_vehicle_{fuel_label}",
                        )
                        for vehicle_key in selected_vehicles:
                            selected = MOBILE_FUELS_BY_KEY[vehicle_key]
                            qty = st.number_input(
                                f"{fuel_label} · {selected['vehicle_type']} ({selected['unit']})",
                                min_value=0.0,
                                value=0.0,
                                step=100.0,
                                key=f"mobile_common_qty_{vehicle_key}",
                            )
                            factor_kg, factor_year = get_mobile_fuel_factor(vehicle_key, int(company_inputs.get("inventory_year") or DEFAULT_INVENTORY_YEAR))
                            st.caption(f"Factor {factor_year}: {factor_kg:.3f} kgCO2/{selected['unit']}")
                            if qty > 0:
                                mobile_entries.append({"fuel_key": vehicle_key, "quantity": qty})

                    other_mobile_keys = [f["key"] for f in MOBILE_FUELS_CATALOG if f["fuel_label"] not in common_mobile_labels]
                    selected_other_mobile = st.multiselect(
                        "Otros combustibles móviles",
                        options=other_mobile_keys,
                        default=[],
                        format_func=lambda key: f"{MOBILE_FUELS_BY_KEY[key]['fuel_label']} · {MOBILE_FUELS_BY_KEY[key]['vehicle_type']} ({MOBILE_FUELS_BY_KEY[key]['unit']})",
                    )
                    for fuel_key in selected_other_mobile:
                        fuel = MOBILE_FUELS_BY_KEY[fuel_key]
                        qty = st.number_input(
                            f"{fuel['fuel_label']} · {fuel['vehicle_type']} ({fuel['unit']})",
                            min_value=0.0,
                            value=0.0,
                            step=100.0,
                            key=f"mobile_other_qty_{fuel_key}",
                        )
                        factor_kg, factor_year = get_mobile_fuel_factor(fuel_key, int(company_inputs.get("inventory_year") or DEFAULT_INVENTORY_YEAR))
                        st.caption(f"Factor {factor_year}: {factor_kg:.3f} kgCO2/{fuel['unit']}")
                        if qty > 0:
                            mobile_entries.append({"fuel_key": fuel_key, "quantity": qty})

                    company_inputs["mobile_fuels"] = mobile_entries

                    st.markdown("**Emisiones fugitivas**")
                    selected_refrigerants = st.multiselect(
                        "Nombre del gas refrigerante",
                        options=[r["name"] for r in REFRIGERANTS_CATALOG],
                        default=[],
                    )
                    refrigerant_entries = []
                    for name in selected_refrigerants:
                        qty = st.number_input(
                            f"{name} (kg recargados/año)",
                            min_value=0.0,
                            value=0.0,
                            step=10.0,
                            key=f"refrigerant_qty_{name}",
                        )
                        if qty > 0:
                            refrigerant_entries.append({"name": name, "quantity": qty})
                    company_inputs["refrigerants"] = refrigerant_entries

            with tab_scope2:
                render_tab_intro(
                    "Alcance 2",
                    "Indica la electricidad comprada y, si aplica, el calor o vapor comprado para calcular las emisiones indirectas por energía.",
                )
                build_scope2_ui(company_inputs)

            with tab_finance:
                render_tab_intro(
                    "Supuestos financieros",
                    "Define horizonte, presupuesto y criterios de priorización para valorar las iniciativas con una lógica financiera coherente.",
                )
                finance_left, finance_right = st.columns([1.7, 1.0])
                with finance_left:
                    financial_assumptions = build_financial_assumptions_ui()
                    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
                    investment_criteria = build_investment_criteria_ui()
                with finance_right:
                    build_financial_summary_ui({**financial_assumptions, **investment_criteria})

        horizon_years = int(financial_assumptions["horizon_years"])
        discount_rate_pct = float(financial_assumptions["discount_rate_pct"])
        co2_price = float(financial_assumptions["carbon_price_eur_t"])
        budget_eur = float(investment_criteria["capex_budget_eur"])
        min_co2_t = float(investment_criteria["min_co2_target_tpy"] or 0.0)
        company_inputs["horizon_years"] = horizon_years
        company_inputs["discount_rate"] = discount_rate_pct / 100.0
        company_inputs["carbon_price_eur_t"] = co2_price
        company_inputs["capex_budget_eur"] = budget_eur
        company_inputs["min_co2_target_tpy"] = investment_criteria["min_co2_target_tpy"]
        company_inputs["max_payback_years"] = investment_criteria["max_payback_years"]
        discount_rate = discount_rate_pct / 100.0
        confidence_floor = 1.0
        w_npv, w_co2 = 0.30, 0.70
        w_strategy = 0.0
        objective = "Balanced score (NPV + CO2 + strategy)"

        footprint = calculate_company_footprint(company_inputs)
        with render_phase_block(
            "CÁLCULO DE HUELLA DE CARBONO",
            "carbon",
            "Esta fase agrupa el cálculo de emisiones de Alcance 1 y 2 y la visualización consolidada de resultados.",
        ):
            render_footprint_results(footprint)
            if get_refrigerant_entries(company_inputs) and not footprint["refrigerant_factor_found"]:
                st.warning(
                    f"No se encontró GWP para '{footprint['refrigerant_key']}'. "
                    "La parte de fugitivas puede estar infraestimada."
                )
            elif _to_float_or_zero(footprint.get("refrigerant_gwp")) >= 2000:
                st.warning("Refrigerante con GWP alto. Considera plan de sustitución y control de fugas.")
            if _to_float_or_zero(company_inputs.get("annual_electricity_mwh")) > 0 and _to_float_or_zero(footprint.get("used_elec_factor")) == 0:
                st.warning("No hay factor eléctrico válido disponible; revisa el método y los factores.")

        # Normalizar tipo de GdO
        if company_inputs.get("electricity_gdo_type") == "Ninguno/Desconocido":
            company_inputs["electricity_gdo_type"] = None
        elif company_inputs.get("electricity_gdo_type") == "Renovable":
            company_inputs["electricity_gdo_type"] = "renewable"
        elif company_inputs.get("electricity_gdo_type") == "Cogeneración":
            company_inputs["electricity_gdo_type"] = "cogen"
    
        # Derivar agregados para el motor de iniciativas
        stationary_fuel_mwh = estimate_stationary_fuel_mwh(company_inputs)
        company_inputs["annual_fuel_mwh"] = stationary_fuel_mwh if stationary_fuel_mwh > 0 else 0.0
        company_inputs["co2_factor_elec_t_per_mwh"] = footprint.get("used_elec_factor", 0.0)
        if stationary_fuel_mwh > 0:
            company_inputs["co2_factor_fuel_t_per_mwh"] = footprint.get("scope1_stationary_t", 0.0) / stationary_fuel_mwh
        else:
            company_inputs["co2_factor_fuel_t_per_mwh"] = None
    
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
    
        fleet_rows = company_inputs.get("mobile_fuels") or []
        company_inputs["has_fleet"] = any(_to_float_or_zero(r.get("quantity")) > 0 for r in fleet_rows)
        company_inputs["has_refrigerants"] = len(get_refrigerant_entries(company_inputs)) > 0
    
        # Convertir ceros a None (desconocido)
        for k in [
            "annual_electricity_mwh", "annual_fuel_mwh", "electricity_price_eur_mwh", "fuel_price_eur_mwh",
            "co2_factor_elec_t_per_mwh", "co2_factor_fuel_t_per_mwh", "roof_area_m2"
        ]:
            if isinstance(company_inputs.get(k), (int, float)) and company_inputs.get(k) == 0.0:
                company_inputs[k] = None

        if "ai_pestel" not in st.session_state:
            st.session_state["ai_pestel"] = None
        if "ai_pestel_key" not in st.session_state:
            st.session_state["ai_pestel_key"] = None

        pestel_key = json.dumps(company_inputs, sort_keys=True, ensure_ascii=False)
        if st.session_state["ai_pestel_key"] != pestel_key:
            st.session_state["ai_pestel_key"] = pestel_key
            st.session_state["ai_pestel"] = None

        n_inits = 8

        if "ai_initiatives" not in st.session_state:
            st.session_state["ai_initiatives"] = None

        with render_phase_block(
            "ANÁLISIS PESTEL",
            "pestel",
            "Genera contexto estratégico para interpretar la presión regulatoria, tecnológica y de mercado antes de priorizar medidas.",
        ):
            st.markdown(
                """
                <div class="phase-callout">
                    <strong>Contexto PESTEL</strong>
                    <p>Genera una lectura estratégica breve para entender presión regulatoria, mercado, tecnología y condicionantes externos.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            gen_pestel = render_phase_action_button(
                "Generar PESTEL con IA",
                "pestel",
                type="primary",
                key="phase_generate_pestel",
            )

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
            st.markdown("### PESTEL")
            if pestel is None:
                st.info("PESTEL no generado. Puedes continuar con las iniciativas si quieres.")
            if pestel:
                with st.expander("Análisis de factores externos que influyen en el entorno de la empresa.", expanded=True):
                    pcols = st.columns(3)
                    keys = list(pestel.keys())
                    for i, k in enumerate(keys):
                        with pcols[i % 3]:
                            st.subheader(k)
                            for bullet in pestel[k]:
                                st.write(f"- {bullet}")

        with render_phase_block(
            "INICIATIVAS Y PORTFOLIO",
            "portfolio",
            "Esta fase concentra la generación inicial de iniciativas antes de pasar a su evaluación, optimización y exportación.",
        ):
            st.markdown(
                """
                <div class="phase-callout">
                    <strong>Iniciativas</strong>
                    <p>Crea una cartera inicial de medidas de descarbonización usando la huella y los datos operativos introducidos.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            generate_ai = render_phase_action_button(
                "Generar 8 iniciativas con IA",
                "portfolio",
                key="phase_generate_initiatives",
            )

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

            st.markdown("### Iniciativas propuestas")
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
    
    with render_phase_block(
        "INICIATIVAS Y PORTFOLIO",
        "portfolio",
        "Aquí se evalúan las iniciativas, se optimiza el portfolio objetivo y se presentan los resultados finales para su análisis y exportación.",
    ):
        st.markdown("### Evaluación de iniciativas")
        cols_to_show = [
            "id",
            "nombre",
            "initiative_family",
            "initiative",
            "categoria",
            "scope",
            "emission_source",
            "activity_unit",
            "mrv_method",
            "data_dependency",
            "capex_eur",
            "annual_opex_saving_eur",
            "annual_co2_reduction_t",
            "co2_adjusted_t",
            "co2_value_eur_per_year",
            "total_annual_benefit_eur",
            "npv_eur",
            "npv_penalized_eur",
            "tiempo_implementacion",
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
        
        st.markdown("### Optimización del portafolio")
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
        
        st.markdown("### Visuales")
        chart_df = df_opt.copy()
        chart_df["selected_label"] = np.where(chart_df["selected"], "Seleccionada", "No seleccionada")
        
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
        
        st.markdown("### Copiloto IA")
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
                        company_inputs=company_inputs,
                        assumptions=assumptions,
                        constraints=constraints,
                        summary=summary,
                        df_all=df_opt,
                        df_selected=selected_df,
                        pestel=pestel,
                        extra_prompt=ai_extra_prompt.strip(),
                    )
                st.success("Informe IA generado.")
                st.markdown(ai_text if ai_text else "_El modelo no devolvió contenido._")
            except Exception as e:
                st.error(f"Fallo en generación IA: {e}")
        
        st.markdown("### Exportar resultados")
        export_cols = [
            "id",
            "nombre",
            "initiative_family",
            "initiative",
            "categoria",
            "scope",
            "emission_source",
            "activity_unit",
            "mrv_method",
            "capex_eur",
            "annual_opex_saving_eur",
            "annual_co2_reduction_t",
            "co2_adjusted_t",
            "co2_value_eur_per_year",
            "tiempo_implementacion",
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

render_tool_page()
