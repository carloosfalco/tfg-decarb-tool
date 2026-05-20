from __future__ import annotations

import json
import math
import re
import unicodedata
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIONARY_FUELS_DB_PATH = DATA_DIR / "stationary_fuel_factors_es.csv"
MOBILE_FUELS_DB_PATH = DATA_DIR / "mobile_fuel_factors_es.csv"
REFRIGERANTS_DB_PATH = DATA_DIR / "refrigerants_pca_es.csv"
ELECTRICITY_SUPPLIER_FACTORS_DB_PATH = DATA_DIR / "electricity_supplier_factors_es.csv"
ELECTRICITY_GRID_FACTORS_DB_PATH = DATA_DIR / "electricity_grid_factors_es.csv"

DEFAULT_INVENTORY_YEAR = 2025
GENERIC_MARKET_SUPPLIER_NAME = "Factor market-based genérico (comercializadora no identificada)"

REQUIRED_COLUMNS = [
    "id",
    "scope",
    "emission_source",
    "initiative_family",
    "initiative",
    "capex_eur",
    "annual_opex_saving_eur",
    "annual_co2_reduction_t",
    "implementation_months",
    "strategic_score_1_5",
]

OPTIONAL_COLUMNS = [
    "activity_unit",
    "categoria",
    "priority_weight",
    "co2_adjusted_t",
    "nombre",
    "tiempo_implementacion",
    "thematic_bucket",
]

NUMERIC_COLUMNS = [
    "capex_eur",
    "annual_opex_saving_eur",
    "annual_co2_reduction_t",
    "implementation_months",
    "strategic_score_1_5",
    "priority_weight",
    "co2_adjusted_t",
]

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
    "Biomasa hueso aceituna": 0.005,
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

MEASURE_LABELS = [
    "LED",
    "GdO",
    "Paneles solares",
    "Flota eléctrica",
    "Variadores de frecuencia",
    "EMS/submetering",
    "Recuperación de calor",
    "Programa de fugas de aire comprimido",
]


def _normalize_key(text: str) -> str:
    return (
        unicodedata.normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .replace(".", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _repair_mojibake(text: str) -> str:
    value = str(text or "")
    if not any(marker in value for marker in ("\u00c3", "\u00c2", "\u00e2")):
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", _repair_mojibake(str(name or ""))).encode("ascii", "ignore").decode("ascii")
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_SUPPLIER_STOPWORDS = {
    "S",
    "A",
    "U",
    "SA",
    "SAU",
    "SL",
    "SLU",
    "SOCIEDAD",
    "ANONIMA",
    "LIMITADA",
    "UNIPERSONAL",
    "CLIENTES",
    "CLIENTE",
    "ENERGIA",
    "ENERGY",
    "ELECTRICIDAD",
    "ELECTRICA",
    "ELECTRICO",
}


def simplify_supplier_name(name: str) -> str:
    normalized = normalize_name(name)
    tokens = [token for token in normalized.split() if token not in _SUPPLIER_STOPWORDS]
    return " ".join(tokens) if tokens else normalized


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _ok_num(value: Any) -> bool:
    try:
        return value is not None and np.isfinite(float(value))
    except Exception:
        return False


def latest_factor(factors_by_year: Dict[int, float], year: int) -> Tuple[float, int]:
    if year in factors_by_year:
        return float(factors_by_year[year]), year
    fallback_year = max(factors_by_year.keys())
    return float(factors_by_year[fallback_year]), fallback_year


def _year_factor_map(row: pd.Series) -> Dict[int, float]:
    factors: Dict[int, float] = {}
    for col in row.index:
        if str(col).isdigit() and not pd.isna(row[col]):
            factors[int(col)] = float(row[col])
    return factors


@lru_cache(maxsize=1)
def load_stationary_fuels_catalog() -> List[Dict[str, Any]]:
    df = pd.read_csv(STATIONARY_FUELS_DB_PATH)
    fuels: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        label = str(row["Combustible"]).strip()
        fuels.append(
            {
                "key": _normalize_key(label),
                "label": label,
                "unit": str(row["Unidad"]).strip(),
                "mwh_per_unit": STATIONARY_FUEL_MWH_PER_UNIT.get(label),
                "factors_kg_per_unit": _year_factor_map(row),
            }
        )
    return fuels


@lru_cache(maxsize=1)
def load_mobile_fuels_catalog() -> List[Dict[str, Any]]:
    df = pd.read_csv(MOBILE_FUELS_DB_PATH)
    fuels: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        fuel_label = str(row["Combustible"]).strip()
        vehicle_type = str(row["Tipo"]).strip()
        fuels.append(
            {
                "key": _normalize_key(f"{fuel_label}_{vehicle_type}"),
                "fuel_label": fuel_label,
                "vehicle_type": vehicle_type,
                "unit": str(row["Unidad"]).strip(),
                "mwh_per_unit": MOBILE_FUEL_MWH_PER_UNIT.get(fuel_label),
                "factors_kg_per_unit": _year_factor_map(row),
            }
        )
    return fuels


@lru_cache(maxsize=1)
def load_refrigerants_catalog() -> List[Dict[str, Any]]:
    df = pd.read_csv(REFRIGERANTS_DB_PATH)
    return [{"name": str(row["Nombre"]).strip(), "gwp": float(row["PCA_6AR"])} for _, row in df.iterrows()]


@lru_cache(maxsize=1)
def stationary_fuels_by_key() -> Dict[str, Dict[str, Any]]:
    return {item["key"]: item for item in load_stationary_fuels_catalog()}


@lru_cache(maxsize=1)
def mobile_fuels_by_key() -> Dict[str, Dict[str, Any]]:
    return {item["key"]: item for item in load_mobile_fuels_catalog()}


@lru_cache(maxsize=1)
def refrigerants_by_name() -> Dict[str, Dict[str, Any]]:
    return {item["name"].upper(): item for item in load_refrigerants_catalog()}


def _load_supplier_factor_db() -> pd.DataFrame:
    if not ELECTRICITY_SUPPLIER_FACTORS_DB_PATH.exists():
        return pd.DataFrame(columns=["year", "supplier_name", "supplier_name_norm", "factor_kg_co2e_kwh"])
    df = pd.read_csv(ELECTRICITY_SUPPLIER_FACTORS_DB_PATH)
    if df.empty:
        return pd.DataFrame(columns=["year", "supplier_name", "supplier_name_norm", "factor_kg_co2e_kwh"])
    out = pd.DataFrame(
        {
            "year": df["year"].astype(int),
            "supplier_name": df["supplier_name"].astype(str),
            "supplier_name_norm": df["supplier_name"].map(normalize_name),
            "factor_kg_co2e_kwh": df["factor_kg_kwh"].astype(float),
        }
    )
    return out.drop_duplicates(subset=["year", "supplier_name_norm"], keep="first")


@lru_cache(maxsize=1)
def load_supplier_factors() -> pd.DataFrame:
    return _load_supplier_factor_db().reset_index(drop=True)


@lru_cache(maxsize=1)
def load_grid_factors() -> Dict[int, float]:
    df = pd.read_csv(ELECTRICITY_GRID_FACTORS_DB_PATH)
    required_cols = {"year", "factor_name", "factor_kg_kwh"}
    if not required_cols.issubset(df.columns):
        raise ValueError("La base de datos de factores electricos no tiene el formato esperado.")
    preferred_rows = df[df["factor_name"] == "ree_generation_location_es"]
    location_rows = preferred_rows if not preferred_rows.empty else df[df["factor_name"] == "mix_location_es"]
    if location_rows.empty:
        raise ValueError("La base de datos de factores electricos no contiene factores location-based.")
    return {int(row["year"]): float(row["factor_kg_kwh"]) for _, row in location_rows.iterrows()}


def get_supplier_factor(year: int, supplier_name: str) -> Tuple[Optional[float], Optional[str]]:
    df = load_supplier_factors()
    norm = normalize_name(supplier_name)
    if norm == normalize_name(GENERIC_MARKET_SUPPLIER_NAME):
        year_df = df[df["year"] == year]
        if year_df.empty:
            return None, None
        factor = float(year_df["factor_kg_co2e_kwh"].mean())
        return factor, GENERIC_MARKET_SUPPLIER_NAME
    match = df[(df["year"] == year) & (df["supplier_name_norm"] == norm)]
    if match.empty:
        year_df = df[df["year"] == year].copy()
        simple_norm = simplify_supplier_name(supplier_name)
        if simple_norm:
            year_df["supplier_name_simple"] = year_df["supplier_name"].map(simplify_supplier_name)
            simple_match = year_df[year_df["supplier_name_simple"] == simple_norm]
            if simple_match.empty:
                candidates = year_df["supplier_name_simple"].dropna().tolist()
                close = get_close_matches(simple_norm, candidates, n=1, cutoff=0.88)
                if close:
                    simple_match = year_df[year_df["supplier_name_simple"] == close[0]]
            if not simple_match.empty:
                match = simple_match
    if match.empty:
        return None, None
    row = match.iloc[0]
    return float(row["factor_kg_co2e_kwh"]), str(row["supplier_name"])


def get_location_factor(year: int) -> float:
    factors = load_grid_factors()
    if year in factors:
        return float(factors[year])
    return float(factors[max(factors.keys())])


def calc_market_emissions_multi(rows: List[Dict[str, Any]], pct_gdo: float, year: int) -> Dict[str, Any]:
    valid_rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    notes: List[str] = []
    emissions_gross_kg = 0.0
    total_mwh = 0.0
    for idx, row in enumerate(rows, start=1):
        supplier_name = str(row.get("supplier_name") or "").strip()
        consumo_mwh = max(0.0, _to_float(row.get("consumo_mwh")))
        if consumo_mwh <= 0:
            continue
        if not supplier_name:
            errors.append(f"Fila {idx}: comercializadora vacía con consumo > 0.")
            continue
        factor, pretty_name = get_supplier_factor(year, supplier_name)
        if factor is None:
            errors.append(f"No existe factor para '{supplier_name}' en la tabla del año {year}.")
            continue
        emissions_row_kg = consumo_mwh * 1000.0 * factor
        total_mwh += consumo_mwh
        emissions_gross_kg += emissions_row_kg
        if factor == 0:
            notes.append(
                f"La comercializadora '{pretty_name}' tiene factor reportado 0 kg CO2e/kWh para {year}."
            )
        valid_rows.append(
            {
                "supplier_name": pretty_name,
                "consumo_mwh": consumo_mwh,
                "factor_kg_kwh": factor,
                "emisiones_kg": emissions_row_kg,
            }
        )
    pct = min(100.0, max(0.0, pct_gdo))
    adjusted = emissions_gross_kg * (1.0 - pct / 100.0)
    weighted = emissions_gross_kg / (total_mwh * 1000.0) if total_mwh > 0 else 0.0
    return {
        "rows": valid_rows,
        "errors": errors,
        "notes": notes,
        "consumo_total_mwh": total_mwh,
        "emisiones_brutas_kg": emissions_gross_kg,
        "emisiones_ajustadas_kg": adjusted,
        "factor_ponderado_kg_kwh": weighted,
    }


def list_stationary_fuels() -> List[Dict[str, Any]]:
    return load_stationary_fuels_catalog()


def list_mobile_fuels() -> List[Dict[str, Any]]:
    return load_mobile_fuels_catalog()


def list_refrigerants() -> List[Dict[str, Any]]:
    return load_refrigerants_catalog()


def list_electricity_suppliers(year: int) -> List[Dict[str, Any]]:
    df = load_supplier_factors()
    filtered = df[df["year"] == year].sort_values("supplier_name")
    items = filtered[["supplier_name", "factor_kg_co2e_kwh"]].rename(
        columns={"supplier_name": "name"}
    ).to_dict(orient="records")
    if not filtered.empty:
        generic_factor = float(filtered["factor_kg_co2e_kwh"].mean())
        return [
            {
                "name": GENERIC_MARKET_SUPPLIER_NAME,
                "factor_kg_co2e_kwh": generic_factor,
            },
            *items,
        ]
    return items


def get_stationary_fuel_entries(company: Dict[str, Any]) -> List[Dict[str, Any]]:
    return company.get("stationary_fuels") or []


def get_mobile_fuel_entries(company: Dict[str, Any]) -> List[Dict[str, Any]]:
    return company.get("mobile_fuels") or []


def get_refrigerant_entries(company: Dict[str, Any]) -> List[Dict[str, Any]]:
    return company.get("refrigerants") or []


def estimate_stationary_fuel_mwh(company: Dict[str, Any]) -> float:
    fuels = stationary_fuels_by_key()
    total = 0.0
    for entry in get_stationary_fuel_entries(company):
        fuel = fuels.get(str(entry.get("fuel_key") or ""))
        if fuel and isinstance(fuel.get("mwh_per_unit"), (int, float)):
            total += _to_float(entry.get("quantity")) * float(fuel["mwh_per_unit"])
    return total


def calculate_scope1_stationary(company: Dict[str, Any]) -> Dict[str, Any]:
    year = int(company.get("inventory_year") or DEFAULT_INVENTORY_YEAR)
    fuels = stationary_fuels_by_key()
    total = 0.0
    breakdown: Dict[str, float] = {}
    years_used: set[int] = set()
    for entry in get_stationary_fuel_entries(company):
        key = str(entry.get("fuel_key") or "")
        fuel = fuels.get(key)
        if not fuel:
            continue
        factor_kg_per_unit, factor_year = latest_factor(fuel["factors_kg_per_unit"], year)
        emissions_t = _to_float(entry.get("quantity")) * factor_kg_per_unit / 1000.0
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


def calculate_scope1_mobile(company: Dict[str, Any]) -> Dict[str, Any]:
    year = int(company.get("inventory_year") or DEFAULT_INVENTORY_YEAR)
    fuels = mobile_fuels_by_key()
    total = 0.0
    details: List[Dict[str, Any]] = []
    for entry in get_mobile_fuel_entries(company):
        key = str(entry.get("fuel_key") or "")
        fuel = fuels.get(key)
        if not fuel:
            continue
        factor_kg_per_unit, factor_year = latest_factor(fuel["factors_kg_per_unit"], year)
        emissions_t = _to_float(entry.get("quantity")) * factor_kg_per_unit / 1000.0
        total += emissions_t
        details.append(
            {
                "fuel": fuel["fuel_label"],
                "vehicle": fuel["vehicle_type"],
                "t": emissions_t,
                "note": f"factor {factor_year}",
            }
        )
    return {"emissions_t": total, "details": details}


def calculate_fugitive_emissions(company: Dict[str, Any]) -> Dict[str, Any]:
    refrigerants = refrigerants_by_name()
    total = 0.0
    found = True
    details: List[Dict[str, Any]] = []
    for entry in get_refrigerant_entries(company):
        ref = refrigerants.get(str(entry.get("name") or "").upper())
        if not ref:
            found = False
            continue
        kg = _to_float(entry.get("quantity"))
        emissions_t = kg * ref["gwp"] / 1000.0
        total += emissions_t
        details.append({"name": entry.get("name"), "gwp": ref["gwp"], "kg": kg, "emissions_t": emissions_t})
    primary = details[0] if details else {"name": "", "gwp": 0.0}
    return {
        "emissions_t": total,
        "gwp": primary["gwp"],
        "source": "catálogo local PCA 6AR",
        "found": found,
        "details": details,
        "primary_name": primary["name"],
    }


def calculate_scope2_electricity(company: Dict[str, Any]) -> Dict[str, Any]:
    year = int(company.get("inventory_year") or DEFAULT_INVENTORY_YEAR)
    method = str(company.get("electricity_method") or "location").lower()
    location_factor = get_location_factor(year)
    if method == "market":
        rows = company.get("scope2_supplier_rows") or []
        if not rows:
            supplier_name = str(company.get("supplier_name") or "").strip()
            annual_electricity_mwh = _to_float(company.get("annual_electricity_mwh"))
            if supplier_name and annual_electricity_mwh > 0:
                rows = [{"supplier_name": supplier_name, "consumo_mwh": annual_electricity_mwh}]
        pct_gdo = _to_float(company.get("gdo_coverage_pct"))
        market = calc_market_emissions_multi(rows, pct_gdo, year)
        location_emissions_kg = market["consumo_total_mwh"] * 1000.0 * location_factor
        return {
            "method": "market-based",
            "emissions_t": market["emisiones_ajustadas_kg"] / 1000.0,
            "gross_emissions_t": market["emisiones_brutas_kg"] / 1000.0,
            "used_factor": market["factor_ponderado_kg_kwh"],
            "location_factor": location_factor,
            "location_emissions_t": location_emissions_kg / 1000.0,
            "source": f"comercializadoras CNMC + GdO ({year})",
            "rows": market["rows"],
            "errors": market["errors"],
            "notes": market["notes"],
            "consumo_total_mwh": market["consumo_total_mwh"],
            "difference_t": market["emisiones_ajustadas_kg"] / 1000.0 - location_emissions_kg / 1000.0,
            "difference_pct": (
                ((market["emisiones_ajustadas_kg"] - location_emissions_kg) / location_emissions_kg) * 100.0
                if location_emissions_kg > 0
                else 0.0
            ),
        }
    elec_mwh = _to_float(company.get("annual_electricity_mwh"))
    emissions_t = elec_mwh * location_factor
    return {
        "method": "location-based",
        "emissions_t": emissions_t,
        "gross_emissions_t": emissions_t,
        "used_factor": location_factor,
        "location_factor": location_factor,
        "location_emissions_t": emissions_t,
        "source": f"factor REE España ({year})",
        "rows": [],
        "errors": [],
        "notes": [],
        "consumo_total_mwh": elec_mwh,
        "difference_t": 0.0,
        "difference_pct": 0.0,
    }


def calculate_scope2_heat(company: Dict[str, Any]) -> Dict[str, Any]:
    heat_mwh = _to_float(company.get("annual_purchased_heat_mwh"))
    heat_factor = _to_float(company.get("co2_factor_heat_t_per_mwh"))
    used_factor = heat_factor if heat_factor > 0 else 0.20
    return {
        "emissions_t": heat_mwh * used_factor,
        "used_factor": used_factor,
        "source": "factor calor aportado por la empresa" if heat_factor > 0 else "fallback interno 0.20 tCO2/MWh",
    }


def get_data_quality_score(company: Dict[str, Any], footprint_meta: Dict[str, Any]) -> Tuple[str, str]:
    score = 0
    score += 2 if company.get("has_invoices") else 0
    score += 2 if company.get("has_meters") else 0
    score += 1 if company.get("has_submetering") else 0
    score += 1 if company.get("cnmc_supplier_known") else 0
    score += 1 if company.get("electricity_has_gdo") else 0
    score += 2 if company.get("has_energy_audit") else 0
    score += 1 if footprint_meta.get("used_elec_factor", 0) > 0 else 0
    if score >= 7:
        return "Alta", "Datos con alta trazabilidad (facturas, medición y auditoría)."
    if score >= 4:
        return "Media", "Datos razonables; mejorar trazabilidad con facturas y medición."
    return "Baja", "Datos incompletos; conviene sustituir supuestos por datos verificados."


def calculate_company_footprint(company: Dict[str, Any]) -> Dict[str, Any]:
    scope1_stationary = calculate_scope1_stationary(company)
    scope1_mobile = calculate_scope1_mobile(company)
    scope1_fugitive = calculate_fugitive_emissions(company)
    scope2_electricity = calculate_scope2_electricity(company)
    scope2_heat = calculate_scope2_heat(company)
    scope1_t = scope1_stationary["emissions_t"] + scope1_mobile["emissions_t"] + scope1_fugitive["emissions_t"]
    scope2_t = scope2_electricity["emissions_t"] + scope2_heat["emissions_t"]
    total_t = scope1_t + scope2_t
    quality = get_data_quality_score(
        {
            **company,
            "cnmc_supplier_known": len(scope2_electricity.get("rows", [])) > 0,
            "electricity_has_gdo": _to_float(company.get("gdo_coverage_pct")) > 0,
        },
        {"used_elec_factor": scope2_electricity["used_factor"]},
    )
    return {
        "scope1_t": scope1_t,
        "scope2_t": scope2_t,
        "total_t": total_t,
        "scope1_stationary_t": scope1_stationary["emissions_t"],
        "scope1_fleet_t": scope1_mobile["emissions_t"],
        "scope1_fugitive_t": scope1_fugitive["emissions_t"],
        "scope2_elec_t": scope2_electricity["emissions_t"],
        "scope2_heat_t": scope2_heat["emissions_t"],
        "scope2_elec_method": scope2_electricity["method"],
        "scope2_location_t": scope2_electricity["location_emissions_t"],
        "scope2_market_t": scope2_electricity["emissions_t"] if scope2_electricity["method"] == "market-based" else scope2_electricity["location_emissions_t"],
        "scope2_difference_t": scope2_electricity["difference_t"],
        "scope2_difference_pct": scope2_electricity["difference_pct"],
        "used_elec_factor": scope2_electricity["used_factor"],
        "used_heat_factor": scope2_heat["used_factor"],
        "scope1_factor_source": scope1_stationary["source"],
        "scope2_elec_source": scope2_electricity["source"],
        "scope2_heat_source": scope2_heat["source"],
        "refrigerant_factor_found": scope1_fugitive["found"],
        "refrigerant_key": scope1_fugitive["primary_name"],
        "refrigerant_gwp": scope1_fugitive["gwp"],
        "breakdown": {
            "Combustión fija": scope1_stationary["emissions_t"],
            "Flota móvil": scope1_mobile["emissions_t"],
            "Refrigerantes": scope1_fugitive["emissions_t"],
            "Electricidad": scope2_electricity["emissions_t"],
            "Calor/vapor comprado": scope2_heat["emissions_t"],
        },
        "quality": {"label": quality[0], "text": quality[1]},
        "scope2_rows": scope2_electricity["rows"],
        "scope2_notes": scope2_electricity["notes"],
        "scope2_errors": scope2_electricity["errors"],
    }


def enrich_company_inputs(company: Dict[str, Any], footprint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    enriched = dict(company)
    fp = footprint or calculate_company_footprint(enriched)

    stationary_fuel_mwh = estimate_stationary_fuel_mwh(enriched)
    enriched["annual_fuel_mwh"] = stationary_fuel_mwh if stationary_fuel_mwh > 0 else 0.0
    enriched["co2_factor_elec_t_per_mwh"] = fp.get("used_elec_factor", 0.0)
    enriched["co2_factor_fuel_t_per_mwh"] = (
        fp.get("scope1_stationary_t", 0.0) / stationary_fuel_mwh if stationary_fuel_mwh > 0 else None
    )

    total_energy_mwh = _to_float(enriched.get("annual_electricity_mwh")) + _to_float(enriched.get("annual_fuel_mwh"))
    if total_energy_mwh >= 20000:
        enriched["energy_intensity"] = "Alta"
    elif total_energy_mwh >= 5000:
        enriched["energy_intensity"] = "Media"
    else:
        enriched["energy_intensity"] = "Baja"

    elec_factor = _to_float(enriched.get("co2_factor_elec_t_per_mwh"))
    if elec_factor >= 0.3:
        enriched["grid_emissions_level"] = "Alto"
    elif 0 < elec_factor <= 0.1:
        enriched["grid_emissions_level"] = "Bajo"
    else:
        enriched["grid_emissions_level"] = "Medio"

    enriched["fossil_heat_use"] = "Alto" if stationary_fuel_mwh > 0 else "Ninguno"
    country_text = str(enriched.get("country_region") or enriched.get("country") or "").lower()
    enriched["eu_context"] = any(token in country_text for token in ["espa", "spain", "ue", "eu"])
    enriched["has_fleet"] = any(_to_float(row.get("quantity")) > 0 for row in get_mobile_fuel_entries(enriched))
    enriched["has_refrigerants"] = len(get_refrigerant_entries(enriched)) > 0

    for key in [
        "annual_electricity_mwh",
        "annual_fuel_mwh",
        "electricity_price_eur_mwh",
        "fuel_price_eur_mwh",
        "co2_factor_elec_t_per_mwh",
        "co2_factor_fuel_t_per_mwh",
        "roof_area_m2",
    ]:
        if isinstance(enriched.get(key), (int, float)) and float(enriched[key]) == 0.0:
            enriched[key] = None

    return enriched


def build_ai_company_context(company_inputs: Dict[str, Any], footprint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    annual_electricity_mwh = _to_float(company_inputs.get("annual_electricity_mwh"))
    annual_fuel_mwh = estimate_stationary_fuel_mwh(company_inputs)
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
        "province": company_inputs.get("province") or "",
        "postal_code": company_inputs.get("postal_code") or "",
        "main_customer_locations": company_inputs.get("main_customer_locations") or "",
        "inventory_year": int(company_inputs.get("inventory_year") or DEFAULT_INVENTORY_YEAR),
        "electricity_method": company_inputs.get("electricity_method") or "location",
        "annual_electricity_mwh": annual_electricity_mwh,
        "annual_fuel_mwh": annual_fuel_mwh,
        "size_signal": size_band,
        "implemented_measures": company_inputs.get("implemented_measures") or {},
        "footprint_total_t": _to_float((footprint or {}).get("total_t"), default=np.nan),
    }


def build_emissions_input_context(company_inputs: Dict[str, Any], footprint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fp = footprint or {}
    return {
        "inventory_year": int(company_inputs.get("inventory_year") or DEFAULT_INVENTORY_YEAR),
        "scope1": {
            "stationary_fuels": company_inputs.get("stationary_fuels") or [],
            "mobile_fuels": company_inputs.get("mobile_fuels") or [],
            "refrigerants": company_inputs.get("refrigerants") or [],
            "annual_fuel_mwh": company_inputs.get("annual_fuel_mwh"),
            "co2_factor_fuel_t_per_mwh": company_inputs.get("co2_factor_fuel_t_per_mwh"),
            "scope1_stationary_t": fp.get("scope1_stationary_t"),
            "scope1_fleet_t": fp.get("scope1_fleet_t"),
            "scope1_fugitive_t": fp.get("scope1_fugitive_t"),
        },
        "scope2": {
            "electricity_method": company_inputs.get("electricity_method"),
            "annual_electricity_mwh": company_inputs.get("annual_electricity_mwh"),
            "scope2_supplier_rows": company_inputs.get("scope2_supplier_rows") or [],
            "supplier_name": company_inputs.get("supplier_name"),
            "gdo_coverage_pct": company_inputs.get("gdo_coverage_pct"),
            "annual_purchased_heat_mwh": company_inputs.get("annual_purchased_heat_mwh"),
            "co2_factor_heat_t_per_mwh": company_inputs.get("co2_factor_heat_t_per_mwh"),
            "co2_factor_elec_t_per_mwh": company_inputs.get("co2_factor_elec_t_per_mwh"),
            "scope2_elec_t": fp.get("scope2_elec_t"),
            "scope2_heat_t": fp.get("scope2_heat_t"),
            "scope2_location_t": fp.get("scope2_location_t"),
            "scope2_market_t": fp.get("scope2_market_t"),
        },
        "footprint_summary": {
            "scope1_t": fp.get("scope1_t"),
            "scope2_t": fp.get("scope2_t"),
            "total_t": fp.get("total_t"),
            "breakdown": fp.get("breakdown"),
        },
    }


def generate_pestel_rule_based(company: Dict[str, Any], footprint: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    province = str(company.get("province") or "").strip()
    method = str(company.get("electricity_method") or "location").lower()
    inventory_year = int(company.get("inventory_year") or DEFAULT_INVENTORY_YEAR)
    total_t = _to_float((footprint or {}).get("total_t"), default=0.0)
    implemented = company.get("implemented_measures") or {}
    measures = [k for k, v in implemented.items() if str(v).strip().lower() not in {"no", "", "false"}]

    return {
        "Político": [
            f"El año base {inventory_year} obliga a sostener una metodología de inventario consistente y auditable.",
            f"La localización en {province or 'España'} condiciona tramitación energética, ayudas y viabilidad de autoconsumo.",
        ],
        "Económico": [
            "La volatilidad de electricidad y combustibles refuerza el valor de iniciativas con ahorro verificable.",
            "El CAPEX debe equilibrar quick wins con proyectos estructurales de calor, suministro y electrificación.",
        ],
        "Social": [
            "Clientes, financiadores y dirección esperan resultados medibles y narrativa técnica sólida.",
            "La implantación real depende de coordinación entre operaciones, mantenimiento, compras y finanzas.",
        ],
        "Tecnológico": [
            f"El método eléctrico actual ({method}-based) condiciona el atractivo relativo de GdO, PPA, FV y electrificación.",
            "La medición granular y el MRV robusto son la base para decidir con menor incertidumbre.",
        ],
        "Ambiental": [
            f"La huella estimada ronda {total_t:,.1f} tCO2e/año y debe desagregarse para priorizar por fuente emisora.",
            "Las decisiones sobre calor, electricidad comprada y refrigerantes suelen concentrar el mayor potencial de reducción.",
        ],
        "Legal": [
            "Conviene documentar factores, facturas, supuestos y evidencias para auditoría o aseguramiento limitado.",
            f"Las medidas ya implantadas ({', '.join(measures) if measures else 'ninguna declarada'}) deben reflejarse de forma coherente en el plan.",
        ],
    }


def _extract_json_text(text: str) -> str:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(content):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(content[index:])
            return content[index : index + end]
        except json.JSONDecodeError:
            continue
    return content


def _loads_gemini_json(text: str) -> Any:
    content = _extract_json_text(text)
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        preview = re.sub(r"\s+", " ", str(text or "")).strip()[:500]
        raise RuntimeError(
            "Gemini no devolvió JSON válido. "
            f"Detalle técnico: {exc.msg} en línea {exc.lineno}, columna {exc.colno}. "
            f"Fragmento recibido: {preview or 'respuesta vacía'}"
        ) from exc


def _clean_ai_bullet(item: Any) -> str:
    if isinstance(item, dict):
        for key in ["bullet", "text", "idea", "descripcion", "description", "content"]:
            value = item.get(key)
            if value:
                item = value
                break
        else:
            item = " ".join(str(value) for value in item.values() if str(value).strip())
    return re.sub(r"\s*\[\d+(?:,\s*\d+)*\]\s*", " ", str(item)).strip()


def _gemini_repair_json(api_key: str, model_name: str, raw_text: str, json_shape_hint: str = "") -> Any:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    system_prompt = (
        "Eres un conversor estricto a JSON. Devuelve SOLO JSON válido, sin markdown, sin explicación y sin texto adicional. "
        "Conserva el contenido sustantivo del texto original y no añadas hechos nuevos."
    )
    user_prompt = (
        "Convierte o repara el siguiente resultado para que sea JSON válido y siga la estructura esperada.\n\n"
        f"ESTRUCTURA ESPERADA: {json_shape_hint or 'JSON válido equivalente al contenido recibido'}\n\n"
        f"TEXTO A CONVERTIR:\n{raw_text}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError(f"Gemini investigó, pero no se pudo convertir la respuesta a JSON: {exc}") from exc
    if response.status_code >= 400:
        raise RuntimeError(
            "Gemini investigó, pero falló la conversión a JSON. "
            f"Error API Gemini ({response.status_code}) con modelo {model_name}: {response.text[:800]}"
        )
    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini investigó, pero la conversión a JSON no devolvió candidatos.")
    parts = candidates[0].get("content", {}).get("parts", [])
    repaired_text = "".join(str(part.get("text", "")) for part in parts).strip()
    return _loads_gemini_json(repaired_text)


def gemini_should_use_web_research(company_inputs: Dict[str, Any]) -> bool:
    return bool(str(company_inputs.get("company_name") or "").strip())


def _gemini_generate_json(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    use_web_research: bool = False,
    require_web_research: bool = False,
    json_shape_hint: str = "",
) -> Dict[str, Any]:
    model_name = (model or "gemini-2.5-flash").strip()
    if model_name.startswith("models/"):
        model_name = model_name.split("/", 1)[1]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.25, "maxOutputTokens": 8192},
    }
    if not use_web_research:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    if use_web_research:
        payload["tools"] = [{"google_search": {}}]
    try:
        response = requests.post(url, json=payload, timeout=90)
    except requests.RequestException as exc:
        raise RuntimeError(f"No se pudo conectar con Gemini: {exc}") from exc
    if response.status_code >= 400 and use_web_research and not require_web_research and response.status_code != 429:
        payload.pop("tools", None)
        try:
            response = requests.post(url, json=payload, timeout=90)
        except requests.RequestException as exc:
            raise RuntimeError(f"No se pudo conectar con Gemini: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text[:1200]
        try:
            error_payload = response.json().get("error", {})
            message = str(error_payload.get("message") or "").strip()
            details = error_payload.get("details") or []
            quota_violations = []
            for item in details:
                for violation in item.get("violations", []) or []:
                    subject = str(violation.get("subject") or "").strip()
                    description = str(violation.get("description") or "").strip()
                    if subject or description:
                        quota_violations.append(" - ".join(part for part in [subject, description] if part))
            if message or quota_violations:
                detail = message
                if quota_violations:
                    detail = f"{detail} | Cuota: {'; '.join(quota_violations)}"
        except Exception:
            pass
        raise RuntimeError(f"Error API Gemini ({response.status_code}) con modelo {model_name}: {detail}")
    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini no devolvió candidatos.")
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        finish_reason = str(candidate.get("finishReason") or "").strip()
        raise RuntimeError(
            "Gemini devolvió texto vacío"
            + (f" (finishReason={finish_reason})." if finish_reason else ".")
        )
    grounding = candidate.get("groundingMetadata") or {}
    grounding_queries = [str(item) for item in grounding.get("webSearchQueries", []) if str(item).strip()]
    grounding_chunks = grounding.get("groundingChunks", []) or []
    grounding_sources: List[Dict[str, str]] = []
    for chunk in grounding_chunks:
        web = chunk.get("web") or {}
        uri = str(web.get("uri") or "").strip()
        title = str(web.get("title") or "").strip()
        if uri:
            grounding_sources.append({"uri": uri, "title": title})
    grounding_used = bool(grounding_queries or grounding_sources)
    if require_web_research and not grounding_used:
        raise RuntimeError(
            "Gemini generó contenido, pero no confirmó búsqueda web. "
            "Vuelve a intentarlo o revisa que el modelo tenga grounding/búsqueda web disponible."
        )
    try:
        parsed_data = _loads_gemini_json(text)
    except RuntimeError:
        if not use_web_research:
            raise
        parsed_data = _gemini_repair_json(api_key, model_name, text, json_shape_hint)
    return {
        "data": parsed_data,
        "grounding_used": grounding_used,
        "grounding_queries": grounding_queries,
        "grounding_sources": grounding_sources,
    }


def _build_ai_web_research_context(
    company: Dict[str, Any],
    footprint: Optional[Dict[str, Any]],
    api_key: str,
    model: str,
    purpose: str,
    require_web_research: bool = False,
) -> Dict[str, Any]:
    company_name = str(company.get("company_name") or "").strip()
    sector = str(company.get("sector") or company.get("cnae_sector") or "").strip()
    province = str(company.get("province") or "").strip()
    postal_code = str(company.get("postal_code") or "").strip()
    if not any([company_name, sector, province]):
        return {"data": {}, "grounding_used": False, "grounding_queries": [], "grounding_sources": []}

    system_prompt = (
        "Eres un analista de investigación web para consultoría de descarbonización. "
        "Debes buscar información pública reciente y devolver SOLO JSON válido. "
        "No inventes datos: si no encuentras información específica de la empresa, usa contexto sectorial/local y márcalo como inferido."
    )
    user_prompt = (
        "Realiza búsqueda web antes de responder. Investiga señales útiles para personalizar un análisis de huella de carbono.\n\n"
        f"OBJETIVO: {purpose}\n"
        f"EMPRESA: {company_name or 'no indicada'}\n"
        f"SECTOR: {sector or 'no indicado'}\n"
        f"PROVINCIA/SEDE: {province or 'no indicada'}\n"
        f"CODIGO_POSTAL: {postal_code or 'no indicado'}\n\n"
        "Busca, como mínimo, combinaciones de: empresa + sostenibilidad, empresa + actividad, empresa + emisiones, "
        "empresa + logística/energía, sector + provincia + ayudas eficiencia energética, sector + regulación climática.\n\n"
        "Devuelve JSON con esta estructura exacta:\n"
        "{\n"
        '  "company_activity": ["2-4 señales concretas sobre qué hace la empresa y cómo opera"],\n'
        '  "location_market": ["2-4 señales sobre ubicación, clientes/mercados, logística o contexto local"],\n'
        '  "sustainability_energy": ["2-4 señales sobre energía, emisiones, sostenibilidad, residuos, refrigeración, transporte o compras"],\n'
        '  "regulation_aid_risks": ["2-4 señales sobre normativa, ayudas, riesgos o presión de mercado"],\n'
        '  "research_limitations": ["datos no encontrados o inferencias prudentes"]\n'
        "}\n\n"
        f"DATOS_HUELLA: {json.dumps(footprint or {}, ensure_ascii=False)}\n"
        f"INPUTS_EMPRESA: {json.dumps(company, ensure_ascii=False)}"
    )
    try:
        return _gemini_generate_json(
            api_key,
            model,
            system_prompt,
            user_prompt,
            use_web_research=True,
            require_web_research=require_web_research,
            json_shape_hint='{"company_activity":["..."],"location_market":["..."],"sustainability_energy":["..."],"regulation_aid_risks":["..."],"research_limitations":["..."]}',
        )
    except Exception as exc:
        if require_web_research:
            raise
        return {
            "data": {"research_limitations": [f"No se pudo completar la investigación web previa: {exc}"]},
            "grounding_used": False,
            "grounding_queries": [],
            "grounding_sources": [],
        }


def _extract_initiative_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        for key in ["initiatives", "iniciativas", "items", "data"]:
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        raise RuntimeError("La salida de Gemini para iniciativas debe ser una lista JSON.")
    return [item for item in data if isinstance(item, dict)]


def generate_ai_pestel(company: Dict[str, Any], footprint: Dict[str, Any], api_key: str, model: str) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("No hay GEMINI_API_KEY para generar el PESTEL con IA.")
    context = build_ai_company_context(company, footprint)
    emissions_context = build_emissions_input_context(company, footprint)
    use_web_research = True
    system_prompt = (
        "Eres un consultor senior de descarbonización industrial. "
        "Devuelve SOLO JSON válido, sin texto adicional. "
        "Responde en español. "
        "Estructura: claves Político, Económico, Social, Tecnológico, Ambiental, Legal "
        "y cada valor es una lista de bullets densos pero claros. "
        "El análisis debe estar adaptado al sector, a la localización de la empresa en España y a su perfil operativo/energético. "
        "Evita bullets genéricos que podrían aplicarse a cualquier empresa. "
        "Antes de responder, realiza búsqueda web si el modelo dispone de esa herramienta. "
        "Investiga primero la empresa y su actividad real; después sintetiza el PESTEL. "
        "Debes investigar la empresa, su ubicación, el sector, su actividad real, productos o servicios, procesos, contexto logístico y cualquier señal pública relevante "
        "que ayude a entender mejor sus retos de descarbonización. "
        "Cruza esa información con todos los datos estructurados facilitados por el usuario. "
        "Si algo no se puede verificar, indícalo con prudencia y apóyate en sector + localización + datos introducidos."
    )
    user_prompt = (
        "Genera un PESTEL accionable y razonado (3-4 bullets por categoría) "
        "basado en los datos de la empresa y, si hay nombre de empresa y la herramienta lo permite, en información pública verificable.\n\n"
        "APARTADO DE BUSQUEDA Y CONTRASTE OBLIGATORIO:\n"
        "- Busca primero informacion publica reciente sobre la empresa, su sector, su provincia/comunidad autonoma, normativa, ayudas aplicables, competidores, cadena de suministro, tecnologia y noticias de los ultimos 6-12 meses.\n"
        "- Amplia la busqueda a senales macro actuales solo si son materialmente relevantes: conflictos geopoliticos, energia, inflacion, tipos de interes, transporte maritimo, materias primas, aranceles, regulacion climatica, consumo, empleo y cambios tecnologicos.\n"
        "- Contrasta actividad real, productos o servicios, procesos, ubicacion, contexto industrial/local, riesgos energeticos, presion regulatoria, expectativas de clientes y dependencias operativas.\n"
        "- Adapta el analisis al modelo de negocio concreto de la empresa. No uses ejemplos fijos: decide que palancas importan segun su actividad real.\n"
        "- Si detectas un shock actual relevante, explica el mecanismo de impacto indirecto sobre costes, energia, logistica, materias primas, inventario, clientes, margen, inversiones o cumplimiento.\n"
        "- En Tecnologico incorpora las tendencias mas actuales y aplicables al sector, incluyendo IA solo cuando tenga sentido operativo: prediccion de demanda, optimizacion de inventario, rutas logisticas, eficiencia energetica, mantenimiento predictivo, analitica de clientes, automatizacion o medicion de huella.\n"
        "- Usa esa busqueda para personalizar cada categoria PESTEL; no escribas bullets genericos.\n"
        "- Si la busqueda no devuelve datos especificos de la empresa, trabaja desde sector, provincia, actividad probable y datos energeticos introducidos, dejando claro el razonamiento sin inventar hechos.\n"
        "- No inventes fuentes, cifras ni hechos concretos no verificados.\n"
        "- No incluyas citas numeradas tipo [1], [2], [19] ni referencias bibliograficas dentro de los bullets.\n\n"
        "PROFUNDIDAD Y RAZONAMIENTO:\n"
        "- No te limites a listar factores: explica por que cada senal importa para la empresa concreta.\n"
        "- Cada bullet debe seguir esta logica: hecho/senal actual -> impacto para la empresa -> implicacion para descarbonizacion, coste, operacion, cliente, cadena de suministro o cumplimiento.\n"
        "- Prioriza informacion actual y material sobre informacion historica o decorativa.\n"
        "- Si hay mucha informacion, selecciona lo mas accionable para la toma de decisiones de descarbonizacion.\n\n"
        "Reglas:\n"
        "- Primero busca información pública sobre la empresa y su ubicación si la herramienta de búsqueda está disponible.\n"
        "- Usa el nombre de la empresa, la provincia, el código postal, el sector y cualquier dato operativo para afinar la búsqueda.\n"
        "- Busca señales sobre actividad, productos, procesos, plantas, logística, clientes, presión regulatoria local y contexto industrial de la zona.\n"
        "- Analiza también los datos introducidos para el cálculo de emisiones para entender dónde están los consumos energéticos, las fuentes emisoras y las áreas principales de mejora.\n"
        "- Ten en cuenta expresamente el sector y la provincia/ubicación.\n"
        "- Da prioridad a implicaciones reales para consumo energético, combustibles, calor de proceso, logística, regulación e inversión.\n"
        "- No repitas obviedades vacías.\n"
        "- Si haces una inferencia, que sea razonable y ligada al contexto.\n\n"
        f"COMPANY_CONTEXT: {json.dumps(context, ensure_ascii=False)}\n"
        f"EMISSIONS_INPUT_CONTEXT: {json.dumps(emissions_context, ensure_ascii=False)}\n"
        f"RAW_INPUTS: {json.dumps(company, ensure_ascii=False)}\n"
        f"FOOTPRINT: {json.dumps(footprint, ensure_ascii=False)}"
    )
    result = _gemini_generate_json(
        api_key,
        model,
        system_prompt,
        user_prompt,
        use_web_research=use_web_research,
        require_web_research=False,
    )
    data = result["data"]
    keys = ["Político", "Económico", "Social", "Tecnológico", "Ambiental", "Legal"]
    return {
        "pestel": {
            key: [_clean_ai_bullet(item) for item in data.get(key, [])]
            for key in keys
        },
        "grounding_used": bool(result["grounding_used"]),
        "grounding_queries": result["grounding_queries"],
        "grounding_sources": result["grounding_sources"],
    }


def generate_ai_pestel(company: Dict[str, Any], footprint: Dict[str, Any], api_key: str, model: str) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("No hay GEMINI_API_KEY para generar el PESTEL con IA.")
    context = build_ai_company_context(company, footprint)
    emissions_context = build_emissions_input_context(company, footprint)
    keys = ["Político", "Económico", "Social", "Tecnológico", "Ambiental", "Legal"]
    research = _build_ai_web_research_context(company, footprint, api_key, model, "PESTEL de descarbonización")
    system_prompt = (
        "Eres un consultor senior de descarbonización empresarial y huella de carbono. "
        "Devuelve SOLO JSON válido, sin texto adicional. Responde en español. "
        "Estructura exacta: claves Político, Económico, Social, Tecnológico, Ambiental, Legal; "
        "cada clave contiene 2-3 bullets. Máximo 18 bullets en total. "
        "Cada bullet debe ser breve, concreto y accionable, de 16 a 26 palabras. "
        "En cada bullet, decide según el contexto 1-2 conceptos clave y márcalos con **negrita** usando sintaxis Markdown. "
        "Antes de responder, realiza búsqueda web si el modelo dispone de esa herramienta. "
        "Investiga primero la empresa, su actividad real y su contexto local; después adapta el análisis a sector, actividad, sede/provincia, "
        "clientes/mercados principales, sostenibilidad, energía y huella de carbono. "
        "Los hallazgos web deben notarse en el contenido: menciona señales concretas de actividad, mercado, ubicación, normativa, energía o clientes sin incluir URLs. "
        "No inventes datos, clientes, fuentes ni cifras. Si algo es inferido, que sea prudente y contextual."
    )
    user_prompt = (
        "Genera un PESTEL sintético y útil para decidir medidas de descarbonización.\n\n"
        "BÚSQUEDA WEB Y CONTEXTO:\n"
        "- Busca información reciente sobre la empresa por nombre exacto y, si es necesario, por nombre + provincia, sector, marcas, productos o actividad.\n"
        "- Busca también sector, provincia/comunidad autónoma, normativa, ayudas, precios/energía, logística, clientes/mercados y sostenibilidad aplicable.\n"
        "- Extrae 3-5 señales concretas antes de redactar: qué hace la empresa, dónde opera, qué presión energética/regulatoria tiene y qué clientes/mercados puede atender.\n"
        "- Considera la sede/provincia y código postal, y las ubicaciones de clientes principales si el usuario las aporta.\n"
        "- Si no hay clientes concretos, usa una zona comercial amplia derivada del sector y la ubicación; no inventes nombres.\n"
        "- Cruza los hallazgos con consumos, combustibles, electricidad, refrigerantes y resultados de huella.\n\n"
        "ESTILO DE SALIDA:\n"
        "- 2-3 bullets por categoría PESTEL; máximo 18 bullets en total.\n"
        "- Cada bullet debe ocupar como máximo 3-4 líneas visuales y evitar explicaciones largas.\n"
        "- Cada bullet: señal actual -> impacto concreto -> implicación para huella, energía, costes, clientes o cumplimiento.\n"
        "- Marca en **negrita** solo los conceptos realmente clave de cada bullet: tecnologías, normas, riesgos, fuentes emisoras, objetivos o palancas de decisión.\n"
        "- No pongas frases completas en negrita; usa 1-2 fragmentos cortos por bullet.\n"
        "- Evita bullets que podrían servir para cualquier empresa. Cada bullet debe referirse a actividad, ubicación, fuente emisora o mercado concreto.\n"
        "- Prioriza sostenibilidad, huella de carbono, eficiencia energética, combustibles, logística, ayudas, regulación y medición de emisiones.\n"
        "- Evita frases genéricas tipo 'cumplir la normativa' si no explican una acción o presión concreta.\n"
        "- No incluyas citas numeradas ni bibliografía dentro de los bullets.\n\n"
        f"WEB_RESEARCH_CONTEXT: {json.dumps(research.get('data', {}), ensure_ascii=False)}\n"
        f"COMPANY_CONTEXT: {json.dumps(context, ensure_ascii=False)}\n"
        f"EMISSIONS_INPUT_CONTEXT: {json.dumps(emissions_context, ensure_ascii=False)}\n"
        f"RAW_INPUTS: {json.dumps(company, ensure_ascii=False)}\n"
        f"FOOTPRINT: {json.dumps(footprint, ensure_ascii=False)}"
    )
    result = _gemini_generate_json(
        api_key,
        model,
        system_prompt,
        user_prompt,
        use_web_research=True,
        require_web_research=False,
        json_shape_hint='{"Político":["..."],"Económico":["..."],"Social":["..."],"Tecnológico":["..."],"Ambiental":["..."],"Legal":["..."]}',
    )
    data = result["data"]
    return {
        "pestel": {
            key: [
                _clean_ai_bullet(item)
                for item in (data.get(key, []) or [])[:2]
                if str(item).strip()
            ]
            for key in keys
        },
        "grounding_used": bool(result["grounding_used"] or research.get("grounding_used")),
        "grounding_queries": list(dict.fromkeys([*research.get("grounding_queries", []), *result["grounding_queries"]])),
        "grounding_sources": [*research.get("grounding_sources", []), *result["grounding_sources"]],
    }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _implemented_status(company: Dict[str, Any], label: str) -> str:
    implemented = company.get("implemented_measures") or {}
    raw = str(implemented.get(label, "No")).strip().lower()
    if raw in {"sí", "si", "yes", "true", "1"}:
        return "yes"
    if raw in {"parcial", "partial"}:
        return "partial"
    return "no"


def classify_initiative_category(row: pd.Series) -> str:
    name = str(row.get("initiative") or "").lower()
    capex = _to_float(row.get("capex_eur"))
    implementation = _to_float(row.get("implementation_months"))
    quick_keywords = ["gdo", "led", "aire comprimido", "rutas", "eco-driving", "ems", "submetering"]
    if any(keyword in name for keyword in quick_keywords):
        return "quick_win"
    if capex <= 50000 and 0 < implementation <= 3:
        return "quick_win"
    return "estrategica"


def classify_thematic_bucket(row: pd.Series) -> str:
    text = f"{row.get('initiative_family', '')} {row.get('initiative', '')}".lower()
    if any(key in text for key in ["solar", "fotovolta", "pv"]):
        return "renovables"
    if any(key in text for key in ["electr", "bomba de calor", "fleet electr"]):
        return "electrificacion"
    if any(key in text for key in ["vfd", "led", "caldera", "boiler", "efficiency"]):
        return "eficiencia"
    if any(key in text for key in ["ems", "submetering", "ruta", "aire comprimido"]):
        return "operativa"
    if any(key in text for key in ["gdo", "ppa", "supply", "supplier"]):
        return "suministro"
    return "otras"


def finalize_initiatives(df: pd.DataFrame, company_inputs: Dict[str, Any], n: int = 8) -> pd.DataFrame:
    df = normalize_columns(df.copy())
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col in NUMERIC_COLUMNS else ""
    df = coerce_numeric(df)
    df["categoria"] = df.apply(classify_initiative_category, axis=1)
    df["priority_weight"] = np.where(df["categoria"] == "estrategica", 1.0, 0.4)
    df["thematic_bucket"] = df.apply(classify_thematic_bucket, axis=1)
    df["nombre"] = df["initiative"].fillna("").astype(str)
    df["tiempo_implementacion"] = df["implementation_months"]
    df["annual_co2_reduction_t"] = pd.to_numeric(df["annual_co2_reduction_t"], errors="coerce").fillna(0.0)
    df["co2_adjusted_t"] = df["annual_co2_reduction_t"] * pd.to_numeric(df["priority_weight"], errors="coerce").fillna(1.0)
    df["strategic_score_1_5"] = pd.to_numeric(df["strategic_score_1_5"], errors="coerce").fillna(3.0).clip(1.0, 5.0)
    df["capex_eur"] = pd.to_numeric(df["capex_eur"], errors="coerce").fillna(0.0)
    df["annual_opex_saving_eur"] = pd.to_numeric(df["annual_opex_saving_eur"], errors="coerce")
    df["implementation_months"] = pd.to_numeric(df["implementation_months"], errors="coerce").fillna(3.0)
    for idx, _ in enumerate(df.index, start=1):
        df.at[df.index[idx - 1], "id"] = idx
    first_cols = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    return df[first_cols].head(n).reset_index(drop=True)


def _provided_flags(company: Dict[str, Any], mapping: Dict[str, Any]) -> List[str]:
    provided: List[str] = []
    for label, value in mapping.items():
        if isinstance(value, bool):
            if value:
                provided.append(label)
        elif _ok_num(value) and float(value) > 0:
            provided.append(label)
        elif isinstance(value, str) and value.strip():
            provided.append(label)
    return provided


def propose_initiatives(company: Dict[str, Any], footprint: Optional[Dict[str, Any]] = None, n: int = 8) -> pd.DataFrame:
    annual_electricity_mwh = _to_float(company.get("annual_electricity_mwh"), default=np.nan)
    annual_fuel_mwh = estimate_stationary_fuel_mwh(company)
    electricity_price = _to_float(company.get("electricity_price_eur_mwh"), default=np.nan)
    if not _ok_num(electricity_price) or electricity_price <= 0:
        electricity_price = 120.0
    fuel_price = _to_float(company.get("fuel_price_eur_mwh"), default=np.nan)
    if not _ok_num(fuel_price) or fuel_price <= 0:
        fuel_price = 55.0
    roof_area_m2 = _to_float(company.get("roof_area_m2"), default=np.nan)
    elec_factor = _to_float((footprint or {}).get("used_elec_factor"), default=np.nan)
    fuel_factor = (
        _to_float((footprint or {}).get("scope1_stationary_t"), default=np.nan) / annual_fuel_mwh
        if annual_fuel_mwh > 0 and footprint
        else np.nan
    )
    has_fleet = any(_to_float(row.get("quantity")) > 0 for row in get_mobile_fuel_entries(company))
    has_refrigerants = any(_to_float(row.get("quantity")) > 0 for row in get_refrigerant_entries(company))
    rows: List[Dict[str, Any]] = []

    def add_if_allowed(label: str, payload: Dict[str, Any]) -> None:
        status = _implemented_status(company, label)
        if status == "yes":
            return
        if status == "partial":
            payload["initiative"] = f"Escalar: {payload['initiative']}"
        rows.append(payload)

    add_if_allowed(
        "GdO",
        {
            "scope": "Scope 2",
            "emission_source": "Electricidad comprada",
            "initiative_family": "Suministro eléctrico",
            "initiative": "Revisión de comercializadora, factor CNMC y cobertura con GdO/PPA",
            "capex_eur": 5000,
            "annual_opex_saving_eur": 0.0,
            "annual_co2_reduction_t": (annual_electricity_mwh * elec_factor * 0.5) if _ok_num(annual_electricity_mwh) and _ok_num(elec_factor) else np.nan,
            "implementation_months": 2,
            "strategic_score_1_5": 5,
            "activity_unit": "kWh",
        },
    )

    add_if_allowed(
        "EMS/submetering",
        {
            "scope": "Scope 2",
            "emission_source": "Electricidad comprada",
            "initiative_family": "Digital / EMS",
            "initiative": "EMS + submetering + analítica energética",
            "capex_eur": 100000,
            "annual_opex_saving_eur": (annual_electricity_mwh * electricity_price * 0.03) if _ok_num(annual_electricity_mwh) and _ok_num(electricity_price) else np.nan,
            "annual_co2_reduction_t": (annual_electricity_mwh * elec_factor * 0.03) if _ok_num(annual_electricity_mwh) and _ok_num(elec_factor) else np.nan,
            "implementation_months": 4,
            "strategic_score_1_5": 5,
            "activity_unit": "kWh",
        },
    )

    add_if_allowed(
        "LED",
        {
            "scope": "Scope 2",
            "emission_source": "Electricidad comprada",
            "initiative_family": "Eficiencia eléctrica",
            "initiative": "Renovación LED con control de horarios y presencia",
            "capex_eur": 60000,
            "annual_opex_saving_eur": (annual_electricity_mwh * electricity_price * 0.01) if _ok_num(annual_electricity_mwh) and _ok_num(electricity_price) else np.nan,
            "annual_co2_reduction_t": (annual_electricity_mwh * elec_factor * 0.01) if _ok_num(annual_electricity_mwh) and _ok_num(elec_factor) else np.nan,
            "implementation_months": 2,
            "strategic_score_1_5": 3,
            "activity_unit": "kWh",
        },
    )

    add_if_allowed(
        "Variadores de frecuencia",
        {
            "scope": "Scope 2",
            "emission_source": "Electricidad comprada",
            "initiative_family": "Eficiencia eléctrica",
            "initiative": "Variadores de frecuencia y sustitución selectiva de motores",
            "capex_eur": 120000,
            "annual_opex_saving_eur": (annual_electricity_mwh * electricity_price * 0.02) if _ok_num(annual_electricity_mwh) and _ok_num(electricity_price) else np.nan,
            "annual_co2_reduction_t": (annual_electricity_mwh * elec_factor * 0.02) if _ok_num(annual_electricity_mwh) and _ok_num(elec_factor) else np.nan,
            "implementation_months": 3,
            "strategic_score_1_5": 3,
            "activity_unit": "kWh",
        },
    )

    add_if_allowed(
        "Paneles solares",
        {
            "scope": "Scope 2",
            "emission_source": "Electricidad comprada",
            "initiative_family": "Renovables",
            "initiative": "Solar fotovoltaica en cubierta para autoconsumo",
            "capex_eur": 225000,
            "annual_opex_saving_eur": ((roof_area_m2 * 0.18 * 1400.0 / 1000.0 * 0.7) * electricity_price) if _ok_num(roof_area_m2) and _ok_num(electricity_price) else np.nan,
            "annual_co2_reduction_t": ((roof_area_m2 * 0.18 * 1400.0 / 1000.0 * 0.7) * elec_factor) if _ok_num(roof_area_m2) and _ok_num(elec_factor) else np.nan,
            "implementation_months": 5,
            "strategic_score_1_5": 4,
            "activity_unit": "kWh",
        },
    )

    add_if_allowed(
        "Programa de fugas de aire comprimido",
        {
            "scope": "Scope 2",
            "emission_source": "Electricidad comprada",
            "initiative_family": "Utilities",
            "initiative": "Programa de fugas de aire comprimido y optimización de presión",
            "capex_eur": 60000,
            "annual_opex_saving_eur": (annual_electricity_mwh * electricity_price * 0.01) if _ok_num(annual_electricity_mwh) and _ok_num(electricity_price) else np.nan,
            "annual_co2_reduction_t": (annual_electricity_mwh * elec_factor * 0.01) if _ok_num(annual_electricity_mwh) and _ok_num(elec_factor) else np.nan,
            "implementation_months": 2,
            "strategic_score_1_5": 2,
            "activity_unit": "kWh",
        },
    )

    if annual_fuel_mwh > 0:
        rows.append(
            {
                "scope": "Scope 1",
                "emission_source": "Combustión fija",
                "initiative_family": "Heat (Scope 1)",
                "initiative": "Mejora de eficiencia en caldera y ajuste de combustión",
                "capex_eur": 180000,
                "annual_opex_saving_eur": (annual_fuel_mwh * fuel_price * 0.05) if _ok_num(fuel_price) else np.nan,
                "annual_co2_reduction_t": (annual_fuel_mwh * fuel_factor * 0.05) if _ok_num(fuel_factor) else np.nan,
                "implementation_months": 6,
                "strategic_score_1_5": 4,
                "activity_unit": "kWh fuel",
            }
        )
        add_if_allowed(
            "Recuperación de calor",
            {
                "scope": "Scope 1",
                "emission_source": "Combustión fija",
                "initiative_family": "Heat",
                "initiative": "Recuperación de calor residual / bomba de calor industrial",
                "capex_eur": 250000,
                "annual_opex_saving_eur": (annual_fuel_mwh * fuel_price * 0.05) if _ok_num(fuel_price) else np.nan,
                "annual_co2_reduction_t": (annual_fuel_mwh * fuel_factor * 0.05) if _ok_num(fuel_factor) else np.nan,
                "implementation_months": 8,
                "strategic_score_1_5": 4,
                "activity_unit": "kWh fuel evitado",
            },
        )
        rows.append(
            {
                "scope": "Scope 1",
                "emission_source": "Combustión fija",
                "initiative_family": "Heat",
                "initiative": "Electrificación parcial del calor de proceso de baja/media temperatura",
                "capex_eur": 420000,
                "annual_opex_saving_eur": np.nan,
                "annual_co2_reduction_t": np.nan,
                "implementation_months": 7,
                "strategic_score_1_5": 4,
                "activity_unit": "kWh fuel evitado + kWh eléctricos",
            }
        )

    if any(_to_float(row.get("quantity")) > 0 for row in get_mobile_fuel_entries(company)):
        rows.append(
            {
                "scope": "Scope 1",
                "emission_source": "Flota móvil",
                "initiative_family": "Fleet",
                "initiative": "Optimización de rutas, eco-driving y mantenimiento de flota",
                "capex_eur": 25000,
                "annual_opex_saving_eur": np.nan,
                "annual_co2_reduction_t": np.nan,
                "implementation_months": 2,
                "strategic_score_1_5": 3,
                "activity_unit": "litros de combustible",
            }
        )
        add_if_allowed(
            "Flota eléctrica",
            {
                "scope": "Scope 1",
                "emission_source": "Flota móvil",
                "initiative_family": "Fleet",
                "initiative": "Electrificación parcial de flota con infraestructura de recarga",
                "capex_eur": 300000,
                "annual_opex_saving_eur": np.nan,
                "annual_co2_reduction_t": np.nan,
                "implementation_months": 12,
                "strategic_score_1_5": 4,
                "activity_unit": "litros evitados + kWh recarga",
            },
        )

    if any(_to_float(row.get("quantity")) > 0 for row in get_refrigerant_entries(company)):
        rows.append(
            {
                "scope": "Scope 1",
                "emission_source": "Refrigerantes",
                "initiative_family": "Refrigeración",
                "initiative": "Plan de control y minimización de fugas con sustitución a menor GWP donde aplique",
                "capex_eur": 80000,
                "annual_opex_saving_eur": np.nan,
                "annual_co2_reduction_t": np.nan,
                "implementation_months": 6,
                "strategic_score_1_5": 4,
                "activity_unit": "kg refrigerante",
            }
        )

    return finalize_initiatives(pd.DataFrame(rows), company, n=n)


def generate_ai_initiatives(
    company: Dict[str, Any],
    footprint: Dict[str, Any],
    api_key: str,
    model: str,
    n: int = 8,
    pestel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("No hay GEMINI_API_KEY para generar iniciativas con IA.")
    context = build_ai_company_context(company, footprint)
    emissions_context = build_emissions_input_context(company, footprint)
    research = _build_ai_web_research_context(
        company,
        footprint,
        api_key,
        model,
        "generación de iniciativas de descarbonización",
        require_web_research=True,
    )
    pestel_context = {
        str(category): [str(item) for item in items if str(item).strip()]
        for category, items in (pestel or {}).items()
        if isinstance(items, list) and any(str(item).strip() for item in items)
    }
    has_pestel_context = bool(pestel_context)
    use_web_research = True
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
        "Incluye scope, emission_source y activity_unit para que la app pueda interpretar internamente la medida. "
        "Las iniciativas deben estar adaptadas al sector, localización y realidad operativa de la empresa; evita propuestas genéricas. "
        "Antes de proponer iniciativas, realiza búsqueda web si el modelo dispone de esa herramienta. "
        "Investiga la empresa, su actividad real, productos/servicios, ubicación, sector, clientes o mercados y señales públicas recientes. "
        "Debes investigar la empresa, su ubicación, el sector, su actividad real, sus productos/servicios, la posible configuración de sus procesos, su logística y su contexto regulatorio local. "
        "Cruza esa investigación con todos los datos estructurados facilitados por el usuario para proponer medidas más precisas. "
        "Cada iniciativa debe nacer de una conexión clara entre hallazgos externos, datos de huella y viabilidad operativa; evita listas estándar de medidas. "
        "Si se facilita PESTEL_CONTEXT, trátalo como contexto estratégico previo: úsalo para convertir riesgos, oportunidades, presiones regulatorias, tendencias tecnológicas, mercado, cadena de suministro y aspectos sociales/legales en iniciativas concretas. "
        "No repitas el PESTEL: traduce sus implicaciones en medidas implantables, supuestos de CAPEX/OPEX y prioridades. "
        "Cuando estimes CAPEX, ahorro OPEX, reducción CO2 e implementación, usa lógica dimensional y de orden de magnitud realista para ese tipo de empresa. "
        "Ten en cuenta tamaño energético, consumos, combustibles, flota, cubierta disponible, calor comprado, calor de proceso si se deduce, restricciones de implantación y presupuesto. "
        "Si la base cuantitativa es insuficiente, usa null antes que inventar."
    )
    user_prompt = (
        "Genera exactamente N iniciativas en JSON (lista de objetos) siguiendo este esquema. "
        "Usa los inputs de la empresa, el contexto estructurado y supuestos conservadores. "
        "Cada iniciativa debe incluir TODAS las columnas requeridas y opcionales. "
        "Para 'scope' usa 'Alcance 1' o 'Alcance 2'. "
        "No propongas iniciativas ya implantadas (salvo si se pide explícitamente ampliación cuando son parciales). "
        "Primero busca información pública sobre la empresa y su ubicación si la herramienta de búsqueda está disponible. "
        "Usa el nombre de la empresa, la provincia, el código postal, el sector y todos los datos operativos para entender actividad, productos/servicios, procesos, plantas, logística, suministro energético y contexto industrial local, "
        "pero no inventes detalles no verificables ni cifras específicas.\n"
        "Reglas de rigor:\n"
        "- Antes de listar medidas, investiga la actividad real de la empresa, su mercado, ubicaciones, productos/servicios y señales públicas recientes; usa esa investigación para personalizar cada iniciativa.\n"
        "- Usa los datos introducidos para el cálculo de emisiones para detectar dónde están los principales consumos, focos emisores y palancas de mejora.\n"
        "- Si PESTEL_CONTEXT no está vacío, usa sus conclusiones para ajustar la selección, prioridad y estimaciones de las iniciativas; por ejemplo, riesgos energéticos, regulación local, ayudas, IA/digitalización, logística, retail/industria, cadena de suministro o actividad específica de la empresa.\n"
        "- Contrasta el PESTEL con FOOTPRINT y EMISSIONS_INPUT_CONTEXT: una oportunidad externa solo debe generar una iniciativa si tiene encaje con las fuentes emisoras reales o con datos operativos plausibles.\n"
        "- Relaciona las iniciativas con las fuentes emisoras reales observadas en combustión fija, flota, refrigerantes, electricidad y calor/vapor comprado.\n"
        f"- Debe haber exactamente {n} iniciativas finales.\n"
        "- Clasifica cada iniciativa como quick_win o estrategica.\n"
        "- Favorece estrategica frente a quick_win: usa peso 1.0 para estrategica y 0.4 para quick_win.\n"
        "- Calcula CO2 ajustado como reduccion_CO2 * peso.\n"
        "- Asegura diversidad entre eficiencia energética, electrificación, renovables/autoconsumo, mejoras operativas y suministro energético.\n"
        "- Las cifras deben ser plausibles para la escala de la empresa y consistentes entre sí.\n"
        "- El OPEX puede ser positivo, cero o negativo según el caso; no supongas siempre ahorro.\n"
        "- El ahorro OPEX debe derivar de consumos/energía/precios o quedar en null si no hay base.\n"
        "- La reducción de CO2 debe guardar relación con los consumos y factores de emisión disponibles.\n"
        "- Los meses de implementación deben reflejar complejidad real: quick wins, proyectos de ingeniería, permisos, obra e integración.\n"
        "- Devuelve únicamente los campos definidos en SCHEMA.\n\n"
        f"N = {n}\n"
        f"SCHEMA: {json.dumps(schema_note, ensure_ascii=False)}\n"
        f"WEB_RESEARCH_CONTEXT: {json.dumps(research.get('data', {}), ensure_ascii=False)}\n"
        f"COMPANY_CONTEXT: {json.dumps(context, ensure_ascii=False)}\n"
        f"EMISSIONS_INPUT_CONTEXT: {json.dumps(emissions_context, ensure_ascii=False)}\n"
        f"PESTEL_CONTEXT_AVAILABLE: {json.dumps(has_pestel_context, ensure_ascii=False)}\n"
        f"PESTEL_CONTEXT: {json.dumps(pestel_context, ensure_ascii=False)}\n"
        f"COMPANY_INPUTS: {json.dumps(company, ensure_ascii=False)}\n"
        f"FOOTPRINT: {json.dumps(footprint, ensure_ascii=False)}"
    )
    result: Dict[str, Any] | None = None
    ai_df = pd.DataFrame()
    retry_note = ""
    json_shape_hint = '[{"id":"I1","initiative":"...","scope":"Alcance 1","emission_source":"...","initiative_family":"...","categoria":"quick_win","capex_eur":0,"annual_opex_saving_eur":0,"annual_co2_reduction_t":0,"co2_adjusted_t":0,"implementation_months":0,"strategic_score_1_5":3,"activity_unit":"..."}]'
    errors: List[str] = []
    for attempt in range(1, 4):
        attempt_prompt = user_prompt + retry_note
        try:
            result = _gemini_generate_json(
                api_key,
                model,
                system_prompt,
                attempt_prompt,
                use_web_research=use_web_research,
                require_web_research=True,
                json_shape_hint=json_shape_hint,
            )
            initiative_rows = _extract_initiative_list(result["data"])
            ai_df = finalize_initiatives(pd.DataFrame(initiative_rows), company, n=n)
            if len(ai_df) == n and result.get("grounding_used"):
                break
            errors.append(
                f"Intento {attempt}: Gemini devolvió {len(ai_df)} iniciativas "
                f"y grounding_used={bool(result.get('grounding_used'))}."
            )
            retry_reason = f"devolvió {len(ai_df)} iniciativas y se necesitan exactamente {n}"
        except Exception as exc:
            errors.append(f"Intento {attempt}: {exc}")
            retry_reason = f"no devolvió una lista JSON válida ({exc})"
        retry_note = (
            "\n\nREINTENTO OBLIGATORIO:\n"
            f"El intento anterior no cumplió el contrato: {retry_reason}.\n"
            "Debes realizar búsqueda web de nuevo y devolver SOLO una lista JSON válida, sin markdown, "
            "sin explicación, sin objeto contenedor y sin claves tipo initiatives/data/items. "
            f"La respuesta debe empezar por '[' y terminar por ']'. Debe contener exactamente {n} iniciativas "
            "distintas y completas. No devuelvas menos filas ni texto fuera del JSON.\n"
        )
    if result is None or len(ai_df) != n or not result.get("grounding_used"):
        raise RuntimeError(
            "Gemini no devolvió una cartera válida tras varios reintentos. "
            "Se requieren exactamente 8 iniciativas generadas por IA con búsqueda web confirmada. "
            + " | ".join(errors)
        )
    return {
        "initiatives": ai_df,
        "grounding_used": bool(result["grounding_used"] or research.get("grounding_used")),
        "grounding_queries": list(dict.fromkeys([*research.get("grounding_queries", []), *result["grounding_queries"]])),
        "grounding_sources": [*research.get("grounding_sources", []), *result["grounding_sources"]],
    }


def compute_metrics(
    initiatives: pd.DataFrame,
    horizon_years: int,
    discount_rate: float,
) -> pd.DataFrame:
    df = normalize_columns(initiatives.copy())
    df = coerce_numeric(df)
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["total_annual_benefit_eur"] = pd.to_numeric(df["annual_opex_saving_eur"], errors="coerce").fillna(0.0)
    df["implementation_years"] = pd.to_numeric(df["implementation_months"], errors="coerce").fillna(0.0) / 12.0

    def npv_row(row: pd.Series) -> float:
        capex = row.get("capex_eur")
        benefit = row.get("total_annual_benefit_eur")
        if pd.isna(capex) or pd.isna(benefit):
            return np.nan
        if benefit <= 0:
            return -float(capex)
        delay = float(row.get("implementation_years") or 0.0)
        start_year = int(math.floor(delay)) + 1
        npv = -float(capex)
        for year in range(start_year, int(horizon_years) + 1):
            npv += float(benefit) / ((1.0 + float(discount_rate)) ** year)
        return npv

    df["npv_eur"] = df.apply(npv_row, axis=1)
    df["payback_years"] = np.where(df["total_annual_benefit_eur"] > 0, df["capex_eur"] / df["total_annual_benefit_eur"], np.nan)
    output_cols = REQUIRED_COLUMNS + OPTIONAL_COLUMNS + [
        "total_annual_benefit_eur",
        "implementation_years",
        "npv_eur",
        "payback_years",
    ]
    return df[[col for col in output_cols if col in df.columns]]


def optimize_portfolio(
    initiatives: pd.DataFrame,
    budget_eur: float,
    min_co2_t: float,
    objective: str,
    w_npv: float,
    w_co2: float,
    w_strategy: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = normalize_columns(initiatives.copy())
    defaults = {
        "capex_eur": 0.0,
        "annual_co2_reduction_t": 0.0,
        "co2_adjusted_t": 0.0,
        "npv_eur": -1e9,
        "strategic_score_1_5": 3.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)

    ids = df["id"].astype(str).tolist()
    model = pulp.LpProblem("decarbonization_portfolio", pulp.LpMaximize)
    variables = pulp.LpVariable.dicts("x", ids, lowBound=0, upBound=1, cat=pulp.LpBinary)
    model += pulp.lpSum(
        variables[i] * float(df.loc[df["id"].astype(str) == i, "capex_eur"].iloc[0]) for i in ids
    ) <= float(budget_eur)

    if min_co2_t > 0:
        model += pulp.lpSum(
            variables[i] * float(df.loc[df["id"].astype(str) == i, "annual_co2_reduction_t"].iloc[0]) for i in ids
        ) >= float(min_co2_t)

    objective_norm = str(objective).strip().lower()
    if objective_norm in {"maximizar npv", "maximize npv"}:
        model += pulp.lpSum(
            variables[i] * float(df.loc[df["id"].astype(str) == i, "npv_eur"].iloc[0]) for i in ids
        )
    elif objective_norm in {"maximizar reducción de co?", "maximizar reducción de co2", "maximize co2 reduction"}:
        model += pulp.lpSum(
            variables[i] * float(df.loc[df["id"].astype(str) == i, "annual_co2_reduction_t"].iloc[0]) for i in ids
        )
    else:
        def norm(series: Iterable[float]) -> np.ndarray:
            array = np.array(list(series), dtype=float)
            lo, hi = np.nanmin(array), np.nanmax(array)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
                return np.zeros_like(array)
            return (array - lo) / (hi - lo)

        df["npv_norm"] = norm(df["npv_eur"].values)
        df["co2_norm"] = norm(df["annual_co2_reduction_t"].values)
        df["strategy_norm"] = norm(df["strategic_score_1_5"].values)
        model += pulp.lpSum(
            variables[i]
            * (
                float(w_npv) * float(df.loc[df["id"].astype(str) == i, "npv_norm"].iloc[0])
                + float(w_co2) * float(df.loc[df["id"].astype(str) == i, "co2_norm"].iloc[0])
                + float(w_strategy) * float(df.loc[df["id"].astype(str) == i, "strategy_norm"].iloc[0])
            )
            for i in ids
        )

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = model.solve(solver)
    selected_ids = [initiative_id for initiative_id in ids if pulp.value(variables[initiative_id]) > 0.5]
    df["selected"] = df["id"].astype(str).isin(selected_ids)
    summary = {
        "status": pulp.LpStatus.get(status, str(status)),
        "total_capex": float(df.loc[df["selected"], "capex_eur"].sum()),
        "total_co2": float(df.loc[df["selected"], "annual_co2_reduction_t"].sum()),
        "total_npv": float(df.loc[df["selected"], "npv_eur"].sum()),
        "selected_count": int(df["selected"].sum()),
    }
    return df, summary
