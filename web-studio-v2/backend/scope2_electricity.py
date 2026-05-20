from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# Replace these defaults with the official REE values you want to use.
DEFAULT_INVENTORY_YEAR = 2025

ELECTRICITY_SUPPLIER_FACTORS_DB_PATH = Path(__file__).resolve().parent / "data" / "electricity_supplier_factors_es.csv"
ELECTRICITY_GRID_FACTORS_DB_PATH = Path(__file__).resolve().parent / "data" / "electricity_grid_factors_es.csv"

def normalize_name(name: str) -> str:
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


@st.cache_data
def load_supplier_factors() -> pd.DataFrame:
    return _load_supplier_factor_db().reset_index(drop=True)


@st.cache_data
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


def get_inventory_year(company_inputs: Dict | None = None) -> int:
    if company_inputs and company_inputs.get("inventory_year") is not None:
        return int(company_inputs["inventory_year"])
    if st.session_state.get("inventory_year") is not None:
        return int(st.session_state["inventory_year"])
    return DEFAULT_INVENTORY_YEAR


def get_location_factor(year: int) -> float:
    factors = load_grid_factors()
    if year in factors:
        return float(factors[year])
    return float(factors[max(factors.keys())])


def get_supplier_factor(year: int, supplier_name: str) -> Tuple[float | None, str | None]:
    df = load_supplier_factors()
    norm = normalize_name(supplier_name)
    match = df[(df["year"] == year) & (df["supplier_name_norm"] == norm)]
    if match.empty:
        return None, None
    row = match.iloc[0]
    return float(row["factor_kg_co2e_kwh"]), str(row["supplier_name"])


def calc_location_emissions(consumo_mwh: float, factor_ree: float) -> float:
    return max(0.0, consumo_mwh) * 1000.0 * max(0.0, factor_ree)


def calc_market_emissions_single(consumo_mwh: float, factor: float, pct_gdo: float) -> Dict:
    emisiones_brutas_kg = max(0.0, consumo_mwh) * 1000.0 * max(0.0, factor)
    pct = min(100.0, max(0.0, pct_gdo))
    emisiones_mb_kg = emisiones_brutas_kg * (1.0 - pct / 100.0)
    factor_ponderado = emisiones_brutas_kg / (consumo_mwh * 1000.0) if consumo_mwh > 0 else 0.0
    return {
        "consumo_total_mwh": max(0.0, consumo_mwh),
        "emisiones_brutas_kg": emisiones_brutas_kg,
        "emisiones_ajustadas_kg": emisiones_mb_kg,
        "factor_ponderado_kg_kwh": factor_ponderado,
    }


def calc_market_emissions_multi(rows: List[Dict], pct_gdo: float) -> Dict:
    year = get_inventory_year()
    valid_rows = []
    errors = []
    notes = []
    emisiones_mb_brutas_kg = 0.0
    consumo_total_mwh = 0.0

    for idx, row in enumerate(rows, start=1):
        supplier_name = str(row.get("supplier_name") or "").strip()
        consumo_mwh = max(0.0, float(row.get("consumo_mwh") or 0.0))

        if consumo_mwh <= 0:
            continue
        if not supplier_name:
            errors.append(f"Fila {idx}: comercializadora vacía con consumo > 0.")
            continue

        factor, pretty_name = get_supplier_factor(year, supplier_name)
        if factor is None:
            errors.append(f"No existe factor para '{supplier_name}' en la tabla del año {year}.")
            continue

        emisiones_row_kg = consumo_mwh * 1000.0 * factor
        consumo_total_mwh += consumo_mwh
        emisiones_mb_brutas_kg += emisiones_row_kg
        if factor == 0:
            notes.append(
                f"La comercializadora seleccionada '{pretty_name}' tiene factor reportado 0 kg CO2e/kWh en el año aplicado"
            )

        valid_rows.append(
            {
                "supplier_name": pretty_name,
                "consumo_mwh": consumo_mwh,
                "factor_kg_kwh": factor,
                "emisiones_kg": emisiones_row_kg,
            }
        )

    pct = min(100.0, max(0.0, pct_gdo))
    emisiones_mb_kg = emisiones_mb_brutas_kg * (1.0 - pct / 100.0)
    factor_ponderado = emisiones_mb_brutas_kg / (consumo_total_mwh * 1000.0) if consumo_total_mwh > 0 else 0.0

    return {
        "rows": valid_rows,
        "errors": errors,
        "notes": notes,
        "consumo_total_mwh": consumo_total_mwh,
        "emisiones_brutas_kg": emisiones_mb_brutas_kg,
        "emisiones_ajustadas_kg": emisiones_mb_kg,
        "factor_ponderado_kg_kwh": factor_ponderado,
    }


def _supplier_options(year: int) -> List[str]:
    df = load_supplier_factors()
    options = df[df["year"] == year]["supplier_name"].dropna().drop_duplicates().sort_values()
    return options.tolist()


def _sync_gdo_from_pct() -> None:
    total_mwh = float(st.session_state.get("scope2_gdo_total_mwh", 0.0) or 0.0)
    pct = min(100.0, max(0.0, float(st.session_state.get("scope2_gdo_pct_input", 0.0) or 0.0)))
    st.session_state["scope2_gdo_pct_input"] = pct
    st.session_state["scope2_gdo_mwh_input"] = total_mwh * pct / 100.0 if total_mwh > 0 else 0.0


def _sync_gdo_from_mwh() -> None:
    total_mwh = float(st.session_state.get("scope2_gdo_total_mwh", 0.0) or 0.0)
    gdo_mwh = max(0.0, float(st.session_state.get("scope2_gdo_mwh_input", 0.0) or 0.0))
    if total_mwh > 0:
        gdo_mwh = min(total_mwh, gdo_mwh)
        pct = gdo_mwh / total_mwh * 100.0
    else:
        gdo_mwh = 0.0
        pct = 0.0
    st.session_state["scope2_gdo_mwh_input"] = gdo_mwh
    st.session_state["scope2_gdo_pct_input"] = pct


def build_scope2_ui(company_inputs: Dict) -> Dict:
    year = get_inventory_year(company_inputs)
    factor_ree = get_location_factor(year)

    st.markdown(
        """
        <style>
        .scope2-card {
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 16px;
            background: #fbfcfd;
            padding: 0.95rem 1rem;
            margin-bottom: 0.9rem;
        }
        .scope2-chip {
            display: inline-block;
            font-size: 0.8rem;
            font-weight: 600;
            color: #1d4ed8;
            background: #eaf2ff;
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            margin-bottom: 0.45rem;
        }
        .scope2-muted {
            font-size: 0.86rem;
            color: #475569;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, header_right = st.columns([1.7, 1.0])

    method_label = st.radio(
        "Método de cálculo",
        ["Location-based", "Market-based"],
        horizontal=True,
        key="scope2_method",
    )
    method = "location" if method_label == "Location-based" else "market"
    company_inputs["electricity_method"] = method

    result = {
        "method": method,
        "consumo_total_mwh": 0.0,
        "market_emissions_kg": 0.0,
        "market_gross_emissions_kg": 0.0,
        "location_emissions_kg": 0.0,
        "market_factor_kg_kwh": 0.0,
        "location_factor_kg_kwh": factor_ree,
        "rows": [],
        "pct_gdo": 0.0,
    }

    if method == "location":
        consumo_mwh = st.number_input(
            "Electricidad comprada (MWh/año)",
            min_value=0.0,
            value=float(company_inputs.get("annual_electricity_mwh") or 0.0),
            step=100.0,
            key="scope2_lb_consumo_mwh",
        )
        st.caption(f"Se usará automáticamente el factor REE para España: {factor_ree:.3f} kg CO2e/kWh")

        emisiones_lb_kg = calc_location_emissions(consumo_mwh, factor_ree)
        result.update(
            {
                "consumo_total_mwh": consumo_mwh,
                "market_emissions_kg": emisiones_lb_kg,
                "market_gross_emissions_kg": emisiones_lb_kg,
                "location_emissions_kg": emisiones_lb_kg,
                "market_factor_kg_kwh": factor_ree,
            }
        )

        with header_right:
            if consumo_mwh > 0:
                market_equivalent_kg = emisiones_lb_kg
                st.markdown(
                    f"""
                    <div style="border:1px solid rgba(15,23,42,0.10); border-radius:14px; padding:0.9rem 1rem; background:#f8fbff;">
                        <div style="font-size:0.9rem; font-weight:600; margin-bottom:0.35rem;">Comparativa market-based</div>
                        <div style="font-size:0.85rem; color:#475569;">Equivalente market-based: {market_equivalent_kg:,.0f} kg CO2e</div>
                        <div style="font-size:0.85rem; color:#475569;">Factor REE: {factor_ree:.3f} kg CO2e/kWh</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if consumo_mwh > 0:
            st.metric("Emisiones Scope 2 electricidad (location-based)", f"{emisiones_lb_kg:,.0f} kg CO2e")

    else:
        supplier_options = _supplier_options(year)
        if "scope2_supplier_count" not in st.session_state:
            st.session_state["scope2_supplier_count"] = 1

        rows: List[Dict] = []
        for idx in range(st.session_state["scope2_supplier_count"]):
            if idx > 0:
                st.markdown("---")
            col_a, col_b = st.columns([1.6, 1.0])
            with col_a:
                supplier_name = st.selectbox(
                    "Comercializadora",
                    options=supplier_options,
                    index=None,
                    placeholder="Selecciona una comercializadora",
                    key=f"scope2_supplier_{idx}",
                )
            with col_b:
                consumo_mwh = st.number_input(
                    "Consumo (MWh)",
                    min_value=0.0,
                    value=0.0,
                    step=100.0,
                    key=f"scope2_consumo_{idx}",
                )

            if supplier_name:
                factor, pretty_name = get_supplier_factor(year, supplier_name)
                if factor is None:
                    st.error(f"No existe factor para '{supplier_name}' en la tabla del año {year}.")
                else:
                    st.caption(f"Factor {year}: {factor:.3f} kg CO2e/kWh")
                    if factor == 0:
                        st.info(
                            "La comercializadora seleccionada tiene factor reportado 0 kg CO2e/kWh en el año aplicado"
                        )
                    rows.append({"supplier_name": pretty_name, "consumo_mwh": consumo_mwh})
            elif consumo_mwh > 0:
                st.warning("Selecciona una comercializadora válida para calcular emisiones.")

            if idx == 0 and st.button("Añadir consumo", use_container_width=False, key="scope2_add_supplier"):
                st.session_state["scope2_supplier_count"] += 1
                st.rerun()

        has_gdo = st.checkbox("Tu electricidad tiene GdO", value=False, key="scope2_has_gdo")
        pct_gdo = 0.0
        gdo_mwh = 0.0
        market = calc_market_emissions_multi(rows, 0.0)
        total_mwh_for_gdo = market["consumo_total_mwh"]
        st.session_state["scope2_gdo_total_mwh"] = total_mwh_for_gdo
        if has_gdo:
            if "scope2_gdo_pct_input" not in st.session_state:
                st.session_state["scope2_gdo_pct_input"] = 0.0
            if "scope2_gdo_mwh_input" not in st.session_state:
                st.session_state["scope2_gdo_mwh_input"] = 0.0

            gdo_cols = st.columns(2)
            with gdo_cols[0]:
                st.number_input(
                    "Porcentaje con GdO (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=5.0,
                    key="scope2_gdo_pct_input",
                    on_change=_sync_gdo_from_pct,
                )
            with gdo_cols[1]:
                st.number_input(
                    "MWh con GdO",
                    min_value=0.0,
                    step=10.0,
                    key="scope2_gdo_mwh_input",
                    on_change=_sync_gdo_from_mwh,
                )
            pct_gdo = float(st.session_state.get("scope2_gdo_pct_input", 0.0) or 0.0)
            gdo_mwh = float(st.session_state.get("scope2_gdo_mwh_input", 0.0) or 0.0)
        else:
            st.session_state.pop("scope2_gdo_pct_input", None)
            st.session_state.pop("scope2_gdo_mwh_input", None)

        market = calc_market_emissions_multi(rows, pct_gdo)
        for error in market["errors"]:
            st.error(error)
        for note in market["notes"]:
            st.info(note)

        emisiones_lb_kg = calc_location_emissions(market["consumo_total_mwh"], factor_ree)
        diff_abs_kg = market["emisiones_ajustadas_kg"] - emisiones_lb_kg
        diff_pct = (diff_abs_kg / emisiones_lb_kg * 100.0) if emisiones_lb_kg > 0 else 0.0

        result.update(
            {
                "consumo_total_mwh": market["consumo_total_mwh"],
                "market_emissions_kg": market["emisiones_ajustadas_kg"],
                "market_gross_emissions_kg": market["emisiones_brutas_kg"],
                "location_emissions_kg": emisiones_lb_kg,
                "market_factor_kg_kwh": market["factor_ponderado_kg_kwh"],
                "rows": market["rows"],
                "pct_gdo": pct_gdo,
            }
        )

        with header_right:
            if market["consumo_total_mwh"] > 0:
                st.markdown(
                    f"""
                    <div style="border:1px solid rgba(15,23,42,0.10); border-radius:14px; padding:0.9rem 1rem; background:#f8fbff;">
                        <div style="font-size:0.9rem; font-weight:600; margin-bottom:0.35rem;">Comparativa location-based</div>
                        <div style="font-size:0.85rem; color:#475569;">Location-based: {emisiones_lb_kg:,.0f} kg CO2e</div>
                        <div style="font-size:0.85rem; color:#475569;">Diferencia absoluta: {diff_abs_kg:,.0f} kg CO2e</div>
                        <div style="font-size:0.85rem; color:#475569;">Diferencia porcentual: {diff_pct:,.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


    st.divider()
    buys_heat = st.checkbox("Se compra calor o vapor", value=False, key="scope2_buys_heat")
    if buys_heat:
        st.markdown("**Calor/vapor comprado**")
        heat_a, heat_b = st.columns(2)
        with heat_a:
            company_inputs["annual_purchased_heat_mwh"] = st.number_input(
                "Calor/vapor comprado (MWh/año)",
                min_value=0.0,
                value=float(company_inputs.get("annual_purchased_heat_mwh") or 0.0),
                step=50.0,
                key="scope2_heat_mwh",
            )
        with heat_b:
            company_inputs["co2_factor_heat_t_per_mwh"] = st.number_input(
                "Factor CO2 calor/vapor (tCO2/MWh) [opcional]",
                min_value=0.0,
                value=float(company_inputs.get("co2_factor_heat_t_per_mwh") or 0.0),
                step=0.01,
                key="scope2_heat_factor",
            )
    else:
        company_inputs["annual_purchased_heat_mwh"] = 0.0
        company_inputs["co2_factor_heat_t_per_mwh"] = 0.0

    company_inputs["annual_electricity_mwh"] = result["consumo_total_mwh"]
    company_inputs["supplier_name"] = result["rows"][0]["supplier_name"] if result["rows"] else ""
    company_inputs["scope2_supplier_rows"] = result["rows"]
    company_inputs["electricity_has_gdo"] = result["pct_gdo"] > 0
    company_inputs["gdo_coverage_pct"] = result["pct_gdo"]
    company_inputs["gdo_coverage_kwh"] = gdo_mwh * 1000.0 if method == "market" else 0.0
    company_inputs["electricity_gdo_type"] = "renewable" if result["pct_gdo"] > 0 else None
    company_inputs["cnmc_supplier_known"] = len(result["rows"]) > 0
    company_inputs["supplier_factor_t_per_mwh"] = result["market_factor_kg_kwh"]
    company_inputs["scope2_market_emissions_kg"] = result["market_emissions_kg"]
    company_inputs["scope2_market_gross_emissions_kg"] = result["market_gross_emissions_kg"]
    company_inputs["scope2_location_emissions_kg"] = result["location_emissions_kg"]
    company_inputs["scope2_location_factor_kg_kwh"] = result["location_factor_kg_kwh"]
    company_inputs["scope2_market_factor_kg_kwh"] = result["market_factor_kg_kwh"]

    return result
