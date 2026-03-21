from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# Replace these defaults with the official REE values you want to use.
DEFAULT_LOCATION_FACTOR = {
    2023: 0.230,
    2024: 0.220,
}

SUPPLIER_FACTORS_2024 = """
AB ENERGÍA 1903, S.L.,0.283
ACCIONA GREEN ENERGY DEVELOPMENTS SL,0.000
ACSOL ENERGÍA GLOBAL, S.A.,0.283
ADEINNOVA ENERGIA S.L,0.266
ADX Renovables, S.L.,0.283
AGRI-ENERGIA, S.A.,0.000
AHORA LUZ ENERGIA, S.L.,0.000
AHORRELUZ SERVICIOS ONLINE S.L,0.283
AIRE LIMPIO SL,0.283
ALCANZIA ENERGIA, S.L.,0.283
ALPEX IBERICA DE ENERGIA, S.L.U,0.283
ALPIQ ENERGIA ESPAÑA SAU,0.262
ALQUILER SEGURO ENERGÍA S.A.,0.282
ARACAN ENERGIA S.L.,0.283
ATENCO ENERGIA SL,0.281
ATLAS ENERGIA COMERCIAL, S.L.,0.282
AUDAX RENOVABLES, S.A,0.264
AUSARTA PRIMA, S.L.,0.283
AVANZALIA ENERGIA COMERCIALIZADORA SA,0.160
AXPO IBERIA S.L.,0.283
BIROU GAS S.L.,0.274
BON PREU, SAU,0.000
BP GAS & POWER IBERIA, S.A.U.,0.000
CANARIAS LUZ ENERGIA RENOVABLE, S.L.,0.283
CARVISA ENERGIA SL,0.278
CEPSA GAS Y ELECTRICIDAD, S.A.U.,0.279
CIDE HCENERGÍA S.A.U,0.283
CIMA ENERGIA COMERCIALIZADORA SL,0.283
COMERCIALIZADORA ADI ESPAÑA, S.L.,0.283
COMERCIALIZADORA LERSA, S.L.,0.283
CONECTA2 ENERGIA, S.L.,0.276
CREA ENERGIA ECO, S.L.U.,0.000
CYE ENERGIA SL,0.280
DAIMUZ ENERGÍA S.L.,0.283
DISA ENERGIA ELECTRICA S.L.,0.000
DRK ENERGY, S.L,0.283
EDP CLIENTES SAU,0.283
EDP ESPAÑA, S.A,0.283
ELECTRA DEL CARDENER ENERGIA, S.A.,0.229
ELECTRA ENERGIA, S.A.,0.271
ELECTRICA DE GUIXES ENERGIA, SL,0.280
ELECTRICA SEROSENSE, S.L.,0.283
ELECTRICA SOLLERENSE, S.A.,0.283
ELECTRICIDAD ELEIA S.L.,0.276
ELEK COMERCIALIZADORA ELECTRICA S.L.,0.282
EMPRESA DE ALUMBRADO ELECTRICO DE CEUTA ENERGIA, S.L.,0.283
ENDESA ENERGÍA S.A.U.,0.275
ENERGETICA HOTELERA, S.L.,0.000
ENERGÍA ECOLÓGICA ECONÓMICA, S.L.,0.283
ENERGÍA LIBRE COMERCIALIZADORA, S.L.,0.283
ENERGIA NUFRI SL,0.268
ENERGIA VIVA SPAIN, S.L.U.,0.283
ENERGÍAS DE PANTICOSA COMERCIALIZADORA, S.L.,0.283
ENERGIAS RENOVADORAS, S.L.,0.281
ENERGY STROM XXI SL,0.207
ENERGY TRADER SOLUTIONS, S.L.,0.283
ENERGYA VM GESTION DE ENERGÍA, S.L,0.283
ENERGYSAVE PROJECTS S.L,0.272
ENERPLUS ENERGIA, S.A.,0.283
ENERXIA GALEGA MAIS SLU,0.260
ENGIE ESPAÑA, S.L,0.269
ENI PLENITUDE IBERIA, S.L.,0.000
ENSTROGA, S.L.,0.283
ESCANDINAVA DE ELECTRICIDAD, S.L.U,0.283
ESTABANELL IMPULSA, S.A.U.,0.000
ESTRATEGIAS ELÉCTRICAS INTEGRALES, S.A.,0.283
EVOLVE ENERGIA, SL,0.283
FACTOR ENERGÍA ESPAÑA, S.A.,0.280
FACTOR ENERGÍA, S.A.,0.275
FENIE ENERGIA, S.A.,0.280
FOENER ENERGÍA, S.L,0.279
FORTIA ENERGIA S.L.,0.283
FOX ENERGÍA S.A,0.283
GALP ENERGÍA ESPAÑA, S.A.U.,0.282
GAS NATURAL COMERCIALIZADORA SA,0.265
GASELEC DIVERSIFICACIÓN S.L.,0.283
GASILUZ ECO ENERCIA S.L.,0.000
GEO ALTERNATIVA S.L.,0.278
GESTERNOVA, S.A,0.000
GLOBAL BIOSFERA PROTEC S.L,0.283
GLOBELIGHT ENERGY S.L,0.282
GREENING SMART ENERGY, S.L.,0.283
HANWHA ENERGY RETAIL SPAIN SL,0.000
HIDROELÉCTRICA DEL VALIRA, S.L.,0.245
IBERDROLA CLIENTES, S.A.U.,0.275
IBERDROLA SERVICIOS ENERGETICOS, S.A.U.,0.000
IBERELECTRICA COMERCIALIZADORA, SL,0.283
IM3 ENERGIA SL,0.273
INDEXO ENERGIA SL,0.000
INER ENERGIA CASTILLA LA MANCHA SL,0.278
INER EUSKADI, S.L.,0.274
INGEBAU SOLUCIONES DE MEDIDA, S.L.,0.283
INTEGRACIÓN EUROPEA DE ENERGIA, S.A.U.,0.283
INTELIGENCIA PARA EL AHORRO ENERGÉTICO, S.L.,0.271
IRIS ENERGÍA EFICIENTE S.A.,0.283
JUAN ENERGY, S.L.,0.000
LONJAS TECNOLOGÍA, S.A.,0.226
LOOP ELECTRICIDAD Y GAS, S.L,0.283
LOVE ENERGY, S.L.,0.283
LUZÍA ENERGÍA, S.L,0.276
LUZY ENERGIA RENOVABLE, S.L.,0.000
MASQLUZ 2020, S.L.,0.283
MET ENERGIA ESPAÑA, S.A,0.283
MY ENERGIA ONER S.L,0.243
NABALIA ENERGIA 2000 S.A,0.283
NATURGY CLIENTES, S.A.U.,0.000
NATURGY IBERIA, S.A.,0.278
NEOELECTRA ENERGIA,0.283
NEÓN ENERGÍA EFICIENTE, S.L,0.283
NEXUS ENERGIA SA,0.000
NIEVES ENERGÍA , S.L.,0.283
NOBE SOLUCIONES Y ENERGÍA, S.L.,0.283
NUEVA COMERCIALIZADORA ESPAÑOLA SL,0.000
OCTOPUS ENERGY ESPAÑA, S.L.U.,0.000
ON DEMAND FACILITIES, SLU,0.281
PASIÓN ENERGÍA, S.L.,0.283
PLENA ENERGIA RENOVABLE, S.L.,0.283
POTENZIA COMERCIALIZADORA SL,0.000
RECICLAJES ECOLOGICOS NAGINI, S.L.,0.283
RELUZCA ENERGÍA, S..L.,0.271
REPSOL COMERCIALIZADORA DE ELECTRICIDAD Y GAS, S.L.U,0.074
RESPIRA ENERGÍA ESPAÑA, S.L.,0.000
RESPIRA ENERGIA MEDITERRANIA, S.A.,0.000
ROFEICA ENERGIA, S.A,0.281
RONDA OESTE ENERGÍA, S.L,0.266
SAMPOL INGENIERIA Y OBRAS SA,0.000
SERVIGAS S XXI SA,0.283
SHELL ESPAÑA, S.A,0.282
SOCIEDAD ARAGONESA DE COMERCIALIZACION DE ENERGIA S.L.,0.219
SWAP ENERGIA SA,0.283
TELEFÓNICA SOLUCIONES DE INFORMÁTICA Y COMUNICACIONES DE ESPAÑA, S.A.U,0.000
TENSINA DE ENERGÍA Y SERVICIOS, S.L.,0.283
THE YELLOW ENERGY, S.L,0.283
TOTALENERGIES CLIENTES S.A.U.,0.000
TOTALENERGIES ELECTRICIDAD Y GAS ESPAÑA, S.A.U.,0.283
TOTALENERGIES MERCADO ESPAÑA, S.A.U. (EXTINGUIDA),0.260
TUNERGIA EFICIENCIA ENERGETICA Y RENOVABLE, S.L.,0.283
UNIELECTRICA ENERGIA, S.A,0.071
VISALIA ENERGIA S.L.,0.000
VIVO ENERGIA FUTURA S.A,0.283
WEKIWI, S.L.,0.268
WIND TO MARKET S.A,0.196
"""

SUPPLIER_FACTORS_2023 = """
AB ENERGÍA 1903, S.L.,0.259
ACCIONA GREEN ENERGY DEVELOPMENTS SL,0.000
ACSOL ENERGÍA GLOBAL, S.A.,0.260
ADEINNOVA ENERGIA S.L,0.260
ADX Renovables, S.L.,0.000
AGRI-ENERGIA, S.A.,0.240
AHORRELUZ SERVICIOS ONLINE S.L,0.260
AIRE LIMPIO SL,0.236
ALCANZIA ENERGIA, S.L.,0.260
ALPIQ ENERGIA ESPAÑA SAU,0.259
ARACAN ENERGIA S.L.,0.260
ATENCO ENERGIA SL,0.257
ATLAS ENERGIA COMERCIAL, S.L.,0.260
AUDAX RENOVABLES, S.A,0.000
AUSARTA PRIMA, S.L.,0.260
AVANZALIA ENERGIA COMERCIALIZADORA SA,0.140
AXPO IBERIA S.L.,0.257
BIROU GAS S.L.,0.256
BLUBAT PULSAR, S.L.,0.256
BP GAS & POWER IBERIA, S.A.U.,0.000
CEPSA GAS Y ELECTRICIDAD, S.A.U.,0.250
CIDE HCENERGÍA S.A.U.,0.249
CIMA ENERGIA COMERCIALIZADORA SL,0.260
COMERCIALIZADORA ADI ESPAÑA, S.L.,0.260
COMERCIALIZADORA LERSA, S.L.,0.260
CONECTA ENERGIA VERDE, S.L.,0.259
CONECTA2 ENERGIA, S.L.,0.258
CYE ENERGIA SL,0.258
DAIMUZ ENERGÍA S.L.,0.260
DISA ENERGIA ELECTRICA S.L.,0.237
DRK ENERGY, S.L,0.260
EBROENERGIA COMERCIALIZADORA, S.L.,0.260
ECOFUTURA LUZ ENERGÍA, S.L.U.,0.259
EDP CLIENTES SAU,0.259
EDP ESPAÑA, S.A,0.260
ELECTRA ENERGIA, S.A.,0.253
ELECTRA NORTE ENERGÍA, S.A.,0.243
ELECTRICA DE GUIXES ENERGIA, SL,0.260
ELECTRICA SEROSENSE, S.L.,0.260
ELECTRICA SOLLERENSE, S.A.,0.259
ELECTRICIDAD ELEIA S.L.,0.260
ELEK COMERCIALIZADORA ELECTRICA S.L.,0.260
EMPRESA DE ALUMBRADO ELECTRICO DE CEUTA ENERGIA, S.L.,0.260
EMPRESA DE ALUMBRADO ELECTRICO DE CEUTA, S.A.,0.260
ENDESA ENERGÍA RENOVABLE, S.L.,0.000
ENDESA ENERGÍA S.A.U.,0.259
ENDI ENERGY TRADING SL,0.253
ENERFIA, SL,0.260
ENERGÍA ECOLÓGICA ECONÓMICA, S.L.,0.260
ENERGÍA LIBRE COMERCIALIZADORA, S.L.,0.259
ENERGIA NUFRI SL,0.260
ENERGIA VIVA SPAIN, S.L.U.,0.260
ENERGY BY COGEN S.L.U.,0.260
ENERGY STROM XXI SL,0.240
ENERGYA VM GESTION DE ENERGÍA, S.L,0.258
ENERGYSAVE PROJECTS S.L,0.260
ENERPLUS ENERGIA, S.A.,0.260
ENERXIA GALEGA MAIS SLU,0.260
ENGIE ESPAÑA, S.L,0.258
ENI PLENITUDE IBERIA, S.L.,0.252
ENSTROGA, S.L.,0.252
ESCANDINAVA DE ELECTRICIDAD, S.L.U,0.259
ESTRATEGIAS ELÉCTRICAS INTEGRALES, S.A.,0.260
FACTOR ENERGÍA ESPAÑA, S.A.,0.260
FACTOR ENERGÍA, S.A.,0.259
FENIE ENERGIA, S.A.,0.250
FOENER ENERGÍA, S.L,0.260
FORTIA ENERGIA S.L.,0.260
FOX ENERGÍA S.A,0.258
GALP ENERGÍA ESPAÑA, S.A.U.,0.259
GAS NATURAL COMERCIALIZADORA SA,0.249
GASELEC DIVERSIFICACIÓN S.L.,0.259
GEO ALTERNATIVA S.L.,0.258
GESTERNOVA, S.A,0.000
GESTINER INGENIEROS, S.L.,0.260
GLOBAL BIOSFERA PROTEC S.L,0.254
GLOBELIGHT ENERGY S.L,0.254
HANWHA ENERGY RETAIL SPAIN SL,0.000
HELIOELEC ENERGIA ELECTRICA, S.L.,0.222
HIDROELÉCTRICA DEL VALIRA, S.L.,0.260
HOLALUZ-CLIDOM, S.A,0.000
IBERDROLA CLIENTES, S.A.U.,0.241
IBERDROLA SERVICIOS ENERGETICOS, S.A.U.,0.000
IM3 ENERGIA SL,0.252
INDEXO ENERGIA SL,0.233
INER ENERGIA CASTILLA LA MANCHA SL,0.257
INER EUSKADI, S.L.,0.234
INGEBAU SOLUCIONES DE MEDIDA, S.L.,0.260
INTEGRACIÓN EUROPEA DE ENERGIA, S.A.U.,0.259
INTELIGENCIA PARA EL AHORRO ENERGÉTICO, S.L.,0.248
IRIS ENERGÍA EFICIENTE S.A.,0.260
JUAN ENERGY, S.L.,0.260
LONJAS TECNOLOGÍA, S.A.,0.204
LOVE ENERGY, S.L.,0.000
LUZÍA ENERGÍA, S.L,0.259
LUZY ENERGIA RENOVABLE, S.L.,0.000
MASQLUZ 2020, S.L.,0.260
MET ENERGIA ESPAÑA, S.A,0.260
MY ENERGIA ONER S.L,0.212
NABALIA ENERGIA 2000 S.A,0.260
NATURGY CLIENTES, S.A.U.,0.000
NATURGY IBERIA, S.A.,0.215
NEOELECTRA ENERGIA,0.259
NEXUS ENERGIA SA,0.000
OCTOPUS ENERGY ESPAÑA, S.L.U.,0.000
ON DEMAND FACILITIES, SLU,0.254
PASIÓN ENERGÍA, S.L.,0.260
PETRONIEVES ENERGIA 1, S.L.,0.260
PLENA ENERGIA RENOVABLE, S.L.,0.259
POTENZIA COMERCIALIZADORA SL,0.260
RECICLAJES ECOLOGICOS NAGINI, S.L.,0.260
RELUZCA ENERGÍA, S..L.,0.258
REPSOL COMERCIALIZADORA DE ELECTRICIDAD Y GAS, S.L.U,0.259
RESPIRA ENERGÍA ESPAÑA, S.L.,0.000
RESPIRA ENERGIA MEDITERRANIA, S.A.,0.009
ROFEICA ENERGIA, S.A,0.263
RONDA OESTE ENERGÍA, S.L,0.250
SAMPOL INGENIERIA Y OBRAS SA,0.000
SERVIGAS S XXI SA,0.257
SHELL ESPAÑA, S.A,0.260
SOCIEDAD ARAGONESA DE COMERCIALIZACION DE ENERGIA S.L.,0.256
SWAP ENERGIA SA,0.260
SYDER COMERCIALIZADORA VERDE SL,0.260
TELEFÓNICA SOLUCIONES DE INFORMÁTICA Y COMUNICACIONES DE ESPAÑA, S.A.U,0.000
TENSINA DE ENERGÍA Y SERVICIOS, S.L.,0.260
THE YELLOW ENERGY, S.L,0.260
TOTALENERGIES CLIENTES S.A.U.,0.000
TOTALENERGIES ELECTRICIDAD Y GAS ESPAÑA, S.A.U.,0.249
TOTALENERGIES MERCADO ESPAÑA, S.A.U,0.247
TUNERGIA EFICIENCIA ENERGETICA Y RENOVABLE, S.L.,0.260
VIVO ENERGIA FUTURA S.A,0.260
WATIO WHOLESALE, S.L,0.260
WATIUM, S.L.,0.260
WIND TO MARKET S.A,0.254
"""


def normalize_name(name: str) -> str:
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_supplier_csv(raw: str, year: int) -> pd.DataFrame:
    rows = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        supplier, factor = line.rsplit(",", 1)
        rows.append(
            {
                "year": year,
                "supplier_name": supplier.strip(),
                "supplier_name_norm": normalize_name(supplier),
                "factor_kg_co2e_kwh": float(factor),
            }
        )
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["year", "supplier_name_norm"], keep="first")
    return df.sort_values(["year", "supplier_name"]).reset_index(drop=True)


@st.cache_data
def load_supplier_factors() -> pd.DataFrame:
    df = pd.concat(
        [
            _parse_supplier_csv(SUPPLIER_FACTORS_2023, 2023),
            _parse_supplier_csv(SUPPLIER_FACTORS_2024, 2024),
        ],
        ignore_index=True,
    )
    return df


def get_inventory_year(company_inputs: Dict | None = None) -> int:
    if company_inputs and company_inputs.get("inventory_year") is not None:
        return int(company_inputs["inventory_year"])
    if st.session_state.get("inventory_year") is not None:
        return int(st.session_state["inventory_year"])
    return 2024


def get_location_factor(year: int) -> float:
    if year in DEFAULT_LOCATION_FACTOR:
        return float(DEFAULT_LOCATION_FACTOR[year])
    return float(DEFAULT_LOCATION_FACTOR[max(DEFAULT_LOCATION_FACTOR.keys())])


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

    header_left, header_right = st.columns([1.7, 1.0])
    with header_left:
        st.markdown(
            """
            <div class="scope2-card">
                <div class="scope2-chip">Scope 2 · Electricidad comprada</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        if st.button("Añadir consumo", use_container_width=False, key="scope2_add_supplier"):
            st.session_state["scope2_supplier_count"] += 1

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
    buys_heat = st.checkbox("Compras calor o vapor", value=False, key="scope2_buys_heat")
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
