import { useMemo, useState } from "react";
import type { CatalogItem, CompanyInputs, FootprintResult } from "../../types";

const LOCATION_FACTORS: Record<2023 | 2024 | 2025, number> = {
  2023: 0.120,
  2024: 0.103,
  2025: 0.108,
};

function normalizeScope2(value: CompanyInputs) {
  const supplierRows = value.scope2_supplier_rows.filter((row) => row.supplier_name || row.consumo_mwh > 0);
  const supplierTotal = supplierRows.reduce((sum, row) => sum + (row.consumo_mwh || 0), 0);
  return {
    ...value,
    scope2_supplier_rows: supplierRows,
    annual_electricity_mwh: value.electricity_method === "market" ? supplierTotal : value.annual_electricity_mwh,
    supplier_name: supplierRows.find((row) => row.supplier_name)?.supplier_name ?? "",
    gdo_coverage_pct: Math.min(100, Math.max(0, value.gdo_coverage_pct || 0)),
  };
}

export function Scope2Form({
  value,
  supplierCatalog,
  footprint,
  onChange,
  onPrevious,
  onSubmit,
}: {
  value: CompanyInputs;
  supplierCatalog: CatalogItem[];
  footprint?: FootprintResult | null;
  onChange?: (value: CompanyInputs) => void;
  onPrevious?: () => void;
  onSubmit: (value: CompanyInputs) => void;
}) {
  const [pendingSupplier, setPendingSupplier] = useState("");

  const locationFactor = LOCATION_FACTORS[value.inventory_year];
  const supplierTotal = useMemo(
    () => value.scope2_supplier_rows.reduce((sum, row) => sum + (row.consumo_mwh || 0), 0),
    [value.scope2_supplier_rows]
  );
  const marketFactor = useMemo(() => {
    if (!supplierTotal) return 0;
    const weighted = value.scope2_supplier_rows.reduce((sum, row) => {
      const factor = supplierCatalog.find((item) => item.name === row.supplier_name)?.factor_kg_co2e_kwh ?? 0;
      return sum + factor * row.consumo_mwh;
    }, 0);
    return weighted / supplierTotal;
  }, [supplierCatalog, supplierTotal, value.scope2_supplier_rows]);

  const locationT = (value.electricity_method === "market" ? supplierTotal : value.annual_electricity_mwh) * locationFactor;
  const marketT = supplierTotal * marketFactor * (1 - (value.gdo_coverage_pct || 0) / 100);
  const gdoMwh = (supplierTotal * (value.gdo_coverage_pct || 0)) / 100;
  const differenceT = marketT - locationT;
  const hasSupplierRows = value.scope2_supplier_rows.some((row) => row.supplier_name);

  function update(next: CompanyInputs) {
    onChange?.(normalizeScope2(next));
  }

  function addSupplier(name: string) {
    if (!name || value.scope2_supplier_rows.some((row) => row.supplier_name === name)) return;
    update({
      ...value,
      scope2_supplier_rows: [...value.scope2_supplier_rows, { supplier_name: name, consumo_mwh: 0.01 }],
    });
    setPendingSupplier("");
  }

  return (
    <section className="panel stack">
      <div className="summary-grid">
        <div className="metric-card">
          <strong>{(footprint?.scope2_t ?? 0).toFixed(2)} tCO2e</strong>
          <span>Alcance 2 en vivo</span>
        </div>
        <div className="metric-card">
          <strong>{(footprint?.scope2_elec_t ?? 0).toFixed(2)} tCO2e</strong>
          <span>Electricidad</span>
        </div>
        <div className="metric-card">
          <strong>{(footprint?.scope2_heat_t ?? 0).toFixed(2)} tCO2e</strong>
          <span>Calor / vapor</span>
        </div>
        <div className="metric-card">
          <strong>{(footprint?.total_t ?? 0).toFixed(2)} tCO2e</strong>
          <span>Total inventario</span>
        </div>
      </div>

      <div className={value.electricity_method === "location" ? "scope2-method-grid" : "form-grid__two"}>
        <label>
          <span>Método de cálculo</span>
          <select
            value={value.electricity_method}
            onChange={(event) => update({ ...value, electricity_method: event.target.value as "location" | "market" })}
          >
            <option value="location">Location-based</option>
            <option value="market">Market-based</option>
          </select>
        </label>
        {value.electricity_method === "location" ? (
          <>
            <label>
              <span>Electricidad comprada</span>
              <div className="input-with-unit">
                <input
                  type="number"
                  step="100"
                  value={value.annual_electricity_mwh}
                  onChange={(event) => update({ ...value, annual_electricity_mwh: Number(event.target.value) })}
                />
                <small>MWh/año</small>
              </div>
            </label>
            <label>
              <span>Factor REE {value.inventory_year}</span>
              <input value={`${locationFactor.toFixed(3)} kg CO2e/kWh`} disabled />
            </label>
          </>
        ) : null}
      </div>

      {footprint?.scope2_errors?.length ? (
        <div className="inline-banner inline-banner--warning">
          {footprint.scope2_errors.map((error) => (
            <p key={error}>{error}</p>
          ))}
        </div>
      ) : null}

      {value.electricity_method === "market" ? (
        <>
          <div className="row-header">
            <div>
              <h3>Comercializadoras</h3>
              <p className="muted">
                Añade las comercializadoras conocidas. Si no se conocen todos los proveedores del ejercicio, puedes usar el
                factor market-based genérico para el consumo no identificado.
              </p>
            </div>
          </div>

          <label>
            <span>Seleccionar comercializadora o factor genérico</span>
            <div className="picker-row picker-row--auto">
              <select
                value={pendingSupplier}
                onChange={(event) => addSupplier(event.target.value)}
              >
                <option value="">Selecciona comercializadora o factor genérico</option>
                {supplierCatalog
                  .filter((item) => !value.scope2_supplier_rows.some((row) => row.supplier_name === item.name))
                  .map((item) => (
                    <option key={item.name} value={item.name}>
                      {item.name}
                    </option>
                  ))}
              </select>
            </div>
          </label>

          {!hasSupplierRows ? (
            <div className="empty-state">
              Añade una comercializadora conocida o el factor market-based genérico para activar el cálculo y comparar contra location-based.
            </div>
          ) : null}

          {value.scope2_supplier_rows.map((row) => {
            const supplier = supplierCatalog.find((item) => item.name === row.supplier_name);
            if (!row.supplier_name) return null;
            return (
              <div className="scope-card" key={`supplier-${row.supplier_name}`}>
                <label>
                  <span>{row.supplier_name}</span>
                  <input
                    type="number"
                    step="100"
                    value={row.consumo_mwh}
                    onChange={(event) =>
                      update({
                        ...value,
                        scope2_supplier_rows: value.scope2_supplier_rows.map((item) =>
                          item.supplier_name === row.supplier_name
                            ? { ...item, consumo_mwh: Number(event.target.value) }
                            : item
                        ),
                      })
                    }
                  />
                </label>
                <div className="row-meta">
                  <span>Consumo imputado: {row.consumo_mwh.toFixed(2)} MWh</span>
                  <span>Factor proveedor: {supplier?.factor_kg_co2e_kwh?.toFixed(3) ?? "No disponible"} kg CO2e/kWh</span>
                  <button
                    className="text-button"
                    type="button"
                    onClick={() =>
                      update({
                        ...value,
                        scope2_supplier_rows: value.scope2_supplier_rows.filter((item) => item.supplier_name !== row.supplier_name),
                      })
                    }
                  >
                    Quitar
                  </button>
                </div>
              </div>
            );
          })}

          <div className="form-grid__two">
            <label>
              <span>Porcentaje cubierto por GdO</span>
              <div className="input-with-unit">
                <input
                  type="number"
                  step="1"
                  min="0"
                  max="100"
                  value={value.gdo_coverage_pct}
                  onChange={(event) => update({ ...value, gdo_coverage_pct: Number(event.target.value) })}
                />
                <small>%</small>
              </div>
            </label>
            <label>
              <span>MWh cubiertos por GdO</span>
              <div className="input-with-unit">
                <input
                  type="number"
                  step="100"
                  min="0"
                  value={gdoMwh}
                  onChange={(event) =>
                    update({
                      ...value,
                      gdo_coverage_pct: supplierTotal > 0 ? Math.min(100, (Number(event.target.value) / supplierTotal) * 100) : 0,
                    })
                  }
                />
                <small>MWh</small>
              </div>
            </label>
          </div>

          <div className="comparison-strip comparison-strip--three">
            <div className="comparison-card">
              <span>Location-based equivalente</span>
              <strong>{locationT.toFixed(1)} tCO2e</strong>
            </div>
            <div className="comparison-card">
              <span>Market-based estimado</span>
              <strong>{marketT.toFixed(1)} tCO2e</strong>
            </div>
            <div className="comparison-card">
              <span>Diferencia</span>
              <strong className={differenceT <= 0 ? "is-positive" : "is-negative"}>{differenceT.toFixed(1)} tCO2e</strong>
            </div>
          </div>

          {footprint?.scope2_notes?.length ? (
            <div className="inline-banner">
              {footprint.scope2_notes.map((note) => (
                <p key={note}>{note}</p>
              ))}
            </div>
          ) : null}
        </>
      ) : null}

      <div className="form-grid__two">
        <label>
          <span>Calor o vapor comprado</span>
          <div className="input-with-unit">
            <input
              type="number"
              step="1000"
              value={value.annual_purchased_heat_mwh}
              onChange={(event) => update({ ...value, annual_purchased_heat_mwh: Number(event.target.value) })}
            />
            <small>MWh/año</small>
          </div>
        </label>
        <label>
          <span>Factor CO2 calor/vapor</span>
          <div className="input-with-unit">
            <input
              type="number"
              step="0.01"
              value={value.co2_factor_heat_t_per_mwh}
              onChange={(event) => update({ ...value, co2_factor_heat_t_per_mwh: Number(event.target.value) })}
            />
            <small>tCO2/MWh</small>
          </div>
        </label>
      </div>

      <div className="form-actions">
        <button className="button button--secondary" type="button" onClick={onPrevious}>
          Anterior: Alcance 1
        </button>
        <button className="button button--primary" type="button" onClick={() => onSubmit(normalizeScope2(value))}>
          Siguiente: PESTEL
        </button>
      </div>
    </section>
  );
}

