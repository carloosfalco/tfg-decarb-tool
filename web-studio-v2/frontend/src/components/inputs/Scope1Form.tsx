import { useMemo, useState } from "react";
import type { CatalogItem, CompanyInputs, FootprintResult } from "../../types";

const COMMON_STATIONARY_LABELS = new Set(["Gasóleo C", "Gasóleo B", "Gas natural", "LPG"]);
const COMMON_MOBILE_LABELS = ["B7", "E5"];
const HIDDEN_SECONDARY_FUEL_LABELS = new Set(["E10"]);
const COMMON_MOBILE_DISPLAY_LABELS: Record<string, string> = {
  B7: "B7 (Gasóleo)",
  E5: "E5 (Gasolina)",
};

function latestFactor(item: CatalogItem | undefined, year: 2023 | 2024 | 2025) {
  if (!item?.factors_kg_per_unit) return null;
  return item.factors_kg_per_unit[year] ?? null;
}

function sanitize(value: CompanyInputs): CompanyInputs {
  return {
    ...value,
    stationary_fuels: value.stationary_fuels.filter((item) => item.fuel_key && item.quantity > 0),
    mobile_fuels: value.mobile_fuels.filter((item) => item.fuel_key && item.quantity > 0),
    refrigerants: value.refrigerants.filter((item) => item.name && item.quantity > 0),
  };
}

function parseQuantity(raw: string) {
  if (raw.trim() === "") return 0;
  const value = Number(raw);
  return Number.isFinite(value) ? Math.max(0, value) : 0;
}

function displayQuantity(value: number | null | undefined) {
  return value && value > 0 ? String(value) : "";
}

function displayMobileFuelLabel(label: string | null | undefined) {
  const clean = String(label ?? "");
  return COMMON_MOBILE_DISPLAY_LABELS[clean] ?? clean;
}

export function Scope1Form({
  value,
  stationaryCatalog,
  mobileCatalog,
  refrigerantCatalog,
  footprint,
  onChange,
  onPrevious,
  onSubmit,
}: {
  value: CompanyInputs;
  stationaryCatalog: CatalogItem[];
  mobileCatalog: CatalogItem[];
  refrigerantCatalog: CatalogItem[];
  footprint?: FootprintResult | null;
  onChange?: (value: CompanyInputs) => void;
  onPrevious?: () => void;
  onSubmit: (value: CompanyInputs) => void;
}) {
  const [pendingMobileKeys, setPendingMobileKeys] = useState<Record<string, string>>({});
  const [pendingOtherMobileKey] = useState("");
  const [pendingRefrigerantName] = useState("");

  const commonStationary = useMemo(
    () => stationaryCatalog.filter((item) => item.label && COMMON_STATIONARY_LABELS.has(item.label)),
    [stationaryCatalog]
  );
  const otherStationary = useMemo(
    () =>
      stationaryCatalog.filter(
        (item) => item.label && !COMMON_STATIONARY_LABELS.has(item.label) && !HIDDEN_SECONDARY_FUEL_LABELS.has(item.label)
      ),
    [stationaryCatalog]
  );

  const commonMobileGroups = useMemo(() => {
    const grouped = new Map<string, CatalogItem[]>();
    for (const item of mobileCatalog) {
      if (!item.fuel_label || !COMMON_MOBILE_LABELS.includes(item.fuel_label)) continue;
      grouped.set(item.fuel_label, [...(grouped.get(item.fuel_label) ?? []), item]);
    }
    return COMMON_MOBILE_LABELS.map((label) => [label, grouped.get(label) ?? []] as const);
  }, [mobileCatalog]);

  const commonMobileKeys = useMemo(
    () => new Set(commonMobileGroups.flatMap(([, items]) => items.map((item) => item.key ?? ""))),
    [commonMobileGroups]
  );
  const otherMobile = useMemo(
    () =>
      mobileCatalog.filter(
        (item) => item.key && !commonMobileKeys.has(item.key) && !HIDDEN_SECONDARY_FUEL_LABELS.has(item.fuel_label ?? "")
      ),
    [mobileCatalog, commonMobileKeys]
  );

  const selectedStationaryKeys = value.stationary_fuels.map((item) => item.fuel_key);
  const selectedMobileKeys = value.mobile_fuels.map((item) => item.fuel_key);
  const selectedRefrigerants = value.refrigerants.map((item) => item.name);

  const getStationaryQty = (fuelKey: string) => value.stationary_fuels.find((item) => item.fuel_key === fuelKey)?.quantity ?? 0;
  const getMobileQty = (fuelKey: string) => value.mobile_fuels.find((item) => item.fuel_key === fuelKey)?.quantity ?? 0;
  const getRefrigerantQty = (name: string) => value.refrigerants.find((item) => item.name === name)?.quantity ?? 0;

  function update(next: CompanyInputs) {
    onChange?.(next);
  }

  function setStationaryQuantity(fuelKey: string, quantity: number) {
    const clean = value.stationary_fuels.filter((item) => item.fuel_key !== fuelKey);
    update({
      ...value,
      stationary_fuels: fuelKey ? [...clean, { fuel_key: fuelKey, quantity }] : clean,
    });
  }

  function setMobileQuantity(fuelKey: string, quantity: number) {
    const clean = value.mobile_fuels.filter((item) => item.fuel_key !== fuelKey);
    update({
      ...value,
      mobile_fuels: fuelKey ? [...clean, { fuel_key: fuelKey, quantity }] : clean,
    });
  }

  function setRefrigerantQuantity(name: string, quantity: number) {
    const clean = value.refrigerants.filter((item) => item.name !== name);
    update({
      ...value,
      refrigerants: name ? [...clean, { name, quantity }] : clean,
    });
  }

  function removeStationaryFuel(fuelKey: string) {
    update({ ...value, stationary_fuels: value.stationary_fuels.filter((item) => item.fuel_key !== fuelKey) });
  }

  function removeMobileFuel(fuelKey: string) {
    update({ ...value, mobile_fuels: value.mobile_fuels.filter((item) => item.fuel_key !== fuelKey) });
  }

  function removeRefrigerant(name: string) {
    update({ ...value, refrigerants: value.refrigerants.filter((item) => item.name !== name) });
  }

  function addStationaryFuel(fuelKey: string) {
    if (!fuelKey || selectedStationaryKeys.includes(fuelKey)) return;
    update({
      ...value,
      stationary_fuels: [...value.stationary_fuels, { fuel_key: fuelKey, quantity: 0 }],
    });
  }

  function addMobileFuel(fuelKey: string, fuelLabel?: string) {
    if (!fuelKey || selectedMobileKeys.includes(fuelKey)) return;
    update({
      ...value,
      mobile_fuels: [...value.mobile_fuels, { fuel_key: fuelKey, quantity: 0 }],
    });
    if (fuelLabel) {
      setPendingMobileKeys((prev) => ({ ...prev, [fuelLabel]: "" }));
    }
  }

  function addRefrigerant(name: string) {
    if (!name || selectedRefrigerants.includes(name)) return;
    update({
      ...value,
      refrigerants: [...value.refrigerants, { name, quantity: 0 }],
    });
  }

  return (
    <section className="panel stack">
      <div className="summary-grid">
        <div className="metric-card">
          <strong>{(footprint?.scope1_t ?? 0).toFixed(2)} tCO2e</strong>
          <span>Alcance 1 en vivo</span>
        </div>
        <div className="metric-card">
          <strong>{(footprint?.scope1_stationary_t ?? 0).toFixed(2)} tCO2e</strong>
          <span>Combustión fija</span>
        </div>
        <div className="metric-card">
          <strong>{(footprint?.scope1_fleet_t ?? 0).toFixed(2)} tCO2e</strong>
          <span>Combustión móvil</span>
        </div>
        <div className="metric-card">
          <strong>{(footprint?.scope1_fugitive_t ?? 0).toFixed(2)} tCO2e</strong>
          <span>Fugitivas</span>
        </div>
      </div>

      <div className="row-header scope-section-header scope-section-header--stationary">
        <div>
          <h3>Combustión fija</h3>
          <p className="muted">Los combustibles habituales se muestran directamente. El resto se añade desde la lista.</p>
        </div>
      </div>

      <div className="form-grid__two">
        {commonStationary.map((fuel) => {
          const factor = latestFactor(fuel, value.inventory_year);
          return (
            <div className="scope-card" key={`stationary-common-${fuel.key}`}>
              <label>
                <span>
                  {fuel.label} ({fuel.unit})
                </span>
                <input
                  type="number"
                  step="1000"
                  placeholder="Introduce la cifra real"
                  value={displayQuantity(getStationaryQty(fuel.key ?? ""))}
                  onChange={(event) => setStationaryQuantity(fuel.key ?? "", parseQuantity(event.target.value))}
                />
              </label>
              <div className="row-meta">
                <span>Unidad: {fuel.unit}</span>
                <span>Factor {value.inventory_year}: {factor !== null ? `${factor.toFixed(3)} kgCO2/${fuel.unit}` : "No disponible"}</span>
              </div>
            </div>
          );
        })}
      </div>

      <label>
        <span>Otros combustibles</span>
        <div className="picker-row picker-row--auto">
          <select value="" onChange={(event) => addStationaryFuel(event.target.value)}>
            <option value="">Selecciona combustible</option>
            {otherStationary
              .filter((item) => !selectedStationaryKeys.includes(item.key ?? ""))
              .map((item) => (
                <option key={item.key} value={item.key}>
                  {item.label} ({item.unit})
                </option>
              ))}
          </select>
        </div>
      </label>

      {value.stationary_fuels
        .filter((item) => !commonStationary.some((fuel) => fuel.key === item.fuel_key))
        .map((entry) => {
          const fuel = stationaryCatalog.find((item) => item.key === entry.fuel_key);
          if (!fuel) return null;
          const factor = latestFactor(fuel, value.inventory_year);
          return (
            <div className="scope-card" key={`stationary-extra-${entry.fuel_key}`}>
              <label>
                <span>
                  {fuel.label} ({fuel.unit})
                </span>
                <input
                  type="number"
                  step="1000"
                  placeholder="Introduce la cifra real"
                  value={displayQuantity(entry.quantity)}
                  onChange={(event) => setStationaryQuantity(entry.fuel_key, parseQuantity(event.target.value))}
                />
              </label>
              <div className="row-meta">
                <span>Unidad: {fuel.unit}</span>
                <span>Factor {value.inventory_year}: {factor !== null ? `${factor.toFixed(3)} kgCO2/${fuel.unit}` : "No disponible"}</span>
                <button className="text-button" type="button" onClick={() => removeStationaryFuel(entry.fuel_key)}>
                  Quitar
                </button>
              </div>
            </div>
          );
        })}

      <div className="row-header scope-section-header scope-section-header--mobile">
        <div>
          <h3>Combustión móvil (flota)</h3>
          <p className="muted">Cada vehículo se añade de forma independiente. Un consumo no afecta a otro.</p>
        </div>
      </div>

      {commonMobileGroups.map(([fuelLabel, options]) => (
        <div className="stack" key={`mobile-group-${fuelLabel}`}>
          <label>
            <span>{displayMobileFuelLabel(fuelLabel)}: seleccionar tipo de vehículo</span>
            <div className="picker-row picker-row--auto">
              <select
                value={pendingMobileKeys[fuelLabel] ?? ""}
                onChange={(event) => addMobileFuel(event.target.value, fuelLabel)}
              >
                <option value="">Selecciona tipo</option>
                {options
                  .filter((item) => !selectedMobileKeys.includes(item.key ?? ""))
                  .map((item) => (
                    <option key={item.key} value={item.key}>
                      {item.vehicle_type}
                    </option>
                  ))}
              </select>
            </div>
          </label>

          <div className="form-grid__two">
            {value.mobile_fuels
              .filter((entry) => options.some((item) => item.key === entry.fuel_key))
              .map((entry) => {
                const item = options.find((option) => option.key === entry.fuel_key);
                if (!item) return null;
                const factor = latestFactor(item, value.inventory_year);
                return (
                  <div className="scope-card" key={`mobile-common-${entry.fuel_key}`}>
                    <label>
                      <span>
                        {displayMobileFuelLabel(fuelLabel)} · {item.vehicle_type} ({item.unit})
                      </span>
                      <input
                        type="number"
                        step="1000"
                        placeholder="Introduce la cifra real"
                        value={displayQuantity(entry.quantity)}
                        onChange={(event) => setMobileQuantity(entry.fuel_key, parseQuantity(event.target.value))}
                      />
                    </label>
                    <div className="row-meta">
                      <span>Unidad: {item.unit}</span>
                      <span>Factor {value.inventory_year}: {factor !== null ? `${factor.toFixed(3)} kgCO2/${item.unit}` : "No disponible"}</span>
                      <button className="text-button" type="button" onClick={() => removeMobileFuel(entry.fuel_key)}>
                        Quitar
                      </button>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      ))}

      <label>
        <span>Otros combustibles móviles</span>
        <div className="picker-row picker-row--auto">
          <select value={pendingOtherMobileKey} onChange={(event) => addMobileFuel(event.target.value)}>
            <option value="">Selecciona combustible y tipo</option>
            {otherMobile
              .filter((item) => !selectedMobileKeys.includes(item.key ?? ""))
              .map((item) => (
                <option key={item.key} value={item.key}>
                  {displayMobileFuelLabel(item.fuel_label)} · {item.vehicle_type} ({item.unit})
                </option>
              ))}
          </select>
        </div>
      </label>

      {value.mobile_fuels
        .filter((entry) => !commonMobileKeys.has(entry.fuel_key))
        .map((entry) => {
          const item = mobileCatalog.find((option) => option.key === entry.fuel_key);
          if (!item) return null;
          const factor = latestFactor(item, value.inventory_year);
          return (
            <div className="scope-card" key={`mobile-extra-${entry.fuel_key}`}>
              <label>
                <span>
                  {displayMobileFuelLabel(item.fuel_label)} · {item.vehicle_type} ({item.unit})
                </span>
                <input
                  type="number"
                  step="1000"
                  placeholder="Introduce la cifra real"
                  value={displayQuantity(entry.quantity)}
                  onChange={(event) => setMobileQuantity(entry.fuel_key, parseQuantity(event.target.value))}
                />
              </label>
              <div className="row-meta">
                <span>Unidad: {item.unit}</span>
                <span>Factor {value.inventory_year}: {factor !== null ? `${factor.toFixed(3)} kgCO2/${item.unit}` : "No disponible"}</span>
                <button className="text-button" type="button" onClick={() => removeMobileFuel(entry.fuel_key)}>
                  Quitar
                </button>
              </div>
            </div>
          );
        })}

      <div className="row-header scope-section-header scope-section-header--fugitive">
        <div>
          <h3>Emisiones fugitivas</h3>
          <p className="muted">Cada refrigerante se añade desde lista y mantiene su cantidad de forma independiente.</p>
        </div>
      </div>

      <label>
        <span>Nombre del gas refrigerante</span>
        <div className="picker-row picker-row--auto">
          <select value={pendingRefrigerantName} onChange={(event) => addRefrigerant(event.target.value)}>
            <option value="">Selecciona refrigerante</option>
            {refrigerantCatalog
              .filter((item) => !selectedRefrigerants.includes(item.name ?? ""))
              .map((item) => (
                <option key={item.name} value={item.name}>
                  {item.name}
                </option>
              ))}
          </select>
        </div>
      </label>

      {value.refrigerants.map((entry) => {
        const item = refrigerantCatalog.find((option) => option.name === entry.name);
        if (!item) return null;
        return (
          <div className="scope-card" key={`refrigerant-${entry.name}`}>
            <label>
              <span>{entry.name} (kg recargados/año)</span>
              <input
                type="number"
                step="0.01"
                placeholder="Introduce la cifra real"
                value={displayQuantity(entry.quantity)}
                onChange={(event) => setRefrigerantQuantity(entry.name, parseQuantity(event.target.value))}
              />
            </label>
            <div className="row-meta">
              <span>Unidad: kg recargados/año</span>
              <span>PCA 6AR: {item.gwp ? item.gwp.toFixed(0) : "No disponible"}</span>
              <button className="text-button" type="button" onClick={() => removeRefrigerant(entry.name)}>
                Quitar
              </button>
            </div>
          </div>
        );
      })}

      <div className="form-actions">
        <button className="button button--secondary" type="button" onClick={onPrevious}>
          Anterior: Empresa
        </button>
        <button className="button button--primary" type="button" onClick={() => onSubmit(sanitize(value))}>
          Siguiente: Alcance 2
        </button>
      </div>
    </section>
  );
}

