import { useEffect, useMemo, useRef, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  Tooltip,
  XAxis,
  YAxis,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import type { FinancialParams, Initiative, PortfolioResult } from "../../types";

type OptimizeParams = {
  financialParams: FinancialParams;
  budget: number;
  minCo2: number;
};

const COLORS = ["#0d6b5f", "#2f80ed", "#cb6b36", "#6d5bd0", "#0f9b9a", "#d49b16", "#4c8b3b", "#7d5a9f"];

function formatCurrency(value: number | null | undefined) {
  if (!Number.isFinite(Number(value))) return "N/D";
  return `${Number(value).toLocaleString("es-ES", { maximumFractionDigits: 0 })} €`;
}

function formatNumber(value: number | null | undefined, digits = 1) {
  if (!Number.isFinite(Number(value))) return "N/D";
  return Number(value).toLocaleString("es-ES", { maximumFractionDigits: digits });
}

function costPerTon(initiative: Initiative) {
  const co2 = Number(initiative.annual_co2_reduction_t ?? 0);
  if (co2 <= 0) return null;
  return Number(initiative.capex_eur ?? 0) / co2;
}

function displayCategory(initiative: Initiative) {
  const family = String(initiative.initiative_family || "").trim();
  const rawCategory = String(initiative.categoria || "").trim();
  if (family) return family;
  if (rawCategory === "quick_win") return "Quick win";
  if (rawCategory === "estrategica") return "Estratégica";
  return rawCategory || "Sin categoría";
}

function finiteOr(value: number, fallback: number) {
  return Number.isFinite(value) ? value : fallback;
}

function PortfolioBarTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload?: { name?: string; co2?: number; capex?: number } }>;
}) {
  const data = payload?.[0]?.payload;
  if (!active || !data) return null;
  return (
    <div className="chart-tooltip">
      <strong>{data.name}</strong>
      <span>{formatNumber(data.co2)} tCO₂e/año</span>
    </div>
  );
}

function PortfolioPieTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload?: { name?: string; capex?: number } }>;
}) {
  const data = payload?.[0]?.payload;
  if (!active || !data) return null;
  return (
    <div className="chart-tooltip">
      <strong>{data.name}</strong>
      <span>CAPEX: {formatCurrency(data.capex)}</span>
    </div>
  );
}

function toCsv(rows: Initiative[]) {
  const headers = [
    "id",
    "initiative",
    "categoria",
    "capex_eur",
    "annual_co2_reduction_t",
    "annual_opex_saving_eur",
    "payback_years",
    "npv_eur",
    "selected",
  ];
  const lines = [
    headers.join(","),
    ...rows.map((row) =>
      headers
        .map((header) => {
          const value =
            header === "categoria"
              ? displayCategory(row)
              : String((row as unknown as Record<string, unknown>)[header] ?? "");
          return `"${String(value).replace(/"/g, '""')}"`;
        })
        .join(",")
    ),
  ];
  return lines.join("\n");
}

export function PortfolioOptimizer({
  initiatives,
  financialParams,
  result,
  isLoading,
  onFinancialChange,
  onOptimize,
}: {
  initiatives: Initiative[];
  financialParams: FinancialParams;
  result: PortfolioResult | null;
  isLoading: boolean;
  onFinancialChange: (value: FinancialParams) => void;
  onOptimize: (params: OptimizeParams) => Promise<void>;
}) {
  const [budget, setBudget] = useState(500000);
  const [minCo2, setMinCo2] = useState(0);
  const [hasRequestedOptimization, setHasRequestedOptimization] = useState(false);
  const [lastError, setLastError] = useState("");
  const debounceRef = useRef<number | null>(null);

  const selected = useMemo(() => (result?.initiatives ?? []).filter((item) => item.selected), [result]);
  const selectedOpex = selected.reduce((sum, item) => sum + Number(item.annual_opex_saving_eur ?? 0), 0);
  const selectedPaybacks = selected
    .map((item) => Number(item.payback_years))
    .filter((value) => Number.isFinite(value) && value >= 0);
  const maxPayback = selectedPaybacks.length ? Math.max(...selectedPaybacks) : null;
  const budgetUsedPct = budget > 0 && result ? Math.min(100, (result.summary.total_capex / budget) * 100) : 0;

  const selectedChartData = selected.map((item, index) => ({
    id: item.id,
    name: item.initiative,
    capex: Number(item.capex_eur ?? 0),
    co2: Number(item.annual_co2_reduction_t ?? 0),
    fill: COLORS[index % COLORS.length],
  }));

  const exportHref = useMemo(() => {
    if (!selected.length) return "";
    const blob = new Blob([toCsv(selected)], { type: "text/csv;charset=utf-8;" });
    return URL.createObjectURL(blob);
  }, [selected]);

  const infeasible =
    Boolean(result) &&
    (result?.summary.status !== "Optimal" || result.summary.selected_count === 0) &&
    (minCo2 > 0 || budget > 0);

  function runOptimization(markAsRequested = true) {
    if (!initiatives.length || isLoading) return Promise.resolve();
    if (markAsRequested) setHasRequestedOptimization(true);
    setLastError("");
    return onOptimize({
      financialParams,
      budget: finiteOr(Number(budget), 0),
      minCo2: finiteOr(Number(minCo2), 0),
    }).catch((error: unknown) => {
      setLastError(error instanceof Error ? error.message : "No se pudo optimizar el portafolio.");
    });
  }

  useEffect(() => {
    if (!hasRequestedOptimization || !initiatives.length) return;
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => {
      void runOptimization(false);
    }, 450);
    return () => {
      if (debounceRef.current) window.clearTimeout(debounceRef.current);
    };
  }, [
    hasRequestedOptimization,
    initiatives,
    budget,
    minCo2,
    financialParams.discount_rate,
    financialParams.horizon,
    financialParams.max_payback_years,
  ]);

  return (
    <section className="portfolio-module" id="portfolio-top">
      <header className="panel portfolio-header">
        <div>
          <span className="badge">Decisión final</span>
          <h2>Optimización del portafolio de iniciativas</h2>
          <p>
            Selecciona la combinación de medidas más adecuada según el presupuesto disponible, la reducción mínima
            deseada y los supuestos financieros definidos.
          </p>
        </div>
      </header>

      {!initiatives.length ? (
        <section className="panel empty-state">
          <p>Primero debes generar iniciativas de descarbonización antes de optimizar el portafolio.</p>
        </section>
      ) : (
        <>
          <section className="panel portfolio-controls">
            <div className="portfolio-controls__head">
              <div>
                <h3>Supuestos financieros y restricciones</h3>
                <p>Estos valores se aplican al cálculo de métricas y a la selección optimizada.</p>
              </div>
              <button className="button button--primary portfolio-button" type="button" onClick={() => void runOptimization()} disabled={isLoading}>
                {isLoading ? "Optimizando portafolio..." : "Optimizar portafolio"}
              </button>
            </div>

            <div className="portfolio-input-grid">
              <label>
                <span>Tasa de descuento (%)</span>
                <small>Tasa utilizada para calcular métricas financieras.</small>
                <input
                  type="number"
                  step="0.1"
                  value={Number((financialParams.discount_rate * 100).toFixed(2))}
                  onChange={(event) =>
                    onFinancialChange({
                      ...financialParams,
                      discount_rate: finiteOr(Number(event.target.value) / 100, 0),
                    })
                  }
                />
              </label>
              <label>
                <span>Horizonte temporal (años)</span>
                <small>Periodo de evaluación financiera.</small>
                <input
                  type="number"
                  min="1"
                  value={financialParams.horizon}
                  onChange={(event) =>
                    onFinancialChange({ ...financialParams, horizon: Math.max(1, finiteOr(Number(event.target.value), 1)) })
                  }
                />
              </label>
              <label>
                <span>Presupuesto máximo / CAPEX máximo (€)</span>
                <small>CAPEX máximo disponible para seleccionar iniciativas.</small>
                <input
                  type="number"
                  step="10000"
                  value={budget}
                  onChange={(event) => setBudget(finiteOr(Number(event.target.value), 0))}
                />
              </label>
              <label>
                <span>Objetivo mínimo de reducción anual de CO₂e</span>
                <small>Reducción anual mínima deseada de emisiones.</small>
                <input type="number" value={minCo2} onChange={(event) => setMinCo2(finiteOr(Number(event.target.value), 0))} />
              </label>
              <label>
                <span>Payback máximo aceptable</span>
                <small>Periodo máximo de recuperación aceptado.</small>
                <input
                  type="number"
                  step="0.1"
                  value={financialParams.max_payback_years ?? ""}
                  placeholder="Sin límite"
                  onChange={(event) =>
                    onFinancialChange({
                      ...financialParams,
                      max_payback_years: event.target.value === "" ? null : Number(event.target.value),
                    })
                  }
                />
              </label>
            </div>
          </section>

          {lastError ? (
            <section className="panel inline-banner inline-banner--warning">
              <p>{lastError}</p>
            </section>
          ) : null}

          {!result && !hasRequestedOptimization ? null : infeasible ? (
            <section className="panel inline-banner inline-banner--warning">
              <p>
                No se ha encontrado una combinación de iniciativas que cumpla las restricciones introducidas. Prueba a
                aumentar el presupuesto máximo, reducir el objetivo mínimo de CO₂e o ampliar el payback máximo aceptable.
              </p>
            </section>
          ) : result ? (
            <section className="portfolio-results">
              <div className="portfolio-results__head">
                <div>
                  <h3>Portafolio optimizado</h3>
                  <p>Selección recomendada según los supuestos y restricciones introducidos.</p>
                </div>
                {exportHref ? (
                  <a className="button button--secondary" href={exportHref} download="portfolio_seleccionado.csv">
                    Exportar selección
                  </a>
                ) : null}
              </div>

              <div className="portfolio-summary-grid">
                <div className="metric-card">
                  <strong>{result.summary.selected_count}</strong>
                  <span>Iniciativas seleccionadas</span>
                </div>
                <div className="metric-card">
                  <strong>{formatCurrency(result.summary.total_capex)}</strong>
                  <span>Inversión total seleccionada</span>
                </div>
                <div className="metric-card">
                  <strong>{formatNumber(result.summary.total_co2)} tCO₂e</strong>
                  <span>Reducción anual total estimada</span>
                </div>
                <div className="metric-card">
                  <strong>{formatCurrency(selectedOpex)}</strong>
                  <span>Ahorro anual estimado</span>
                </div>
                <div className="metric-card">
                  <strong>{maxPayback === null ? "N/D" : `${formatNumber(maxPayback, 1)} años`}</strong>
                  <span>Payback máximo seleccionado</span>
                </div>
                <div className="metric-card">
                  <strong>{formatCurrency(result.summary.total_npv)}</strong>
                  <span>VAN total del portafolio</span>
                </div>
              </div>

              <section className="panel budget-usage">
                <div>
                  <strong>Presupuesto usado</strong>
                  <span>
                    {formatCurrency(result.summary.total_capex)} de {formatCurrency(budget)}
                  </span>
                </div>
                <div className="budget-usage__bar" aria-hidden="true">
                  <span style={{ width: `${budgetUsedPct}%` }} />
                </div>
              </section>

              <section className="panel selected-table-panel">
                <h3>Iniciativas seleccionadas</h3>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Iniciativa</th>
                        <th>Categoría</th>
                        <th>CAPEX</th>
                        <th>Reducción anual CO₂e</th>
                        <th>Ahorro anual</th>
                        <th>Payback</th>
                        <th>VAN</th>
                        <th>€/tCO₂e evitada</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.map((initiative) => (
                        <tr key={initiative.id}>
                          <td>{initiative.initiative}</td>
                          <td>{displayCategory(initiative)}</td>
                          <td>{formatCurrency(initiative.capex_eur)}</td>
                          <td>{formatNumber(initiative.annual_co2_reduction_t)} t</td>
                          <td>{formatCurrency(initiative.annual_opex_saving_eur)}</td>
                          <td>
                            {Number.isFinite(Number(initiative.payback_years))
                              ? `${formatNumber(initiative.payback_years, 1)} años`
                              : "N/D"}
                          </td>
                          <td>{formatCurrency(initiative.npv_eur)}</td>
                          <td>{costPerTon(initiative) === null ? "N/D" : `${formatCurrency(costPerTon(initiative))}/t`}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="portfolio-chart-grid">
                <article className="panel">
                  <h3>Reducción anual de CO₂e por iniciativa</h3>
                  <p className="muted">Unidad del eje vertical: tCO₂e/año. Cada barra es una iniciativa seleccionada.</p>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={selectedChartData}>
                      <XAxis dataKey="id" label={{ value: "Iniciativa seleccionada", position: "insideBottom", offset: -4 }} />
                      <YAxis
                        width={70}
                        tickFormatter={(value) => formatNumber(Number(value), 0)}
                        label={{ value: "tCO₂e/año", angle: -90, position: "insideLeft" }}
                      />
                      <Tooltip content={<PortfolioBarTooltip />} />
                      <Bar dataKey="co2" fill="#0d6b5f" />
                    </BarChart>
                  </ResponsiveContainer>
                </article>
                <article className="panel">
                  <h3>Distribución del CAPEX</h3>
                  <p className="muted">Unidad del gráfico: euros de CAPEX por iniciativa.</p>
                  <ResponsiveContainer width="100%" height={260}>
                    <PieChart>
                      <Pie data={selectedChartData} dataKey="capex" nameKey="name" innerRadius={52} outerRadius={92}>
                        {selectedChartData.map((entry) => (
                          <Cell key={entry.id} fill={entry.fill} />
                        ))}
                      </Pie>
                      <Tooltip content={<PortfolioPieTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                </article>
              </section>
            </section>
          ) : null}
        </>
      )}
    </section>
  );
}
