import {
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { FootprintResult } from "../../types";

const colors = ["#0d6b5f", "#2f8475", "#78a99f", "#cb6b36", "#d8a887"];

export function FootprintOverview({ footprint }: { footprint: FootprintResult | null }) {
  if (!footprint) {
    return (
      <section className="panel empty-state">
        <h3>Huella de carbono</h3>
        <p>Calcula primero la huella para visualizar el desglose y la calidad del dato.</p>
      </section>
    );
  }

  const breakdown = Object.entries(footprint.breakdown).map(([name, value]) => ({ name, value }));
  const warnings: string[] = [];

  if (footprint.refrigerant_factor_found === false && footprint.refrigerant_key) {
    warnings.push(
      `No se encontró GWP para '${footprint.refrigerant_key}'. La parte de fugitivas puede estar infraestimada.`
    );
  } else if ((footprint.refrigerant_gwp ?? 0) >= 2000) {
    warnings.push("Refrigerante con GWP alto. Considera plan de sustitución y control de fugas.");
  }

  if ((footprint.scope2_elec_t ?? 0) > 0 && (footprint.used_elec_factor ?? 0) === 0) {
    warnings.push("No hay factor eléctrico válido disponible; revisa el método y los factores.");
  }

  for (const note of footprint.scope2_notes ?? []) warnings.push(note);
  for (const error of footprint.scope2_errors ?? []) warnings.push(error);

  return (
    <section className="stack">
      <div className="hero-result panel">
        <div>
          <span className="badge">Resultado consolidado</span>
          <h2>{footprint.total_t.toFixed(1)} tCO2e/año</h2>
          <p>
            Alcance 1: {footprint.scope1_t.toFixed(1)} tCO2e · Alcance 2: {footprint.scope2_t.toFixed(1)} tCO2e
          </p>
        </div>
      </div>

      <div className="chart-grid chart-grid--results">
        <article className="panel">
          <h3>Desglose por fuente (tCO2e/año)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={breakdown} dataKey="value" nameKey="name" innerRadius={65} outerRadius={92} cy="42%">
                {breakdown.map((entry, index) => (
                  <Cell key={entry.name} fill={colors[index % colors.length]} />
                ))}
              </Pie>
              <Legend
                iconType="circle"
                verticalAlign="bottom"
                formatter={(value: string) => <span className="chart-legend-label">{value}</span>}
              />
              <Tooltip formatter={(value: number) => `${value.toFixed(2)} tCO2e`} />
            </PieChart>
          </ResponsiveContainer>
        </article>

        <article className="panel trace-panel">
          <h3>Trazabilidad de factores</h3>
          <p><strong>Alcance 1:</strong> {footprint.scope1_factor_source ?? "N/D"}</p>
          <p><strong>Electricidad:</strong> {footprint.scope2_elec_source ?? "N/D"}</p>
          <p><strong>Calor / vapor:</strong> {footprint.scope2_heat_source ?? "N/D"}</p>
        </article>
      </div>

      {warnings.length ? (
        <article className="panel stack">
          <h3>Alertas y notas técnicas</h3>
          {warnings.map((warning, index) => (
            <div key={`warning-${index}`} className="inline-banner inline-banner--warning">
              {warning}
            </div>
          ))}
        </article>
      ) : null}
    </section>
  );
}
