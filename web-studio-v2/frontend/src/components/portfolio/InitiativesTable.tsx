import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useMemo } from "react";
import type { AiGenerationMeta, Initiative } from "../../types";

const columnHelper = createColumnHelper<Initiative>();

function numberInputValue(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? String(numeric) : "";
}

export function InitiativesTable({
  data,
  meta,
  error,
  isLoading = false,
  onChange,
  onGenerateAi,
}: {
  data: Initiative[];
  meta: AiGenerationMeta | null;
  error: string | null;
  isLoading?: boolean;
  onChange: (rows: Initiative[]) => void;
  onGenerateAi: () => void;
}) {
  const columns = useMemo(
    () => [
      columnHelper.accessor("scope", { header: "Scope" }),
      columnHelper.accessor("initiative_family", { header: "Familia" }),
      columnHelper.accessor("initiative", { header: "Iniciativa" }),
      columnHelper.accessor("capex_eur", {
        header: "CAPEX (€)",
        cell: (info) => (
          <input
            className="numeric-input"
            type="number"
            value={numberInputValue(info.getValue())}
            onChange={(event) => {
              const next = [...data];
              next[info.row.index] = { ...next[info.row.index], capex_eur: Number(event.target.value) };
              onChange(next);
            }}
          />
        ),
      }),
      columnHelper.accessor("annual_opex_saving_eur", {
        header: "Ahorro OPEX (€/año)",
        cell: (info) => (
          <input
            className="numeric-input"
            type="number"
            value={numberInputValue(info.getValue())}
            onChange={(event) => {
              const next = [...data];
              next[info.row.index] = {
                ...next[info.row.index],
                annual_opex_saving_eur: Number(event.target.value),
              };
              onChange(next);
            }}
          />
        ),
      }),
      columnHelper.accessor("annual_co2_reduction_t", {
        header: "CO₂ evitado (t/año)",
        cell: (info) => (
          <input
            className="numeric-input"
            type="number"
            value={numberInputValue(info.getValue())}
            onChange={(event) => {
              const next = [...data];
              next[info.row.index] = {
                ...next[info.row.index],
                annual_co2_reduction_t: Number(event.target.value),
              };
              onChange(next);
            }}
          />
        ),
      }),
      columnHelper.accessor("implementation_months", {
        header: "Meses",
        cell: (info) => (
          <input
            className="numeric-input"
            type="number"
            value={numberInputValue(info.getValue())}
            onChange={(event) => {
              const next = [...data];
              next[info.row.index] = {
                ...next[info.row.index],
                implementation_months: Number(event.target.value),
              };
              onChange(next);
            }}
          />
        ),
      }),
      columnHelper.accessor("strategic_score_1_5", {
        header: "Encaje estratégico (1-5)",
        cell: (info) => (
          <input
            className="numeric-input"
            type="number"
            min="1"
            max="5"
            value={numberInputValue(info.getValue() ?? 3)}
            onChange={(event) => {
              const next = [...data];
              next[info.row.index] = {
                ...next[info.row.index],
                strategic_score_1_5: Number(event.target.value),
              };
              onChange(next);
            }}
          />
        ),
      }),
    ],
    [data, onChange]
  );

  const table = useReactTable({ data, columns, getCoreRowModel: getCoreRowModel() });

  return (
    <section className="stack">
      <div className="panel action-bar">
        <span className="muted">Genera o actualiza la cartera; después puedes editar las hipótesis inline. Encaje estratégico (1-5) mide prioridad y ajuste con la empresa.</span>
        <button className="button button--primary" type="button" onClick={onGenerateAi} disabled={isLoading}>
          {isLoading ? <span className="spinner" aria-hidden="true" /> : null}
          {isLoading ? "Generando..." : "Generar con IA"}
        </button>
      </div>
      {isLoading ? (
        <div className="panel ai-status ai-status--loading ai-status--prominent">
          <span className="spinner spinner--large" aria-hidden="true" />
          <div>
            <strong>Generando iniciativas...</strong>
            <p>La IA está preparando la cartera de medidas con los datos de la empresa.</p>
          </div>
        </div>
      ) : null}
      {meta ? (
        <div className="panel ai-status ai-status--ok">
          <strong>Origen:</strong> IA · <strong>Modelo:</strong> {meta.model} · <strong>Búsqueda web:</strong>{" "}
          {meta.grounding_used ? "confirmada" : "no confirmada"}
        </div>
      ) : data.length ? (
        <div className="panel ai-status ai-status--warning">
          <strong>Origen:</strong> no registrado · vuelve a generar las iniciativas para confirmar el modelo usado.
        </div>
      ) : null}
      {meta?.grounding_queries?.length ? (
        <div className="panel ai-status">
          <strong>Consultas web:</strong> {meta.grounding_queries.join(" | ")}
        </div>
      ) : null}
      {error ? (
        <div className="panel ai-status ai-status--error">
          <strong>Error IA:</strong> {error}
        </div>
      ) : null}
      {data.length ? (
        <div className="panel ai-status">
          Las iniciativas y sus valores son una estimación inicial. Revisa y ajusta CAPEX, ahorros, CO₂ evitado,
          plazos y encaje estratégico si dispones de datos internos más precisos.
        </div>
      ) : null}

      <div className={`panel table-shell${isLoading ? " table-shell--loading" : ""}`}>
        <table className="data-table">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th key={header.id}>
                    {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

