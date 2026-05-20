import type { AiGenerationMeta, PestelResult } from "../../types";

const labels = ["Político", "Económico", "Social", "Tecnológico", "Ambiental", "Legal"];

function renderAiFormattedText(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>;
    }
    return <span key={`${part}-${index}`}>{part}</span>;
  });
}

export function PestelGrid({
  pestel,
  meta,
  error,
  isLoading = false,
  onGenerate,
}: {
  pestel: PestelResult | null;
  meta: AiGenerationMeta | null;
  error: string | null;
  isLoading?: boolean;
  onGenerate: () => void;
}) {
  const hasPestel = labels.some((label) => (pestel?.[label] ?? []).length > 0);

  return (
    <section className="stack">
      <div className="panel action-bar">
        <span className="muted">Genera o actualiza ideas PESTEL concretas con IA y búsqueda web.</span>
        <button className="button button--primary" type="button" onClick={onGenerate} disabled={isLoading}>
          {isLoading ? <span className="spinner" aria-hidden="true" /> : null}
          {isLoading ? "Generando..." : "Generar con IA"}
        </button>
      </div>
      {isLoading ? (
        <div className="panel ai-status ai-status--loading">
          <span className="spinner" aria-hidden="true" />
          <strong>IA generando PESTEL...</strong>
        </div>
      ) : null}
      {meta ? (
        <div className="panel ai-status ai-status--ok">
          <strong>Origen:</strong> IA · <strong>Modelo:</strong> {meta.model} · <strong>Búsqueda web:</strong>{" "}
          {meta.grounding_used ? "confirmada" : "no confirmada"}
        </div>
      ) : hasPestel ? (
        <div className="panel ai-status ai-status--warning">
          <strong>Origen:</strong> no registrado · vuelve a generar el PESTEL para confirmar el modelo usado.
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
      <div className="pestel-grid">
        {labels.map((label) => (
          <article key={label} className="panel pestel-card">
            <h3>{label}</h3>
            <ul className="simple-list">
              {(pestel?.[label] ?? []).slice(0, 3).map((item) => (
                <li key={item}>{renderAiFormattedText(item)}</li>
              ))}
            </ul>
            {!pestel?.[label]?.length ? <p className="muted">Sin análisis generado todavía.</p> : null}
          </article>
        ))}
      </div>
    </section>
  );
}
