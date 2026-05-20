import { Download, FileText } from "lucide-react";
import { downloadCarbonReportPdf } from "../../reports/carbonReportPdf";
import type { AppState } from "../../types";

export function ReportExportPanel({ state }: { state: AppState }) {
  const hasAnyReportData = Boolean(state.footprint || state.pestel || state.initiatives.length || state.portfolioResult);

  return (
    <section className="panel report-export-panel">
      <div className="report-export-panel__icon" aria-hidden="true">
        <FileText size={24} />
      </div>
      <div className="report-export-panel__copy">
        <span className="badge">Informe final</span>
        <h3>Exportar informe completo en PDF</h3>
        <p>
          Genera un documento con resumen ejecutivo, huella de carbono, PESTEL, iniciativas y portafolio. Usa solo los
          datos ya disponibles en la aplicacion, sin realizar nuevas llamadas a IA.
        </p>
      </div>
      <button
        className="button button--primary report-export-panel__button"
        type="button"
        disabled={!hasAnyReportData}
        onClick={() => downloadCarbonReportPdf(state)}
      >
        <Download size={18} />
        Exportar PDF
      </button>
    </section>
  );
}
