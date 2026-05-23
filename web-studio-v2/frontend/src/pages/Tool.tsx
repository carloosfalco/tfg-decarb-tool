import { useEffect, useMemo, useState } from "react";
import { Download } from "lucide-react";
import { Link } from "react-router-dom";
import {
  calculateFootprint,
  computeMetrics,
  fetchCatalog,
  generateAiInitiatives,
  generatePestel,
  getApiError,
  optimizePortfolio,
} from "../api/client";
import { CompanyForm } from "../components/inputs/CompanyForm";
import { Scope1Form } from "../components/inputs/Scope1Form";
import { Scope2Form } from "../components/inputs/Scope2Form";
import { Sidebar } from "../components/layout/Sidebar";
import { PortfolioOptimizer } from "../components/portfolio/PortfolioOptimizer";
import { InitiativesTable } from "../components/portfolio/InitiativesTable";
import { ReportExportPanel } from "../components/reports/ReportExportPanel";
import { FootprintOverview } from "../components/results/FootprintOverview";
import { PestelGrid } from "../components/results/PestelGrid";
import { downloadCarbonReportPdf } from "../reports/carbonReportPdf";
import { useAppDispatch, useAppState } from "../store/AppState";
import type { CatalogItem, StepId, StepStatus } from "../types";

export default function Tool() {
  const state = useAppState();
  const dispatch = useAppDispatch();
  const [catalogs, setCatalogs] = useState<{
    stationary: CatalogItem[];
    mobile: CatalogItem[];
    refrigerants: CatalogItem[];
    suppliers: CatalogItem[];
  }>({ stationary: [], mobile: [], refrigerants: [], suppliers: [] });
  const [catalogStatus, setCatalogStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [appError, setAppError] = useState<string | null>(null);
  const [aiLoading, setAiLoading] = useState<"pestel" | "initiatives" | null>(null);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const hasAnyReportData = Boolean(state.footprint || state.pestel || state.initiatives.length || state.portfolioResult);

  useEffect(() => {
    let cancelled = false;
    setCatalogStatus("loading");
    void Promise.all([
      fetchCatalog("/api/catalogs/stationary-fuels"),
      fetchCatalog("/api/catalogs/mobile-fuels"),
      fetchCatalog("/api/catalogs/refrigerants"),
      fetchCatalog(`/api/catalogs/electricity-suppliers?year=${state.companyInputs.inventory_year}`),
    ])
      .then(([stationary, mobile, refrigerants, suppliers]) => {
        if (cancelled) return;
        setCatalogs({
          stationary: stationary as CatalogItem[],
          mobile: mobile as CatalogItem[],
          refrigerants: refrigerants as CatalogItem[],
          suppliers: suppliers as CatalogItem[],
        });
        setCatalogStatus("ready");
        setAppError(null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setCatalogStatus("error");
        setAppError(getApiError(error, "No se pudieron cargar los catalogos base."));
      });
    return () => {
      cancelled = true;
    };
  }, [state.companyInputs.inventory_year]);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [state.currentStep]);

  const statuses = useMemo<Record<StepId, StepStatus>>(() => {
    const companyReady = Boolean(state.companyInputs.sector && state.companyInputs.province);
    const scope1Ready =
      state.companyInputs.stationary_fuels.some((item) => item.fuel_key) ||
      state.companyInputs.mobile_fuels.some((item) => item.fuel_key) ||
      state.companyInputs.refrigerants.some((item) => item.name);
    const scope2Ready =
      state.companyInputs.annual_electricity_mwh > 0 ||
      state.companyInputs.scope2_supplier_rows.some((item) => item.supplier_name);
    return {
      empresa: companyReady ? "completed" : "in-progress",
      alcance1: scope1Ready ? "completed" : "in-progress",
      alcance2: scope2Ready ? "completed" : "in-progress",
      pestel: state.pestel ? "completed" : "in-progress",
      iniciativas: state.initiatives.length ? "completed" : "in-progress",
      portafolio: state.portfolioResult ? "completed" : "in-progress",
    };
  }, [state.companyInputs, state.footprint, state.pestel, state.initiatives, state.metricsResult, state.portfolioResult]);

  const stepTitle: Record<StepId, { title: string; description: string }> = {
    empresa: {
      title: "Empresa",
      description: "Contexto corporativo, inventario y trazabilidad del dato.",
    },
    alcance1: {
      title: "Alcance 1",
      description: "Combustión fija, flota móvil y emisiones fugitivas.",
    },
    alcance2: {
      title: "Alcance 2",
      description: "Electricidad comprada, comparativa de métodos y calor o vapor adquirido.",
    },
    pestel: {
      title: "PESTEL",
      description: "Lectura estratégica del entorno regulatorio, tecnológico y económico.",
    },
    iniciativas: {
      title: "Iniciativas",
      description: "Cartera editable de medidas de descarbonización.",
    },
    portafolio: {
      title: "Portafolio",
      description: "Evaluación financiera, optimización y selección final.",
    },
  };

  async function handleCalculateFootprint(inputs = state.companyInputs) {
    try {
      const footprint = await calculateFootprint(inputs);
      dispatch({ type: "set-footprint", payload: footprint });
      setAppError(null);
      return footprint;
    } catch (error) {
      dispatch({ type: "set-footprint", payload: null });
      setAppError(getApiError(error, "No se pudo calcular la huella."));
      return null;
    }
  }

  const stepOrder: StepId[] = ["empresa", "alcance1", "alcance2", "pestel", "iniciativas", "portafolio"];
  const currentStepIndex = stepOrder.indexOf(state.currentStep);
  const previousStep = currentStepIndex > 0 ? stepOrder[currentStepIndex - 1] : null;
  const nextStep = currentStepIndex < stepOrder.length - 1 ? stepOrder[currentStepIndex + 1] : null;
  const stepLabels: Record<StepId, string> = {
    empresa: "Empresa",
    alcance1: "Alcance 1",
    alcance2: "Alcance 2",
    pestel: "PESTEL",
    iniciativas: "Iniciativas",
    portafolio: "Portafolio",
  };

  const footprintTrigger = useMemo(
    () =>
      JSON.stringify({
        inventory_year: state.companyInputs.inventory_year,
        stationary_fuels: state.companyInputs.stationary_fuels,
        mobile_fuels: state.companyInputs.mobile_fuels,
        refrigerants: state.companyInputs.refrigerants,
        electricity_method: state.companyInputs.electricity_method,
        annual_electricity_mwh: state.companyInputs.annual_electricity_mwh,
        scope2_supplier_rows: state.companyInputs.scope2_supplier_rows,
        gdo_coverage_pct: state.companyInputs.gdo_coverage_pct,
        annual_purchased_heat_mwh: state.companyInputs.annual_purchased_heat_mwh,
        co2_factor_heat_t_per_mwh: state.companyInputs.co2_factor_heat_t_per_mwh,
        has_invoices: state.companyInputs.has_invoices,
        has_meters: state.companyInputs.has_meters,
        has_submetering: state.companyInputs.has_submetering,
        has_energy_audit: state.companyInputs.has_energy_audit,
      }),
    [state.companyInputs]
  );

  useEffect(() => {
    const hasOperationalInputs =
      state.companyInputs.stationary_fuels.some((item) => item.fuel_key && item.quantity > 0) ||
      state.companyInputs.mobile_fuels.some((item) => item.fuel_key && item.quantity > 0) ||
      state.companyInputs.refrigerants.some((item) => item.name && item.quantity > 0) ||
      state.companyInputs.annual_electricity_mwh > 0 ||
      state.companyInputs.scope2_supplier_rows.some((item) => item.supplier_name || item.consumo_mwh > 0) ||
      state.companyInputs.annual_purchased_heat_mwh > 0;

    if (!hasOperationalInputs) {
      dispatch({ type: "set-footprint", payload: null });
      return;
    }

    const timeoutId = window.setTimeout(() => {
      void handleCalculateFootprint(state.companyInputs);
    }, 120);

    return () => window.clearTimeout(timeoutId);
  }, [dispatch, footprintTrigger]);

  return (
    <main className="app-shell">
      <Sidebar
        currentStep={state.currentStep}
        statuses={statuses}
        onStepChange={(step) => dispatch({ type: "set-step", payload: step })}
      />

      <section className="content-shell">
        <header className="topbar panel">
          <div>
            <span className="badge">Flujo metodológico</span>
            <h1>{stepTitle[state.currentStep].title}</h1>
            <p>{stepTitle[state.currentStep].description}</p>
          </div>
          <div className="topbar__actions">
            <Link className="button button--secondary" to="/">
              Volver
            </Link>
            {state.currentStep !== "portafolio" && state.initiatives.length ? (
              <button className="button button--secondary" type="button" onClick={() => dispatch({ type: "set-step", payload: "portafolio" })}>
                Supuestos financieros
              </button>
            ) : null}
            <button
              className="button button--primary"
              type="button"
              disabled={!hasAnyReportData}
              onClick={() => downloadCarbonReportPdf(state)}
              title={hasAnyReportData ? "Exportar informe PDF" : "Genera datos antes de exportar el informe"}
            >
              <Download size={18} />
              Informe PDF
            </button>
          </div>
        </header>

        {appError ? (
          <div className="app-alert app-alert--error" role="alert">
            <strong>Algo no ha ido bien.</strong>
            <span>{appError}</span>
          </div>
        ) : null}

        {catalogStatus === "loading" ? <div className="app-alert">Cargando catalogos de factores...</div> : null}

        {state.currentStep === "empresa" ? (
          <CompanyForm
            value={state.companyInputs}
            onChange={(next) => dispatch({ type: "set-company-inputs", payload: next })}
            onSubmit={(next) => {
              dispatch({ type: "set-company-inputs", payload: next });
              dispatch({ type: "set-step", payload: "alcance1" });
            }}
          />
        ) : null}

        {state.currentStep === "alcance1" ? (
          <Scope1Form
            value={state.companyInputs}
            stationaryCatalog={catalogs.stationary}
            mobileCatalog={catalogs.mobile}
            refrigerantCatalog={catalogs.refrigerants}
            footprint={state.footprint}
            onChange={(next) => dispatch({ type: "set-company-inputs", payload: next })}
            onPrevious={() => dispatch({ type: "set-step", payload: "empresa" })}
            onSubmit={(next) => {
              dispatch({ type: "set-company-inputs", payload: next });
              dispatch({ type: "set-step", payload: "alcance2" });
            }}
          />
        ) : null}

        {state.currentStep === "alcance2" ? (
          <Scope2Form
            value={state.companyInputs}
            supplierCatalog={catalogs.suppliers}
            footprint={state.footprint}
            onChange={(next) => dispatch({ type: "set-company-inputs", payload: next })}
            onPrevious={() => dispatch({ type: "set-step", payload: "alcance1" })}
            onSubmit={(next) => {
              dispatch({ type: "set-company-inputs", payload: next });
              void handleCalculateFootprint(next);
              dispatch({ type: "set-step", payload: "pestel" });
            }}
          />
        ) : null}

        {state.currentStep === "pestel" ? (
          <PestelGrid
            pestel={state.pestel}
            meta={state.pestelMeta}
            error={state.aiError?.area === "pestel" ? state.aiError.message : null}
            isLoading={aiLoading === "pestel"}
            onGenerate={() =>
              {
                if (aiLoading) return;
                setAiLoading("pestel");
                dispatch({ type: "set-ai-error", payload: null });
                const footprintPromise = state.footprint
                  ? Promise.resolve(state.footprint)
                  : calculateFootprint(state.companyInputs);
                void footprintPromise
                  .then((footprint) => {
                    dispatch({ type: "set-footprint", payload: footprint });
                    return generatePestel(state.companyInputs, footprint);
                  })
                  .then((result) => {
                    dispatch({ type: "set-pestel", payload: result.pestel });
                    dispatch({ type: "set-pestel-meta", payload: result.meta });
                  })
                  .catch((error: unknown) => {
                    dispatch({ type: "set-pestel-meta", payload: null });
                    dispatch({
                      type: "set-ai-error",
                      payload: {
                        area: "pestel",
                        message: getApiError(error, "No se pudo generar el PESTEL con IA."),
                      },
                    });
                  })
                  .finally(() => setAiLoading(null));
              }
            }
          />
        ) : null}

        {state.currentStep === "iniciativas" ? (
          <InitiativesTable
            data={state.initiatives}
            meta={state.initiativesMeta}
            error={state.aiError?.area === "initiatives" ? state.aiError.message : null}
            isLoading={aiLoading === "initiatives"}
            onChange={(rows) => dispatch({ type: "set-initiatives", payload: rows })}
            onGenerateAi={() =>
              {
                if (aiLoading) return;
                setAiLoading("initiatives");
                dispatch({ type: "set-ai-error", payload: null });
                const footprintPromise = state.footprint
                  ? Promise.resolve(state.footprint)
                  : calculateFootprint(state.companyInputs);
                void footprintPromise
                  .then((footprint) => {
                    dispatch({ type: "set-footprint", payload: footprint });
                    return generateAiInitiatives(state.companyInputs, footprint, state.pestel);
                  })
                  .then((result) => {
                    dispatch({ type: "set-initiatives", payload: result.initiatives });
                    dispatch({ type: "set-initiatives-meta", payload: result.meta });
                  })
                  .catch((error: unknown) => {
                    dispatch({ type: "set-initiatives-meta", payload: null });
                    dispatch({
                      type: "set-ai-error",
                      payload: {
                        area: "initiatives",
                        message: getApiError(error, "No se pudieron generar iniciativas con IA."),
                      },
                    });
                  })
                  .finally(() => setAiLoading(null));
              }
            }
          />
        ) : null}

        {state.currentStep === "portafolio" ? (
          <section className="stack portfolio-step">
            <PortfolioOptimizer
              initiatives={state.initiatives}
              financialParams={state.financialParams}
              result={state.portfolioResult}
              isLoading={portfolioLoading}
              onFinancialChange={(value) => dispatch({ type: "set-financial", payload: value })}
              onOptimize={async ({ financialParams, budget, minCo2 }) => {
                setPortfolioLoading(true);
                try {
                  const metrics = await computeMetrics(state.initiatives, financialParams);
                  const filteredMetrics =
                    financialParams.max_payback_years === null || financialParams.max_payback_years === undefined
                      ? metrics
                      : metrics.filter((item) => {
                          const payback = Number(item.payback_years);
                          return Number.isFinite(payback) && payback <= Number(financialParams.max_payback_years);
                        });
                  dispatch({ type: "set-metrics", payload: filteredMetrics });
                  if (!filteredMetrics.length) {
                    dispatch({ type: "set-portfolio", payload: null });
                    throw new Error(
                      "No hay iniciativas que cumplan el payback máximo aceptable. Amplía el payback máximo o elimina esa restricción."
                    );
                  }
                  const cleanedMetrics = filteredMetrics.map((item) => ({
                    ...item,
                    capex_eur: Number.isFinite(Number(item.capex_eur)) ? Number(item.capex_eur) : 0,
                    annual_co2_reduction_t: Number.isFinite(Number(item.annual_co2_reduction_t))
                      ? Number(item.annual_co2_reduction_t)
                      : 0,
                    annual_opex_saving_eur: Number.isFinite(Number(item.annual_opex_saving_eur))
                      ? Number(item.annual_opex_saving_eur)
                      : 0,
                    strategic_score_1_5: Number.isFinite(Number(item.strategic_score_1_5))
                      ? Number(item.strategic_score_1_5)
                      : 3,
                    npv_eur: Number.isFinite(Number(item.npv_eur)) ? Number(item.npv_eur) : 0,
                  }));
                  const result = await optimizePortfolio(
                    cleanedMetrics,
                    budget,
                    minCo2,
                    "Balanced score (NPV + CO2 + strategy)",
                    { w_npv: 0.3, w_co2: 0.7, w_strategy: 0 }
                  );
                  dispatch({ type: "set-portfolio", payload: result });
                  setAppError(null);
                } catch (error) {
                  dispatch({ type: "set-portfolio", payload: null });
                  setAppError(getApiError(error, "No se pudo optimizar el portafolio."));
                } finally {
                  setPortfolioLoading(false);
                }
              }}
            />
          </section>
        ) : null}

        {state.currentStep !== "empresa" &&
        state.currentStep !== "pestel" &&
        state.currentStep !== "iniciativas" &&
        state.currentStep !== "portafolio" ? (
          <FootprintOverview footprint={state.footprint} />
        ) : null}

        {["pestel", "iniciativas", "portafolio"].includes(state.currentStep) ? (
        <div className="tool-shortcuts panel">
          {previousStep ? (
            <button
              className="button button--secondary"
              type="button"
              onClick={() => dispatch({ type: "set-step", payload: previousStep })}
            >
              Anterior: {stepLabels[previousStep]}
            </button>
          ) : null}
          {nextStep ? (
            <button
              className="button button--primary"
              type="button"
              onClick={() => dispatch({ type: "set-step", payload: nextStep })}
            >
              Siguiente: {stepLabels[nextStep]}
            </button>
          ) : null}
        </div>
        ) : null}

        {state.currentStep === "portafolio" ? <ReportExportPanel state={state} /> : null}

        {aiLoading ? <div className="loading-bar">Procesando...</div> : null}
      </section>
    </main>
  );
}
