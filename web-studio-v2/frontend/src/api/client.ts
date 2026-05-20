import axios from "axios";
import type {
  AiGenerationMeta,
  CompanyInputs,
  FinancialParams,
  FootprintResult,
  Initiative,
  PestelResult,
  PortfolioResult,
} from "../types";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8010",
  timeout: 120000,
});

type Envelope<T> = {
  ok: boolean;
  data: T;
  message?: string;
};

export function getApiError(error: unknown, fallback: string) {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail || error.response?.data?.message;
    if (detail) return String(detail);
    if (error.code === "ECONNABORTED") {
      return "La API ha tardado demasiado en responder. La generación con investigación web puede durar varios minutos; intenta de nuevo o revisa el servidor local.";
    }
    if (error.message === "Network Error") {
      return "No se pudo conectar con el backend local. Revisa que el servidor de la API este arrancado en http://127.0.0.1:8010.";
    }
    return error.message || fallback;
  }
  return error instanceof Error ? error.message : fallback;
}

function unwrapEnvelope<T>(envelope: Envelope<T>, fallback: string): T {
  if (!envelope?.ok) {
    throw new Error(envelope?.message || fallback);
  }
  if (!envelope.data) {
    throw new Error(fallback);
  }
  return envelope.data;
}

async function getWithRetry<T>(path: string, fallback: string, attempt = 1): Promise<T> {
  try {
    const { data } = await api.get<Envelope<T>>(path);
    return unwrapEnvelope(data, fallback);
  } catch (error) {
    if (attempt < 2 && axios.isAxiosError(error) && (!error.response || error.response.status >= 500)) {
      await new Promise((resolve) => window.setTimeout(resolve, 250));
      return getWithRetry<T>(path, fallback, attempt + 1);
    }
    throw new Error(getApiError(error, fallback));
  }
}

async function postEnvelope<T>(path: string, body: unknown, fallback: string, timeout?: number): Promise<T> {
  try {
    const { data } = await api.post<Envelope<T>>(path, body, timeout ? { timeout } : undefined);
    return unwrapEnvelope(data, fallback);
  } catch (error) {
    throw new Error(getApiError(error, fallback));
  }
}

export async function fetchCatalog(path: string) {
  const data = await getWithRetry<{ items: unknown[] }>(path, "No se pudo cargar el catalogo solicitado.");
  return data.items;
}

export async function calculateFootprint(companyInputs: CompanyInputs) {
  const data = await postEnvelope<{ footprint: FootprintResult }>(
    "/api/calculate-footprint",
    { company_inputs: companyInputs },
    "No se pudo calcular la huella."
  );
  return data.footprint;
}

export async function generatePestel(companyInputs: CompanyInputs, footprint: FootprintResult | null, geminiApiKey = "") {
  const data = await postEnvelope<{
    pestel: PestelResult;
    source: "ai";
    model: string;
    grounding_used: boolean;
    grounding_queries: string[];
    grounding_sources: { uri: string; title: string }[];
  }>(
    "/api/generate-pestel",
    { company_inputs: companyInputs, footprint, gemini_api_key: geminiApiKey },
    "No se pudo generar el PESTEL con IA.",
    180000
  );
  return {
    pestel: data.pestel,
    meta: {
      source: data.source,
      model: data.model,
      generated_at: new Date().toISOString(),
      grounding_used: data.grounding_used,
      grounding_queries: data.grounding_queries,
      grounding_sources: data.grounding_sources,
    } satisfies AiGenerationMeta,
  };
}

export async function generateInitiatives(companyInputs: CompanyInputs, footprint: FootprintResult) {
  const data = await postEnvelope<{ initiatives: Initiative[] }>(
    "/api/generate-initiatives",
    { company_inputs: companyInputs, footprint },
    "No se pudieron generar iniciativas."
  );
  return data.initiatives;
}

export async function generateAiInitiatives(
  companyInputs: CompanyInputs,
  footprint: FootprintResult,
  pestel: PestelResult | null = null,
  geminiApiKey = ""
) {
  const data = await postEnvelope<{
    initiatives: Initiative[];
    source: "ai";
    model: string;
    grounding_used: boolean;
    grounding_queries: string[];
    grounding_sources: { uri: string; title: string }[];
  }>(
    "/api/generate-ai-initiatives",
    { company_inputs: companyInputs, footprint, pestel, gemini_api_key: geminiApiKey },
    "No se pudieron generar iniciativas con IA.",
    240000
  );
  return {
    initiatives: data.initiatives,
    meta: {
      source: data.source,
      model: data.model,
      generated_at: new Date().toISOString(),
      grounding_used: data.grounding_used,
      grounding_queries: data.grounding_queries,
      grounding_sources: data.grounding_sources,
    } satisfies AiGenerationMeta,
  };
}

export async function computeMetrics(initiatives: Initiative[], financialParams: FinancialParams) {
  const data = await postEnvelope<{ initiatives: Initiative[] }>(
    "/api/compute-metrics",
    { initiatives, financial_params: financialParams },
    "No se pudieron calcular las metricas financieras."
  );
  return data.initiatives;
}

export async function optimizePortfolio(
  initiativesWithMetrics: Initiative[],
  budget_eur: number,
  min_co2_t: number,
  objective: string,
  weights: { w_npv: number; w_co2: number; w_strategy: number }
) {
  return postEnvelope<PortfolioResult>(
    "/api/optimize-portfolio",
    {
      initiatives_with_metrics: initiativesWithMetrics,
      budget_eur,
      min_co2_t,
      objective,
      weights,
    },
    "No se pudo optimizar el portafolio."
  );
}
