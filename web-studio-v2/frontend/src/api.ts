import type {
  CatalogItem,
  CompanyInputs,
  FinancialParams,
  FootprintResult,
  Initiative,
  PestelResult,
  PortfolioResult,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8010";

type Envelope<T> = {
  ok: boolean;
  data: T;
  message?: string;
};

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`API ${response.status}: ${await response.text()}`);
  }
  const payload = (await response.json()) as Envelope<T>;
  return payload.data;
}

export async function fetchCatalog(path: string): Promise<CatalogItem[]> {
  const data = await apiFetch<{ items: CatalogItem[] }>(path);
  return data.items;
}

export async function calculateFootprint(companyInputs: CompanyInputs): Promise<FootprintResult> {
  const data = await apiFetch<{ footprint: FootprintResult }>("/api/calculate-footprint", {
    method: "POST",
    body: JSON.stringify({ company_inputs: companyInputs }),
  });
  return data.footprint;
}

export async function generatePestel(
  companyInputs: CompanyInputs,
  footprint: FootprintResult | null,
  geminiApiKey: string
): Promise<PestelResult> {
  const data = await apiFetch<{ pestel: PestelResult }>("/api/generate-pestel", {
    method: "POST",
    body: JSON.stringify({ company_inputs: companyInputs, footprint, gemini_api_key: geminiApiKey }),
  });
  return data.pestel;
}

export async function generateInitiatives(
  companyInputs: CompanyInputs,
  footprint: FootprintResult,
  geminiApiKey: string,
  pestel: PestelResult | null = null
): Promise<Initiative[]> {
  const path = geminiApiKey ? "/api/generate-ai-initiatives" : "/api/generate-initiatives";
  const data = await apiFetch<{ initiatives: Initiative[] }>(path, {
    method: "POST",
    body: JSON.stringify({ company_inputs: companyInputs, footprint, pestel, gemini_api_key: geminiApiKey }),
  });
  return data.initiatives;
}

export async function computeMetrics(
  initiatives: Initiative[],
  financialParams: FinancialParams
): Promise<Initiative[]> {
  const data = await apiFetch<{ initiatives: Initiative[] }>("/api/compute-metrics", {
    method: "POST",
    body: JSON.stringify({ initiatives, financial_params: financialParams }),
  });
  return data.initiatives;
}

export async function optimizePortfolio(
  initiativesWithMetrics: Initiative[],
  budget: number,
  minCo2: number,
  objective: string,
  weights: { w_npv: number; w_co2: number; w_strategy: number }
): Promise<PortfolioResult> {
  return apiFetch<PortfolioResult>("/api/optimize-portfolio", {
    method: "POST",
    body: JSON.stringify({
      initiatives_with_metrics: initiativesWithMetrics,
      budget_eur: budget,
      min_co2_t: minCo2,
      objective,
      weights,
    }),
  });
}
