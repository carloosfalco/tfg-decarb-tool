export type StepId =
  | "empresa"
  | "alcance1"
  | "alcance2"
  | "pestel"
  | "iniciativas"
  | "portafolio";

export type StepStatus = "pending" | "in-progress" | "completed" | "locked";

export type MeasureStatus = "No" | "Parcial" | "Sí";

export interface StationaryFuelEntry {
  fuel_key: string;
  quantity: number;
}

export interface MobileFuelEntry {
  fuel_key: string;
  quantity: number;
}

export interface RefrigerantEntry {
  name: string;
  quantity: number;
}

export interface SupplierRow {
  supplier_name: string;
  consumo_mwh: number;
}

export interface CompanyInputs {
  company_name: string;
  sector: string;
  country?: string;
  province: string;
  postal_code?: string;
  main_customer_locations?: string;
  inventory_year: 2023 | 2024 | 2025;
  country_region: string;
  electricity_method: "location" | "market";
  annual_electricity_mwh: number;
  annual_purchased_heat_mwh: number;
  co2_factor_heat_t_per_mwh: number;
  electricity_price_eur_mwh: number;
  fuel_price_eur_mwh: number;
  roof_area_m2: number;
  supplier_name: string;
  gdo_coverage_pct: number;
  stationary_fuels: StationaryFuelEntry[];
  mobile_fuels: MobileFuelEntry[];
  refrigerants: RefrigerantEntry[];
  scope2_supplier_rows: SupplierRow[];
  has_invoices: boolean;
  has_meters: boolean;
  has_submetering: boolean;
  has_energy_audit: boolean;
  implemented_measures: Record<string, MeasureStatus>;
}

export interface CatalogItem {
  key?: string;
  label?: string;
  name?: string;
  fuel_label?: string;
  vehicle_type?: string;
  unit?: string;
  factor_kg_co2e_kwh?: number;
  factors_kg_per_unit?: Record<number, number>;
  mwh_per_unit?: number | null;
  gwp?: number;
}

export interface FootprintResult {
  scope1_t: number;
  scope2_t: number;
  total_t: number;
  scope1_stationary_t: number;
  scope1_fleet_t: number;
  scope1_fugitive_t: number;
  scope2_elec_t: number;
  scope2_heat_t: number;
  scope2_elec_method: string;
  scope2_location_t: number;
  scope2_market_t: number;
  scope2_difference_t: number;
  scope2_difference_pct: number;
  breakdown: Record<string, number>;
  quality: {
    label: string;
    text: string;
  };
  used_elec_factor?: number;
  used_heat_factor?: number;
  scope1_factor_source?: string;
  scope2_elec_source?: string;
  scope2_heat_source?: string;
  refrigerant_factor_found?: boolean;
  refrigerant_key?: string;
  refrigerant_gwp?: number;
  scope2_notes?: string[];
  scope2_errors?: string[];
}

export interface PestelResult {
  [key: string]: string[];
}

export interface AiGenerationMeta {
  source: "ai";
  model: string;
  generated_at: string;
  grounding_used: boolean;
  grounding_queries: string[];
  grounding_sources: { uri: string; title: string }[];
}

export interface AiGenerationError {
  area: "pestel" | "initiatives";
  message: string;
}

export interface Initiative {
  id: number;
  scope: string;
  emission_source: string;
  initiative_family: string;
  initiative: string;
  capex_eur: number;
  annual_opex_saving_eur: number | null;
  annual_co2_reduction_t: number | null;
  implementation_months: number;
  strategic_score_1_5: number;
  activity_unit?: string;
  categoria?: string;
  selected?: boolean;
  npv_eur?: number | null;
  payback_years?: number | null;
  total_annual_benefit_eur?: number | null;
}

export interface FinancialParams {
  discount_rate: number;
  horizon: number;
  max_payback_years?: number | null;
}

export interface PortfolioResult {
  initiatives: Initiative[];
  summary: {
    status: string;
    total_capex: number;
    total_co2: number;
    total_npv: number;
    selected_count: number;
  };
}

export interface AppState {
  companyInputs: CompanyInputs;
  footprint: FootprintResult | null;
  pestel: PestelResult | null;
  pestelMeta: AiGenerationMeta | null;
  initiatives: Initiative[];
  initiativesMeta: AiGenerationMeta | null;
  metricsResult: Initiative[];
  portfolioResult: PortfolioResult | null;
  financialParams: FinancialParams;
  currentStep: StepId;
  aiError: AiGenerationError | null;
}
