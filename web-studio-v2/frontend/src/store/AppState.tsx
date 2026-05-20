import {
  createContext,
  useContext,
  useMemo,
  useReducer,
  type Dispatch,
  type PropsWithChildren,
} from "react";
import type {
  AppState,
  AiGenerationError,
  AiGenerationMeta,
  CompanyInputs,
  FinancialParams,
  FootprintResult,
  Initiative,
  PestelResult,
  PortfolioResult,
  StepId,
} from "../types";

type Action =
  | { type: "set-company-inputs"; payload: CompanyInputs }
  | { type: "set-footprint"; payload: FootprintResult | null }
  | { type: "set-pestel"; payload: PestelResult | null }
  | { type: "set-pestel-meta"; payload: AiGenerationMeta | null }
  | { type: "set-initiatives"; payload: Initiative[] }
  | { type: "set-initiatives-meta"; payload: AiGenerationMeta | null }
  | { type: "set-metrics"; payload: Initiative[] }
  | { type: "set-portfolio"; payload: PortfolioResult | null }
  | { type: "set-financial"; payload: FinancialParams }
  | { type: "set-step"; payload: StepId }
  | { type: "set-ai-error"; payload: AiGenerationError | null };

const initialState: AppState = {
  companyInputs: {
    company_name: "",
    sector: "",
    country: "España",
    province: "Valencia",
    postal_code: "",
    main_customer_locations: "",
    inventory_year: 2025,
    country_region: "España",
    electricity_method: "location",
    annual_electricity_mwh: 0,
    annual_purchased_heat_mwh: 0,
    co2_factor_heat_t_per_mwh: 0,
    electricity_price_eur_mwh: 0,
    fuel_price_eur_mwh: 55,
    roof_area_m2: 0,
    supplier_name: "",
    gdo_coverage_pct: 0,
    stationary_fuels: [{ fuel_key: "", quantity: 0 }],
    mobile_fuels: [{ fuel_key: "", quantity: 0 }],
    refrigerants: [{ name: "", quantity: 0 }],
    scope2_supplier_rows: [{ supplier_name: "", consumo_mwh: 0 }],
    has_invoices: false,
    has_meters: false,
    has_submetering: false,
    has_energy_audit: false,
    implemented_measures: {
      LED: "No",
      GdO: "No",
      "Paneles solares": "No",
      "Flota eléctrica": "No",
      "Variadores de frecuencia": "No",
      "EMS/submetering": "No",
      "Recuperación de calor": "No",
      "Programa de fugas de aire comprimido": "No",
    },
  },
  footprint: null,
  pestel: null,
  pestelMeta: null,
  initiatives: [],
  initiativesMeta: null,
  metricsResult: [],
  portfolioResult: null,
  financialParams: {
    discount_rate: 0.08,
    horizon: 8,
    max_payback_years: null,
  },
  currentStep: "empresa",
  aiError: null,
};

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "set-company-inputs":
      return { ...state, companyInputs: action.payload };
    case "set-footprint":
      return { ...state, footprint: action.payload };
    case "set-pestel":
      return { ...state, pestel: action.payload };
    case "set-pestel-meta":
      return { ...state, pestelMeta: action.payload };
    case "set-initiatives":
      return { ...state, initiatives: action.payload, metricsResult: [], portfolioResult: null };
    case "set-initiatives-meta":
      return { ...state, initiativesMeta: action.payload };
    case "set-metrics":
      return { ...state, metricsResult: action.payload, portfolioResult: null };
    case "set-portfolio":
      return { ...state, portfolioResult: action.payload };
    case "set-financial":
      return { ...state, financialParams: action.payload };
    case "set-step":
      return { ...state, currentStep: action.payload };
    case "set-ai-error":
      return { ...state, aiError: action.payload };
    default:
      return state;
  }
}

const AppStateContext = createContext<AppState | null>(null);
const AppDispatchContext = createContext<Dispatch<Action> | null>(null);

export function AppStateProvider({ children }: PropsWithChildren) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const memoState = useMemo(() => state, [state]);

  return (
    <AppDispatchContext.Provider value={dispatch}>
      <AppStateContext.Provider value={memoState}>{children}</AppStateContext.Provider>
    </AppDispatchContext.Provider>
  );
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used within AppStateProvider");
  return ctx;
}

export function useAppDispatch() {
  const ctx = useContext(AppDispatchContext);
  if (!ctx) throw new Error("useAppDispatch must be used within AppStateProvider");
  return ctx;
}
