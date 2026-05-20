from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app_logic import (
    calculate_company_footprint,
    compute_metrics,
    enrich_company_inputs,
    generate_ai_initiatives,
    generate_ai_pestel,
    list_electricity_suppliers,
    list_mobile_fuels,
    list_refrigerants,
    list_stationary_fuels,
    optimize_portfolio,
    propose_initiatives,
)
from schemas import (
    ApiEnvelope,
    CalculateFootprintRequest,
    ComputeMetricsRequest,
    GenerateAiInitiativesRequest,
    GenerateInitiativesRequest,
    GeneratePestelRequest,
    OptimizePortfolioRequest,
)

logger = logging.getLogger("decarb_web_studio")

app = FastAPI(
    title="Decarb Web Studio API",
    version="1.0.0",
    description="API paralela para comparar una nueva experiencia web profesional con la app original en Streamlit.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "BACKEND_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5180,http://127.0.0.1:5180,http://localhost:5181,http://127.0.0.1:5181",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
LEGACY_GEMINI_MODELS = {
    "gemini-3.1-flash-lite-preview",
    "models/gemini-3.1-flash-lite-preview",
}
GEMINI_FALLBACK_MODELS = ("gemini-flash-latest", "gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-flash-lite-latest")

try:
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    tomllib = None


def _serialize_df(df: pd.DataFrame) -> list[Dict[str, Any]]:
    return df.replace({pd.NA: None}).where(pd.notna(df), None).to_dict(orient="records")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Request validation error: %s", exc)
    return JSONResponse(
        status_code=422,
        content=ApiEnvelope(ok=False, data={}, message="Los datos enviados no tienen el formato esperado.").model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error")
    return JSONResponse(
        status_code=500,
        content=ApiEnvelope(ok=False, data={}, message=f"Error interno controlado: {exc}").model_dump(),
    )


def _read_env_like_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _load_gemini_settings() -> Dict[str, str]:
    settings = {
        "gemini_api_key": os.getenv("GEMINI_API_KEY", "") or "",
        "gemini_model": os.getenv("GEMINI_MODEL", "") or "",
    }

    if tomllib is not None:
        secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            try:
                data = tomllib.loads(secrets_path.read_text(encoding="utf-8-sig", errors="ignore"))
                settings["gemini_api_key"] = settings["gemini_api_key"] or str(data.get("GEMINI_API_KEY", "") or "")
                settings["gemini_model"] = settings["gemini_model"] or str(data.get("GEMINI_MODEL", "") or "")
            except Exception:
                pass

    for env_path in [PROJECT_ROOT / ".env", BASE_DIR / ".env"]:
        env_values = _read_env_like_file(env_path)
        settings["gemini_api_key"] = settings["gemini_api_key"] or env_values.get("GEMINI_API_KEY", "")
        settings["gemini_model"] = settings["gemini_model"] or env_values.get("GEMINI_MODEL", "")

    return settings


def _resolved_gemini_key(request_key: str) -> str:
    return request_key or _load_gemini_settings()["gemini_api_key"]


def _resolved_gemini_model(request_model: str) -> str:
    model = (request_model or _load_gemini_settings()["gemini_model"] or DEFAULT_GEMINI_MODEL).strip()
    return DEFAULT_GEMINI_MODEL if model in LEGACY_GEMINI_MODELS else model


def _candidate_gemini_models(primary_model: str) -> list[str]:
    candidates: list[str] = []
    for model in [primary_model, *GEMINI_FALLBACK_MODELS]:
        cleaned = (model or "").strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
    return candidates


@app.on_event("startup")
def validate_runtime_catalogs() -> None:
    checks = {
        "stationary_fuels": len(list_stationary_fuels()),
        "mobile_fuels": len(list_mobile_fuels()),
        "refrigerants": len(list_refrigerants()),
        "electricity_suppliers_2025": len(list_electricity_suppliers(2025)),
    }
    missing = [name for name, count in checks.items() if count <= 0]
    if missing:
        raise RuntimeError(f"Catalogos vacios o no disponibles: {', '.join(missing)}")
    logger.info("Runtime catalogs loaded: %s", checks)


@app.get("/health", response_model=ApiEnvelope)
def health() -> ApiEnvelope:
    settings = _load_gemini_settings()
    return ApiEnvelope(
        data={
            "status": "ok",
            "app": "web-studio-v2",
            "gemini_model": _resolved_gemini_model(settings["gemini_model"]),
            "gemini_key_loaded": bool(settings["gemini_api_key"]),
        }
    )


@app.post("/api/calculate-footprint", response_model=ApiEnvelope)
def calculate_footprint(payload: CalculateFootprintRequest) -> ApiEnvelope:
    footprint = calculate_company_footprint(payload.company_inputs)
    return ApiEnvelope(data={"footprint": footprint})


@app.post("/api/generate-pestel", response_model=ApiEnvelope)
def generate_pestel(payload: GeneratePestelRequest) -> ApiEnvelope:
    footprint = payload.footprint or calculate_company_footprint(payload.company_inputs)
    company_inputs = enrich_company_inputs(payload.company_inputs, footprint)
    gemini_key = _resolved_gemini_key(payload.gemini_api_key)
    gemini_model = _resolved_gemini_model(payload.gemini_model)
    if not gemini_key:
        raise HTTPException(status_code=400, detail="No hay GEMINI_API_KEY cargada para generar el PESTEL con IA.")
    errors: list[str] = []
    try:
        for candidate_model in _candidate_gemini_models(gemini_model):
            try:
                result = generate_ai_pestel(company_inputs, footprint, gemini_key, candidate_model)
                gemini_model = candidate_model
                break
            except Exception as exc:
                message = str(exc)
                errors.append(f"{candidate_model}: {message}")
                if "429" not in message:
                    raise
        else:
            raise RuntimeError("No se pudo generar PESTEL con IA. " + " | ".join(errors))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ApiEnvelope(
        data={
            "pestel": result["pestel"],
            "source": "ai",
            "model": gemini_model,
            "grounding_used": result["grounding_used"],
            "grounding_queries": result["grounding_queries"],
            "grounding_sources": result["grounding_sources"],
        }
    )


@app.post("/api/generate-initiatives", response_model=ApiEnvelope)
def generate_initiatives(payload: GenerateInitiativesRequest) -> ApiEnvelope:
    company_inputs = enrich_company_inputs(payload.company_inputs, payload.footprint)
    initiatives = propose_initiatives(company_inputs, payload.footprint, n=8)
    return ApiEnvelope(data={"initiatives": _serialize_df(initiatives)})


@app.post("/api/generate-ai-initiatives", response_model=ApiEnvelope)
def generate_ai_initiatives_endpoint(payload: GenerateAiInitiativesRequest) -> ApiEnvelope:
    company_inputs = enrich_company_inputs(payload.company_inputs, payload.footprint)
    gemini_key = _resolved_gemini_key(payload.gemini_api_key)
    gemini_model = _resolved_gemini_model(payload.gemini_model)
    if not gemini_key:
        raise HTTPException(status_code=400, detail="No hay GEMINI_API_KEY cargada para generar iniciativas con IA.")
    errors: list[str] = []
    try:
        for candidate_model in _candidate_gemini_models(gemini_model):
            try:
                result = generate_ai_initiatives(
                    company_inputs,
                    payload.footprint,
                    gemini_key,
                    candidate_model,
                    payload.n,
                    payload.pestel,
                )
                gemini_model = candidate_model
                break
            except Exception as exc:
                message = str(exc)
                errors.append(f"{candidate_model}: {message}")
                if "429" not in message:
                    raise
        else:
            raise RuntimeError("No se pudieron generar iniciativas con IA. " + " | ".join(errors))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ApiEnvelope(
        data={
            "initiatives": _serialize_df(result["initiatives"]),
            "source": "ai",
            "model": gemini_model,
            "grounding_used": result["grounding_used"],
            "grounding_queries": result["grounding_queries"],
            "grounding_sources": result["grounding_sources"],
        }
    )


@app.post("/api/compute-metrics", response_model=ApiEnvelope)
def compute_metrics_endpoint(payload: ComputeMetricsRequest) -> ApiEnvelope:
    df = pd.DataFrame(payload.initiatives)
    result = compute_metrics(
        df,
        horizon_years=payload.financial_params.horizon,
        discount_rate=payload.financial_params.discount_rate,
    )
    return ApiEnvelope(data={"initiatives": _serialize_df(result)})


@app.post("/api/optimize-portfolio", response_model=ApiEnvelope)
def optimize_portfolio_endpoint(payload: OptimizePortfolioRequest) -> ApiEnvelope:
    df = pd.DataFrame(payload.initiatives_with_metrics)
    optimized, summary = optimize_portfolio(
        df,
        budget_eur=payload.budget_eur,
        min_co2_t=payload.min_co2_t,
        objective=payload.objective,
        w_npv=payload.weights.w_npv,
        w_co2=payload.weights.w_co2,
        w_strategy=payload.weights.w_strategy,
    )
    return ApiEnvelope(data={"initiatives": _serialize_df(optimized), "summary": summary})


@app.get("/api/catalogs/stationary-fuels", response_model=ApiEnvelope)
def stationary_fuels_catalog() -> ApiEnvelope:
    return ApiEnvelope(data={"items": list_stationary_fuels()})


@app.get("/api/catalogs/mobile-fuels", response_model=ApiEnvelope)
def mobile_fuels_catalog() -> ApiEnvelope:
    return ApiEnvelope(data={"items": list_mobile_fuels()})


@app.get("/api/catalogs/refrigerants", response_model=ApiEnvelope)
def refrigerants_catalog() -> ApiEnvelope:
    return ApiEnvelope(data={"items": list_refrigerants()})


@app.get("/api/catalogs/electricity-suppliers", response_model=ApiEnvelope)
def electricity_suppliers_catalog(year: int = 2025) -> ApiEnvelope:
    return ApiEnvelope(data={"items": list_electricity_suppliers(year), "year": year})
