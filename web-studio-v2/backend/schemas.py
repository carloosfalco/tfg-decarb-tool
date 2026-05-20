from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CalculateFootprintRequest(BaseModel):
    company_inputs: Dict[str, Any] = Field(default_factory=dict)


class GeneratePestelRequest(BaseModel):
    company_inputs: Dict[str, Any] = Field(default_factory=dict)
    footprint: Optional[Dict[str, Any]] = None
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite-preview"


class GenerateInitiativesRequest(BaseModel):
    company_inputs: Dict[str, Any] = Field(default_factory=dict)
    footprint: Dict[str, Any] = Field(default_factory=dict)


class GenerateAiInitiativesRequest(BaseModel):
    company_inputs: Dict[str, Any] = Field(default_factory=dict)
    footprint: Dict[str, Any] = Field(default_factory=dict)
    pestel: Optional[Dict[str, Any]] = None
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite-preview"
    n: int = 8


class FinancialParams(BaseModel):
    discount_rate: float = 0.08
    horizon: int = 8


class ComputeMetricsRequest(BaseModel):
    initiatives: List[Dict[str, Any]] = Field(default_factory=list)
    financial_params: FinancialParams = Field(default_factory=FinancialParams)


class OptimizeWeights(BaseModel):
    w_npv: float = 0.30
    w_co2: float = 0.70
    w_strategy: float = 0.0


class OptimizePortfolioRequest(BaseModel):
    initiatives_with_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    budget_eur: float = 0.0
    min_co2_t: float = 0.0
    objective: str = "Balanced score (NPV + CO2 + strategy)"
    weights: OptimizeWeights = Field(default_factory=OptimizeWeights)


class ApiEnvelope(BaseModel):
    ok: bool = True
    data: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None
