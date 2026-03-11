from pydantic import BaseModel
from datetime import datetime

class TransactionRequest(BaseModel):
    transaction_id: str
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class FraudResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_label: str
    model_version: str
    processed_at: datetime

from typing import List, Optional

class FeatureImpact(BaseModel):
    feature: str
    shap_value: float

class ExplanationResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    processing_stage: int
    anomaly_score: float
    threshold_passed: bool
    top_features: Optional[List[FeatureImpact]] = None
    xgb_confidence: Optional[float] = None