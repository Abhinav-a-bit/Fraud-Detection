from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="txn_abc123")
    amount: float = Field(..., gt=0)
    merchant_category: str
    hour_of_day: int = Field(..., ge=0, le=23)
    distance_from_home_km: float
    is_foreign: bool
    card_present: bool

class FraudResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_label: str  
    model_version: str
    processed_at: datetime