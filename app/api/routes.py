from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models import schemas
from app.services import fraud_service

router = APIRouter()

@router.post("/predict", response_model=schemas.FraudResponse)
def predict_fraud(payload: schemas.TransactionRequest, db: Session = Depends(get_db)):
    try:
        result = fraud_service.process_fraud_prediction(db, payload.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/transactions")
def list_transactions(db: Session = Depends(get_db)):
    from app.models.db_models import Transaction
    return db.query(Transaction).all()