from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.db.database import get_db
from app.models import schemas
from app.services import fraud_service

router = APIRouter()

@router.post("/predict", response_model=schemas.FraudResponse)
def predict_transaction(request: schemas.TransactionRequest, db: Session = Depends(get_db)):
    try:
        result = fraud_service.process_fraud_prediction(db, request.model_dump())
        return result
        
    except IntegrityError:
        # If the database screams about a duplicate ID, we rollback.
        db.rollback()
        raise HTTPException(
            status_code=409, 
            detail="Conflict: A transaction with this ID already exists and has been processed."
        )


@router.get("/transactions")
def list_transactions(db: Session = Depends(get_db)):
    from app.models.db_models import Transaction
    return db.query(Transaction).all()

@router.post("/explain", response_model=schemas.ExplanationResponse)
def explain_transaction(request: schemas.TransactionRequest):
    explanation = fraud_service.explain_fraud_prediction(request.model_dump())
    return explanation