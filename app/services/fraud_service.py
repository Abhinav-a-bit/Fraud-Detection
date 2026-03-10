from sqlalchemy.orm import Session
from app.services import ml_service
from app.db import crud

def process_fraud_prediction(db: Session, transaction_data: dict):
    features = [
        transaction_data["amount"],
        len(transaction_data["merchant_category"]), 
        transaction_data["hour_of_day"],
        transaction_data["distance_from_home_km"],
        1.0 if transaction_data["is_foreign"] else 0.0,
        1.0 if transaction_data["card_present"] else 0.0,
        0.0 
    ]

    fraud_prob = ml_service.predict(features)

    if fraud_prob < 0.4:
        risk_label = "LOW"
    elif fraud_prob <= 0.7:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    db_transaction = crud.save_transaction(
        db=db, 
        transaction_data=transaction_data, 
        fraud_score=fraud_prob, 
        risk_label=risk_label
    )

    return {
        "transaction_id": db_transaction.transaction_id,
        "fraud_probability": db_transaction.fraud_probability,
        "risk_label": db_transaction.risk_label,
        "model_version": "xgb-v1.0",
        "processed_at": db_transaction.processed_at
    }