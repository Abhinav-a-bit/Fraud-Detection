from sqlalchemy.orm import Session
from app.models import db_models

def save_transaction(db: Session, transaction_data: dict, fraud_score: float, risk_label: str):
    db_transaction = db_models.Transaction(
        transaction_id=transaction_data["transaction_id"],
        time_feature=transaction_data["Time"],
        amount=transaction_data["Amount"],
        fraud_probability=fraud_score,
        risk_label=risk_label
    )
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    return db_transaction