from sqlalchemy.orm import Session
from app.models import db_models, schemas

def save_transaction(db: Session, transaction_data: dict, fraud_score: float, risk_label: str):
    db_transaction = db_models.Transaction(
        transaction_id=transaction_data["transaction_id"],
        amount=transaction_data["amount"],
        merchant_category=transaction_data["merchant_category"],
        fraud_probability=fraud_score,
        risk_label=risk_label
    )
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    return db_transaction

def get_transaction_by_id(db: Session, transaction_id: str):
    return db.query(db_models.Transaction).filter(
        db_models.Transaction.transaction_id == transaction_id
    ).first()