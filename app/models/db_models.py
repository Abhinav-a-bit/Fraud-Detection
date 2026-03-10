from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from app.db.database import Base
from datetime import datetime

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(64), unique=True, index=True)
    amount = Column(Float)
    merchant_category = Column(String(64))
    fraud_probability = Column(Float)
    risk_label = Column(String(8))
    processed_at = Column(DateTime, default=datetime.utcnow)