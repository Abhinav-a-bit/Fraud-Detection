from sqlalchemy import Column, Integer, String, Float, DateTime
from app.db.database import Base
import datetime

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True)
    time_feature = Column(Float)
    amount = Column(Float)
    fraud_probability = Column(Float)
    risk_label = Column(String)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)