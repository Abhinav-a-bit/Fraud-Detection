from sqlalchemy.orm import Session
from app.services import ml_service, cache_service
from app.db import crud

def process_fraud_prediction(db: Session, transaction_data: dict):
    txn_id = transaction_data["transaction_id"]
    
    cached_result = cache_service.get_prediction(txn_id)
    if cached_result:
        print(f"CACHE HIT! Returning instant result for {txn_id}")
        return cached_result

    features = [transaction_data["Time"]]
    for i in range(1, 29):
        features.append(transaction_data[f"V{i}"])
    features.append(transaction_data["Amount"])

    # 2. Get the probability from our Cascading ML service
    fraud_prob = ml_service.predict(features)

    # 3. Determine the Risk Label
    if fraud_prob < 0.4:
        risk_label = "LOW"
    elif fraud_prob <= 0.7:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    # 4. Save to database
    db_transaction = crud.save_transaction(
        db=db, 
        transaction_data=transaction_data, 
        fraud_score=fraud_prob, 
        risk_label=risk_label
    )

    # 5. Format the response
    result = {
        "transaction_id": db_transaction.transaction_id,
        "fraud_probability": db_transaction.fraud_probability,
        "risk_label": db_transaction.risk_label,
        "model_version": "cascade-v1.0",
        "processed_at": db_transaction.processed_at.isoformat()
    }
    
    # 6. SAVE TO CACHE for the next 60 seconds
    cache_service.set_prediction(txn_id, result)

    return result

def explain_fraud_prediction(transaction_data: dict):
    features = [transaction_data["Time"]]
    for i in range(1, 29):
        features.append(transaction_data[f"V{i}"])
    features.append(transaction_data["Amount"])

    explanation = ml_service.explain(features)
    explanation["transaction_id"] = transaction_data["transaction_id"]
    return explanation