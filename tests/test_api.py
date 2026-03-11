import uuid

VALID_PAYLOAD = {
  "transaction_id": "test-txn-001",
  "Time": 0.0,
  "V1": -1.359807, "V2": -0.072781, "V3": 2.536346, "V4": 1.378155, "V5": -0.338320,
  "V6": 0.462387, "V7": 0.239598, "V8": 0.098697, "V9": 0.363786, "V10": 0.090794,
  "V11": -0.551599, "V12": -0.617800, "V13": -0.991389, "V14": -0.311169, "V15": 1.468176,
  "V16": -0.470400, "V17": 0.207971, "V18": 0.025790, "V19": 0.403992, "V20": 0.251412,
  "V21": -0.018306, "V22": 0.277837, "V23": -0.110473, "V24": 0.066928, "V25": 0.128539,
  "V26": -0.189114, "V27": 0.133558, "V28": -0.021053,
  "Amount": 149.62
}

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'model': 'loaded'}

def test_predict_returns_200(client):
    response = client.post('/api/v1/predict', json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert 'fraud_probability' in data
    assert 0.0 <= data['fraud_probability'] <= 1.0
    assert data['risk_label'] in ['LOW', 'MEDIUM', 'HIGH']

def test_predict_missing_field_returns_422(client):
    response = client.post('/api/v1/predict', json={'Amount': 100.0})
    assert response.status_code == 422 # Unprocessable Entity


def test_duplicate_transaction_returns_409(client, monkeypatch):
    monkeypatch.setattr("app.services.cache_service.get_prediction", lambda txn_id: None)
    
    # Create a truly unique transaction ID so old Redis data doesn't interfere
    test_payload = VALID_PAYLOAD.copy()
    test_payload["transaction_id"] = f"test-txn-409-{uuid.uuid4()}"

    client.post('/api/v1/predict', json=test_payload)
    
    response = client.post('/api/v1/predict', json=test_payload)
    assert response.status_code == 409
    assert "Conflict" in response.json()["detail"]