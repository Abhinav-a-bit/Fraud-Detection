import joblib
import numpy as np
from pathlib import Path

_model = None
_scaler = None

def load_model():
    global _model, _scaler
    _model  = joblib.load(Path('ml/model.pkl'))
    _scaler = joblib.load(Path('ml/preprocessor.pkl'))
    print('Model loaded successfully')

def predict(features: list[float]) -> float:
    if _model is None:
        raise RuntimeError('Model not loaded. Call load_model() first.')
    X = _scaler.transform([features])
    prob = _model.predict_proba(X)[0][1]
    return round(float(prob), 4)