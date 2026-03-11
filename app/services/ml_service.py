import os
import numpy as np
from ml.cascade import CascadingFraudDetector

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../ml/cascade_model.pkl')
detector = None

def load_model():
    global detector
    try:
        # Initialize your custom class
        detector = CascadingFraudDetector()
        # Use your custom method to load the weights and scalers
        detector.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

def predict(features: list) -> float:
    if detector is None:
        raise RuntimeError("Model is not loaded.")
    
    features_array = np.array([features])
    
    fraud_prob = detector.predict_proba(features_array)[0]
    
    return float(fraud_prob)
def explain(features: list) -> dict:
    if detector is None:
        raise RuntimeError("Model is not loaded.")
    
    features_array = np.array([features])
    
    explanation = detector.explain_prediction(features_array, index=0)
    
    return explanation