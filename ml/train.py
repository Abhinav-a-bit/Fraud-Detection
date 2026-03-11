import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from cascade import CascadingFraudDetector # Assumes cascade.py is in the same folder

print("Loading Kaggle dataset...")
df = pd.read_csv('ml/data/creditcard.csv')

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nPhase 1 : Training")
detector = CascadingFraudDetector(
    anomaly_percentile=95, 
    contamination_rate=0.05,
    pos_weight=10 
)

# Train the model on the 80% split
detector.fit(X_train.values, y_train.values, feature_names=X.columns.tolist())

print("\nPhase 2: Testing")
# Ask the model to predict fraud on the 20% it has never seen
print("Scoring test dataset...")
probas = detector.predict_proba(X_test.values)

preds = (probas >= 0.5).astype(int)

print(f"\nROC-AUC Score: {roc_auc_score(y_test, probas):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds))

print("\nPhase 3: Single Transaction Simulation")
explanation = detector.explain_prediction(X_test.values, index=0)

print(f"Transaction True Class: {y_test.iloc[0]} (0=Legit, 1=Fraud)")
print(f"Predicted Fraud Probability: {explanation['fraud_probability']:.4%}")
print(f"Processed by Stage: {explanation['processing_stage']} "
      f"({'Isolation Forest' if explanation['processing_stage'] ==1 else 'XGBoost Deep Analysis'})")
print(f"Anomaly Score: {explanation['anomaly_score']:.4f}")

print("\n Phase 4: Saving")
detector.save_model('ml/cascade_model.pkl')
print("Model pipeline verified and saved! Ready for API integration.")