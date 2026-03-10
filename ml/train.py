import numpy as np
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

X, y = make_classification(
    n_samples=50_000, n_features=7,
    weights=[0.98, 0.02], random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=100,
    scale_pos_weight=49,   
    random_state=42,
    eval_metric='auc'
)
model.fit(X_train_s, y_train)

auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
print(f'AUC-ROC: {auc:.4f}') 

joblib.dump(model,  'ml/model.pkl') 
joblib.dump(scaler, 'ml/preprocessor.pkl') 
print('Model saved to ml/model.pkl') 