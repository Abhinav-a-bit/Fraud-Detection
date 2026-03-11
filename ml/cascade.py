import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, f1_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CascadingFraudDetector:
   
    def __init__(self, 
                 anomaly_percentile=95,    # Top 5% most anomalous
                 contamination_rate=0.05,  # Expected anomaly rate
                 pos_weight=100,           # Handle imbalance
                 min_confidence=0.3,       # Min confidence for classifier
                 cache_size=10000):        # Recent transactions cache
        
        self.anomaly_percentile = anomaly_percentile
        self.contamination_rate = contamination_rate
        self.pos_weight = pos_weight
        self.min_confidence = min_confidence
        self.cache_size = cache_size
        
        self.robust_scaler = RobustScaler()
        self.standard_scaler = StandardScaler()
        
        # Stage 1: Fast anomaly detection
        self.anomaly_detector = IsolationForest(
            contamination=contamination_rate,
            random_state=42,
            n_estimators=100,
            max_samples=256,
            bootstrap=True,
            n_jobs=-1,
            warm_start=True  
        )
        
        # Stage 2: Deep supervised learning
        self.classifier = xgb.XGBClassifier(
            scale_pos_weight=pos_weight,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='aucpr',
            early_stopping_rounds=20 
        )
        
        # Performance tracking
        self.time_stage1 = []
        self.time_stage2 = []
        self.stats = {
            'total_txns': 0,
            'filtered_by_anomaly': 0,
            'processed_by_classifier': 0,
            'avg_confidence': []
        }
        
        # Recent transactions cache (for adaptive thresholding)
        self.recent_scores = []
        self.recent_txns = []
        
    def fit(self, X, y, feature_names=None, sample_weight=None):
        print("Training Cascading Fraud Detection System")
        
        self.feature_names = feature_names if feature_names else [f'V{i}' for i in range(X.shape[1])]
        
        print("Scaling features")
        X_robust = self.robust_scaler.fit_transform(X)
        X_scaled = self.standard_scaler.fit_transform(X_robust)
        
        # 1. Train Stage 1 (Anomaly Detection) on ALL data
        print("Training Stage 1: Isolation Forest")
        self.anomaly_detector.fit(X_scaled)
        
        scores = self.anomaly_detector.score_samples(X_scaled)
        
        self.anomaly_threshold = np.percentile(scores, 100 - self.anomaly_percentile)
        print(f"Stage 1 threshold: {self.anomaly_threshold:.4f} ({self.anomaly_percentile}th percentile)")
        
        # 2. Select suspicious transactions for Stage 2
        is_suspicious = scores <= self.anomaly_threshold
        X_sus = X_scaled[is_suspicious]
        y_sus = y[is_suspicious]
        
        print(f"Stage 2 training: {len(X_sus)} transactions ({len(X_sus)/len(X)*100:.1f}% of total)")
        print(f"Fraud rate in Stage 2 set: {y_sus.mean()*100:.2f}%")
        
        if len(X_sus) == 0:
            print("Warning: No suspicious transactions found. Using all data for Stage 2.")
            X_sus = X_scaled
            y_sus = y
        
        print("Training Stage 2: XGBoost")
        
        # Sample weights for Stage 2
        if sample_weight is not None:
            clf_weights = sample_weight[is_suspicious]
        else:
            clf_weights = np.abs(scores[is_suspicious] - self.anomaly_threshold)
            clf_weights = clf_weights / clf_weights.max()
        
        # Train with early stopping
        self.classifier.fit(
            X_sus, y_sus,
            sample_weight=clf_weights,
            eval_set=[(X_sus, y_sus)],
            verbose=False
        )
        
        print("Training complete.")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def predict_proba(self, X, return_stage_info=False):
        """Predict probability of fraud with cascading logic"""
        start_time = datetime.now()
        X_robust = self.robust_scaler.transform(X)
        X_scaled = self.standard_scaler.transform(X_robust)
        
        # Stage 1: Get anomaly scores
        scores = self.anomaly_detector.score_samples(X_scaled)
        
        # Track processing time
        t1 = (datetime.now() - start_time).total_seconds() * 1000
        self.time_stage1.append(t1)
        
        # Initialize results
        n_samples = len(X)
        fraud_probas = np.zeros(n_samples)
        stage_info = []
        
        # Process each transaction
        for i, (score, x) in enumerate(zip(scores, X_scaled)):
            self.stats['total_txns'] += 1
            
            if score > self.anomaly_threshold:
                base_proba = 1 / (1 + np.exp(score))  # Sigmoid transformation
                fraud_probas[i] = base_proba * 0.5    # Cap at 50% confidence
                stage_info.append({
                    'stage': 1,
                    'anomaly_score': score,
                    'confidence': 'low'
                })
                self.stats['filtered_by_anomaly'] += 1
                
            else:
                # SUSPICIOUS TRANSACTION - Deep analysis
                stage2_start = datetime.now()
                
                proba = self.classifier.predict_proba([x])[0][1]
                
                # Apply confidence weighting based on anomaly score
                confidence = np.exp(-abs(score - self.anomaly_threshold))
                final_proba = proba * (0.5 + 0.5 * confidence)
                
                fraud_probas[i] = final_proba
                
                t2 = (datetime.now() - stage2_start).total_seconds() * 1000
                self.time_stage2.append(t2)
                
                stage_info.append({
                    'stage': 2,
                    'anomaly_score': score,
                    'xgb_proba': proba,
                    'confidence': confidence,
                    'processing_time_ms': t2
                })
                self.stats['processed_by_classifier'] += 1
                self.stats['avg_confidence'].append(confidence)
        
        self.recent_scores.extend(scores.tolist())
        self.recent_scores = self.recent_scores[-self.cache_size:]
        
        if return_stage_info:
            return fraud_probas, stage_info
        return fraud_probas
    
    def get_performance_metrics(self):
        """Return comprehensive performance stats"""
        avg_t1 = np.mean(self.time_stage1) if self.time_stage1 else 0
        avg_t2 = np.mean(self.time_stage2) if self.time_stage2 else 0
        
        total = self.stats['total_txns']
        stage2_pct = (self.stats['processed_by_classifier'] / total * 100) if total > 0 else 0
        
        return {
            'avg_stage1_time_ms': avg_t1,
            'avg_stage2_time_ms': avg_t2,
            'speedup_factor': avg_t2 / (avg_t1 + 1e-10),
            'stage2_percentage': stage2_pct,
            'stage1_filtered_pct': 100 - stage2_pct,
            'avg_confidence': np.mean(self.stats['avg_confidence']) if self.stats['avg_confidence'] else 0,
            'total_processed': total
        }
    
    def explain_prediction(self, X, index=0):
        probas, stage_info = self.predict_proba([X[index]], return_stage_info=True)
        info = stage_info[0]
        
        explanation = {
            'fraud_probability': probas[0],
            'processing_stage': info['stage'],
            'anomaly_score': info['anomaly_score'],
            'threshold_passed': info['anomaly_score'] <= self.anomaly_threshold
        }
        
        if info['stage'] == 2:
            import shap
            explainer = shap.TreeExplainer(self.classifier)
            X_scaled = self.standard_scaler.transform(self.robust_scaler.transform([X[index]]))
            shap_values = explainer.shap_values(X_scaled)
            
            feature_impacts = pd.DataFrame({
                'feature': self.feature_names,
                'shap_value': shap_values[0]
            }).sort_values('shap_value', ascending=False)
            
            explanation['top_features'] = feature_impacts.head(5).to_dict('records')
            explanation['xgb_confidence'] = info['confidence']
        
        return explanation
    
    def save_model(self, filepath):
        model_data = {
            'robust_scaler': self.robust_scaler,
            'standard_scaler': self.standard_scaler,
            'anomaly_detector': self.anomaly_detector,
            'classifier': self.classifier,
            'anomaly_threshold': self.anomaly_threshold,
            'anomaly_percentile': self.anomaly_percentile,
            'pos_weight': self.pos_weight,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.robust_scaler = model_data['robust_scaler']
        self.standard_scaler = model_data['standard_scaler']
        self.anomaly_detector = model_data['anomaly_detector']
        self.classifier = model_data['classifier']
        self.anomaly_threshold = model_data['anomaly_threshold']
        self.anomaly_percentile = model_data['anomaly_percentile']
        self.pos_weight = model_data['pos_weight']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        print(f"Model loaded from {filepath}")
        return self