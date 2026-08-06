"""
Machine learning models for Quora Question Pairs duplicate detection.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import pickle
import os

ENSEMBLE_MODEL_ORDER = [
    "Baseline Logistic Regression",
    "BOW Logistic Regression",
    "TF-IDF Logistic Regression",
    "Logistic Regression (Tuned)",
    "XGBoost (Untuned)",
    "XGBoost (Tuned)",
    "LightGBM (Untuned)",
    "LightGBM (Tuned)",
    "LSTM",
    "BERT Embeddings Logistic Regression",
    "BERT (Fine-tuned)",
    "Ensemble Model",
]


class QuestionPairModel:
    """Base class for question pair similarity models."""
    
    def __init__(self, name="QuestionPairModel"):
        self.name = name
        self.model = None
        
    def fit(self, X, y):
        """Fit the model."""
        raise NotImplementedError
        
    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError
        
    def predict_proba(self, X):
        """Get prediction probabilities."""
        raise NotImplementedError
        
    def save(self, filepath):
        """Save model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
        
    def load(self, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")


class LogisticRegressionModel(QuestionPairModel):
    """Logistic Regression model for question pair classification."""
    
    def __init__(self, **kwargs):
        super().__init__("LogisticRegression")
        self.model = LogisticRegression(**kwargs)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class RandomForestModel(QuestionPairModel):
    """Random Forest model for question pair classification."""
    
    def __init__(self, n_estimators=100, **kwargs):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def feature_importance(self):
        """Get feature importances."""
        return self.model.feature_importances_


class GradientBoostingModel(QuestionPairModel):
    """Gradient Boosting model for question pair classification."""
    
    def __init__(self, n_estimators=100, **kwargs):
        super().__init__("GradientBoosting")
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, **kwargs)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def feature_importance(self):
        """Get feature importances."""
        return self.model.feature_importances_


class SVMModel(QuestionPairModel):
    """Support Vector Machine model for question pair classification."""
    
    def __init__(self, kernel='rbf', **kwargs):
        super().__init__("SVM")
        self.model = SVC(kernel=kernel, probability=True, **kwargs)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def build_ensemble_weights(weight_bert, weight_xgb, weight_lgbm):
    """
    Build a serializable ensemble-weight configuration.

    Args:
        weight_bert: Weight for BERT-based predictions
        weight_xgb: Weight for tuned XGBoost predictions
        weight_lgbm: Weight for tuned LightGBM predictions

    Returns:
        dict: Ensemble weight mapping
    """
    return {
        "weight_bert": weight_bert,
        "weight_xgb": weight_xgb,
        "weight_lgbm": weight_lgbm,
    }


def save_ensemble_components(
    output_dir,
    ensemble_weights,
    model_xgb_tuned,
    scaler_xgb,
    model_lgbm_tuned,
    scaler_lgbm,
    model_bert_fine_tune=None,
    tokenizer=None,
):
    """
    Persist ensemble artifacts to disk.

    Args:
        output_dir: Directory where artifacts are saved
        ensemble_weights: Dict returned by build_ensemble_weights()
        model_xgb_tuned: Trained tuned XGBoost model
        scaler_xgb: Fitted scaler used for tuned XGBoost
        model_lgbm_tuned: Trained tuned LightGBM model
        scaler_lgbm: Fitted scaler used for tuned LightGBM
        model_bert_fine_tune: Optional fine-tuned torch model
        tokenizer: Optional tokenizer paired with model_bert_fine_tune
    """
    if (model_bert_fine_tune is None) != (tokenizer is None):
        raise ValueError("model_bert_fine_tune and tokenizer must both be provided or both omitted.")

    print("Saving Ensemble Model components...")
    os.makedirs(output_dir, exist_ok=True)

    print("Saving Ensemble weights...")
    joblib.dump(ensemble_weights, os.path.join(output_dir, "ensemble_weights.joblib"))

    print("Saving Tuned XGBoost model and its scaler...")
    joblib.dump(model_xgb_tuned, os.path.join(output_dir, "tuned_xgboost_model.joblib"))
    joblib.dump(scaler_xgb, os.path.join(output_dir, "tuned_xgboost_scaler.joblib"))

    print("Saving Tuned LightGBM model and its scaler...")
    joblib.dump(model_lgbm_tuned, os.path.join(output_dir, "tuned_lightgbm_model.joblib"))
    joblib.dump(scaler_lgbm, os.path.join(output_dir, "tuned_lightgbm_scaler.joblib"))

    if model_bert_fine_tune is not None and tokenizer is not None:
        import torch

        torch.save(
            model_bert_fine_tune.state_dict(),
            os.path.join(output_dir, "fine_tuned_bert_model_state_dict.pth"),
        )
        tokenizer.save_pretrained(os.path.join(output_dir, "fine_tuned_bert_tokenizer"))

    print("All Ensemble Model components have been saved.")
