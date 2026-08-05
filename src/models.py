"""
Machine learning models for Quora Question Pairs duplicate detection.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import os


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
