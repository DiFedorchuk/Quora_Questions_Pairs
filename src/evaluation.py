"""
Evaluation utilities for Quora Question Pairs models.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC-AUC), optional
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    
    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    
    return metrics


def print_evaluation_report(metrics):
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics from evaluate_model()
    """
    print("=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    
    if "roc_auc" in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    print("=" * 50)
