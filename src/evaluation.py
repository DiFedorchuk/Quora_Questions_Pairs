"""
Evaluation utilities for Quora Question Pairs models.
"""

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    log_loss,
)


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


def compute_f1_and_logloss(y_true, y_pred_proba, threshold=0.5):
    """
    Compute F1-score and log-loss from prediction probabilities.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for class 1
        threshold: Probability threshold for binary decision

    Returns:
        tuple: (f1_score_value, log_loss_value)
    """
    y_pred = (np.asarray(y_pred_proba) > threshold).astype(int)
    return f1_score(y_true, y_pred), log_loss(y_true, y_pred_proba)


def build_model_comparison_dataframe(model_names, f1_scores, log_losses):
    """
    Build a sorted model-comparison table.

    Args:
        model_names: Sequence of model names
        f1_scores: Sequence of F1-score values
        log_losses: Sequence of Log Loss values

    Returns:
        pandas.DataFrame: Sorted by F1-score descending
    """
    if not (len(model_names) == len(f1_scores) == len(log_losses)):
        raise ValueError("model_names, f1_scores, and log_losses must have identical lengths.")

    results = {
        "Model": model_names,
        "F1-score": f1_scores,
        "Log Loss": log_losses,
    }
    results_df = pd.DataFrame(results)
    return results_df.sort_values(by="F1-score", ascending=False).reset_index(drop=True)


def display_model_comparison(results_df):
    """
    Display model-comparison table in notebook-friendly form.

    Args:
        results_df: DataFrame returned by build_model_comparison_dataframe()
    """
    display(results_df)
