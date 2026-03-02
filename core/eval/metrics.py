"""
Classification metrics computation.

Computes confusion matrix, accuracy, precision, recall, F1, specificity,
and optionally ROC-AUC and PR-AUC when probability scores are available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared_models import (
    ClassificationMetrics,
    ConfusionMatrix,
)


def compute_metrics(
    matched_df: pd.DataFrame,
    pred_col: str = "pred_label",
    score_col: str | None = "score_ai",
    truth_col: str = "truth_label",
    uncertain_handling: str = "exclude",
) -> ClassificationMetrics:
    """
    Compute classification metrics from a matched DataFrame.

    Args:
        matched_df: DataFrame with truth_label and pred_label columns
        pred_col: column with predicted labels
        score_col: column with AI probability scores (optional)
        truth_col: column with ground truth labels
        uncertain_handling:
            "exclude" - drop rows where pred == uncertain (default)
            "third_class" - treat uncertain as its own class (basic)
            "map_to_human" - treat uncertain predictions as human

    Returns:
        ClassificationMetrics
    """
    df = matched_df.copy()

    # Handle uncertain predictions
    if uncertain_handling == "exclude":
        df = df[df[pred_col] != "uncertain"]
    elif uncertain_handling == "map_to_human":
        df[pred_col] = df[pred_col].replace("uncertain", "human")
    # "third_class" just uses them as-is for binary metrics (treated as non-AI)

    if df.empty:
        raise ValueError("No rows remain after filtering uncertain predictions.")

    # Binary labels: ai_generated = 1, human = 0
    truth_binary = (df[truth_col] == "ai_generated").astype(int)
    pred_binary = (df[pred_col] == "ai_generated").astype(int)

    tp = int(((truth_binary == 1) & (pred_binary == 1)).sum())
    tn = int(((truth_binary == 0) & (pred_binary == 0)).sum())
    fp = int(((truth_binary == 0) & (pred_binary == 1)).sum())
    fn = int(((truth_binary == 1) & (pred_binary == 0)).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    specificity = tn / max(tn + fp, 1)

    support_ai = int(truth_binary.sum())
    support_human = int((1 - truth_binary).sum())

    # Optional: ROC-AUC and PR-AUC
    roc_auc: float | None = None
    pr_auc: float | None = None

    if score_col and score_col in df.columns:
        scores = df[score_col].fillna(0.5).astype(float)
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            if len(truth_binary.unique()) == 2:
                roc_auc = float(roc_auc_score(truth_binary, scores))
                pr_auc = float(average_precision_score(truth_binary, scores))
        except Exception:
            pass

    return ClassificationMetrics(
        confusion_matrix=ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn),
        accuracy=round(accuracy, 4),
        precision_ai=round(precision, 4),
        recall_ai=round(recall, 4),
        f1_ai=round(f1, 4),
        specificity=round(specificity, 4),
        support_ai=support_ai,
        support_human=support_human,
        roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
        pr_auc=round(pr_auc, 4) if pr_auc is not None else None,
        uncertain_handling=uncertain_handling,
    )


def compute_roc_curve(
    matched_df: pd.DataFrame,
    score_col: str = "score_ai",
    truth_col: str = "truth_label",
) -> tuple[list[float], list[float], list[float]]:
    """
    Compute ROC curve data.
    Returns (fpr_list, tpr_list, thresholds_list).
    """
    from sklearn.metrics import roc_curve

    df = matched_df[matched_df[score_col].notna()].copy()
    if df.empty:
        return [], [], []

    truth_binary = (df[truth_col] == "ai_generated").astype(int)
    scores = df[score_col].astype(float)

    fpr, tpr, thresholds = roc_curve(truth_binary, scores)
    return fpr.tolist(), tpr.tolist(), thresholds.tolist()


def compute_pr_curve(
    matched_df: pd.DataFrame,
    score_col: str = "score_ai",
    truth_col: str = "truth_label",
) -> tuple[list[float], list[float], list[float]]:
    """
    Compute Precision-Recall curve data.
    Returns (precision_list, recall_list, thresholds_list).
    """
    from sklearn.metrics import precision_recall_curve

    df = matched_df[matched_df[score_col].notna()].copy()
    if df.empty:
        return [], [], []

    truth_binary = (df[truth_col] == "ai_generated").astype(int)
    scores = df[score_col].astype(float)

    precision, recall, thresholds = precision_recall_curve(truth_binary, scores)
    return precision.tolist(), recall.tolist(), thresholds.tolist()


def stratified_metrics(
    matched_df: pd.DataFrame,
    group_col: str,
    pred_col: str = "pred_label",
    truth_col: str = "truth_label",
    uncertain_handling: str = "exclude",
) -> pd.DataFrame:
    """
    Compute per-group metrics for a stratification column.
    Returns a DataFrame with one row per group.
    """
    rows = []
    for group_val, group_df in matched_df.groupby(group_col):
        if len(group_df) < 5:  # skip tiny groups
            continue
        try:
            m = compute_metrics(
                group_df,
                pred_col=pred_col,
                truth_col=truth_col,
                uncertain_handling=uncertain_handling,
            )
            rows.append({
                "group": str(group_val),
                "count": len(group_df),
                "accuracy": m.accuracy,
                "precision_ai": m.precision_ai,
                "recall_ai": m.recall_ai,
                "f1_ai": m.f1_ai,
                "support_ai": m.support_ai,
                "support_human": m.support_human,
            })
        except Exception:
            pass

    return pd.DataFrame(rows) if rows else pd.DataFrame()
