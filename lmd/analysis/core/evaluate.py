import numpy as np
from typing import Dict
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from .types import MetricDirection

def normalize_scores(scores: np.ndarray, direction: MetricDirection) -> np.ndarray:
    if direction == MetricDirection.LOWER_BETTER:
        return -scores
    return scores

def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    mask = np.isfinite(scores)
    if not np.all(mask):
        median_val = np.median(scores[mask]) if np.any(mask) else 0.0
        scores = np.where(mask, scores, median_val)
    scores = np.clip(scores, -1e10, 1e10)
    return scores

def evaluate_metric(labels: np.ndarray, scores: np.ndarray, 
                   direction: MetricDirection = MetricDirection.HIGHER_BETTER) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    scores = sanitize_scores(scores)
    
    if len(np.unique(labels)) < 2:
        base_rate = float(np.mean(labels))
        return {'auroc': 0.5, 'aupr': base_rate, 'fpr95': 1.0}
    
    s = normalize_scores(scores, direction)
    
    try:
        auroc = roc_auc_score(labels, s)
    except:
        auroc = 0.5
    
    try:
        precision, recall, _ = precision_recall_curve(labels, s)
        aupr = auc(recall, precision)
    except:
        aupr = float(np.mean(labels))
    
    try:
        fpr, tpr, _ = roc_curve(labels, s)
        mask = (tpr >= 0.95)
        fpr95 = float(np.min(fpr[mask])) if np.any(mask) else 1.0
    except:
        fpr95 = 1.0
    
    return {
        'auroc': float(auroc),
        'aupr': float(aupr),
        'fpr95': float(fpr95)
    }

def precision_recall_curve(y_true, y_score):
    from sklearn.metrics import precision_recall_curve as prc
    return prc(y_true, y_score)