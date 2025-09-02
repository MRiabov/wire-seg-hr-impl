from typing import Dict
import numpy as np


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Compute binary segmentation metrics on 0/1 numpy masks.

    Args:
        pred_mask: HxW uint8 or bool in {0,1}
        gt_mask:   HxW uint8 or bool in {0,1}
    Returns:
        dict with iou, f1, precision, recall
    """
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)

    tp = int(np.sum((p == 1) & (g == 1)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))

    denom_iou = tp + fp + fn
    iou = (tp / denom_iou) if denom_iou > 0 else 0.0

    prec_den = tp + fp
    rec_den = tp + fn
    precision = (tp / prec_den) if prec_den > 0 else 0.0
    recall = (tp / rec_den) if rec_den > 0 else 0.0

    denom_f1 = precision + recall
    f1 = (2 * precision * recall / denom_f1) if denom_f1 > 0 else 0.0

    return {"iou": float(iou), "f1": float(f1), "precision": float(precision), "recall": float(recall)}
