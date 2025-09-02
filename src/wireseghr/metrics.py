# Metrics placeholder: IoU, F1, Precision, Recall
# TODO: Implement proper metrics matching paper tables.

from typing import Dict


def compute_metrics(pred_mask, gt_mask) -> Dict[str, float]:
    # TODO: implement
    return {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
