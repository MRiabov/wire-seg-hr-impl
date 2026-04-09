import numpy as np

from wireseghr.metrics import compute_metrics


def test_compute_metrics_perfect_match():
    pred = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    gt = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    out = compute_metrics(pred, gt)

    assert out["iou"] == 1.0
    assert out["f1"] == 1.0
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0


def test_compute_metrics_empty_prediction():
    pred = np.zeros((2, 2), dtype=np.uint8)
    gt = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    out = compute_metrics(pred, gt)

    assert out["iou"] == 0.0
    assert out["precision"] == 0.0
    assert out["recall"] == 0.0
    assert out["f1"] == 0.0
