import torch

from wireseghr.model.stitching import get_patch_window


def test_patch_window_is_center_weighted_and_nonzero():
    win = get_patch_window(9, device=torch.device("cpu"))

    assert win.shape == (9, 9)
    assert torch.all(win > 0)
    assert torch.isclose(win.max(), torch.tensor(1.0), atol=1e-6)
    assert win[4, 4] > win[0, 0]
