import torch

from wireseghr.model import WireSegHR


def test_wireseghr_forward_shapes():
    # Use small input to keep test light and avoid downloading weights
    model = WireSegHR(backbone="mit_b3", in_channels=3, pretrained=False)

    x = torch.randn(1, 3, 64, 64)
    logits_coarse, cond = model.forward_coarse(x)
    assert logits_coarse.shape[0] == 1 and logits_coarse.shape[1] == 2
    assert cond.shape[0] == 1 and cond.shape[1] == 1
    # Expect stage 0 resolution ~ 1/4 of input for MiT
    assert logits_coarse.shape[2] == 16 and logits_coarse.shape[3] == 16
    assert cond.shape[2] == 16 and cond.shape[3] == 16

    logits_fine = model.forward_fine(x)
    assert logits_fine.shape == logits_coarse.shape
