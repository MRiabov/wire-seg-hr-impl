import torch

from wireseghr.model.losses import DiceLoss


def test_dice_loss_rewards_correct_logits():
    loss_fn = DiceLoss()

    target = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.long)
    good_logits = torch.tensor(
        [[
            [[-3.0, 3.0], [3.0, -3.0]],
            [[3.0, -3.0], [-3.0, 3.0]],
        ]],
        dtype=torch.float32,
    )
    bad_logits = -good_logits

    good = loss_fn(good_logits, target)
    bad = loss_fn(bad_logits, target)

    assert good.item() < bad.item()
    assert good.item() >= 0.0
    assert bad.item() >= 0.0
