import numpy as np

from wireseghr.data.transforms import TrainTransforms


def test_train_transforms_noop_when_disabled():
    tfm = TrainTransforms({"enable": False})
    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    mask = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8)

    out_img, out_mask = tfm(image, mask)

    assert out_img.dtype == np.uint8
    assert out_mask.dtype == np.uint8
    assert np.array_equal(out_img, image)
    assert np.array_equal(out_mask, mask)


def test_train_transforms_forced_horizontal_flip():
    tfm = TrainTransforms(
        {
            "enable": True,
            "hflip_prob": 1.0,
            "rotation_deg": 0.0,
            "scale_min": 1.0,
            "scale_max": 1.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.0,
            "blur_prob": 0.0,
            "noise_std": 0.0,
            "jpeg_prob": 0.0,
            "cutout_prob": 0.0,
        }
    )
    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[11, 21, 31], [41, 51, 61], [71, 81, 91]],
        ],
        dtype=np.uint8,
    )
    mask = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)

    out_img, out_mask = tfm(image, mask)

    assert np.array_equal(out_img, image[:, ::-1])
    assert np.array_equal(out_mask, mask[:, ::-1])
