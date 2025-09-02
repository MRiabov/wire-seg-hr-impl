# Training-time transforms: scaling, rotation, flip, photometric distortion
# TODO: Implement deterministic transform composition for reproducibility

class TrainTransforms:
    def __init__(self):
        pass

    def __call__(self, image, mask):
        return image, mask
