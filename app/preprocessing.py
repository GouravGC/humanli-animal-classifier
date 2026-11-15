
import numpy as np
import torch
from PIL import Image


IMAGE_SIZE = 224

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert an image into a normalized CNN input tensor."""

    image = image.convert("RGB")
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    array = np.transpose(array, (2, 0, 1)).copy()

    return torch.from_numpy(array).unsqueeze(0)
