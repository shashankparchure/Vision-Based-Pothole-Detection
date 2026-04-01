import os
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "yolo-segmentation", "model", "best.pt"),
    os.path.join(SCRIPT_DIR, "model", "best.pt"),
]

MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.isfile(path)), None)
if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find pretrained weights 'best.pt'. Expected one of: "
        + ", ".join(MODEL_CANDIDATES)
    )

# Load YOLOv8 segmentation model once and reuse it for inference calls.
MODEL = YOLO(MODEL_PATH)


def get_pothole_mask(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run YOLOv8 segmentation on an image and return the largest pothole mask.

    Args:
        image_path: Path to input image.

    Returns:
        A tuple of (binary_mask, original_image) where:
        - binary_mask is an HxW np.uint8 array containing values {0, 1}
        - original_image is the loaded image in OpenCV BGR format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    results = MODEL.predict(source=image, imgsz=640, conf=0.25, verbose=False)
    result = results[0]

    if result.masks is None or len(result.masks.data) == 0:
        return np.zeros((height, width), dtype=np.uint8), image

    masks = result.masks.data.detach().cpu().numpy()
    binary_masks = (masks > 0.5).astype(np.uint8)

    # Select the mask with the largest foreground area.
    areas = binary_masks.reshape(binary_masks.shape[0], -1).sum(axis=1)
    largest_mask = binary_masks[int(np.argmax(areas))]

    if largest_mask.shape != (height, width):
        largest_mask = cv2.resize(
            largest_mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
        largest_mask = (largest_mask > 0).astype(np.uint8)

    return largest_mask, image
