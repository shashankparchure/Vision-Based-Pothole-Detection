import argparse
import os
import sys
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from classifier import classify_severity
from features import extract_depth_features
from segmentation import get_pothole_mask


_DEPTH_MODEL: Optional[object] = None


def _resolve_depth_anything_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "Depth-Anything-V2"),
        os.path.abspath(os.path.join(script_dir, "..", "Depth-Anything-V2")),
    ]

    for path in candidates:
        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        "Could not find Depth-Anything-V2. Expected one of: " + ", ".join(candidates)
    )


def _load_depth_model():
    global _DEPTH_MODEL

    if _DEPTH_MODEL is not None:
        return _DEPTH_MODEL

    depth_anything_root = _resolve_depth_anything_root()
    if depth_anything_root not in sys.path:
        sys.path.append(depth_anything_root)

    from depth_anything_v2.dpt import DepthAnythingV2

    model_path = os.path.join(
        depth_anything_root, "checkpoints", "depth_anything_v2_vits.pth"
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Depth checkpoint not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    _DEPTH_MODEL = model
    return _DEPTH_MODEL


def get_depth_map(image: np.ndarray) -> np.ndarray:
    """Infer an HxW float32 depth map from a BGR image."""
    if image is None:
        raise ValueError("image cannot be None")

    model = _load_depth_model()
    depth_map = model.infer_image(image).astype(np.float32)

    if depth_map.shape != image.shape[:2]:
        depth_map = cv2.resize(
            depth_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    return depth_map


def run_pipeline(
    image_path: str, output_dir: Optional[str] = None, show: bool = True
) -> None:
    mask, original_image = get_pothole_mask(image_path)
    depth_map = get_depth_map(original_image)

    if mask.shape != depth_map.shape:
        depth_map = cv2.resize(
            depth_map,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    features = extract_depth_features(mask, depth_map)
    severity = classify_severity(features)

    if features is None:
        print("mean_depth: None")
        print("max_depth: None")
        print("area: 0")
    else:
        print(f"mean_depth: {features['mean_depth']:.6f}")
        print(f"max_depth: {features['max_depth']:.6f}")
        print(f"area: {features['area']}")
    print(f"severity: {severity}")

    text_overlay = original_image.copy()
    cv2.putText(
        text_overlay,
        f"Severity: {severity}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    depth_heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    mask_vis = (mask * 255).astype(np.uint8)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image + Severity")
    plt.imshow(cv2.cvtColor(text_overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Pothole Mask")
    plt.imshow(mask_vis, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Depth Map Heatmap")
    plt.imshow(cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{image_name}_result.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"saved_visualization: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pothole severity pipeline")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory to save visualization image",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Disable interactive plot display",
    )
    args = parser.parse_args()

    run_pipeline(args.image_path, output_dir=args.output_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
