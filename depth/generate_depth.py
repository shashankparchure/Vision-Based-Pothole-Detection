import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add DepthAnythingV2 to path
import sys
sys.path.append(os.path.abspath("../Depth-Anything-V2"))

from depth_anything_v2.dpt import DepthAnythingV2

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "../Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"

DATA_PATHS = {
    "train": "data/train/images",
    "valid": "data/valid/images",
    "test": "data/test/images"
}

OUTPUT_PATH = "depth_maps"

# -------------------------
# LOAD MODEL
# -------------------------
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# -------------------------
# FUNCTION: process one image
# -------------------------
def get_depth(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = model.infer_image(image)
    return depth

# -------------------------
# MAIN LOOP
# -------------------------
for split in DATA_PATHS:
    input_dir = DATA_PATHS[split]
    output_dir = os.path.join(OUTPUT_PATH, split)

    os.makedirs(output_dir, exist_ok=True)

    images = os.listdir(input_dir)

    print(f"\nProcessing {split}...")

    for img_name in tqdm(images):
        img_path = os.path.join(input_dir, img_name)

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Get depth
        depth = get_depth(image)

        # Normalize depth (IMPORTANT)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Save as .npy
        save_name = os.path.splitext(img_name)[0] + ".npy"
        save_path = os.path.join(output_dir, save_name)

        np.save(save_path, depth)

print("\nDepth maps generated successfully!")
