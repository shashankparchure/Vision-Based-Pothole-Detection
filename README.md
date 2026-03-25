# Vision-Based-Pothole-Detection

First step is to generate depth maps for the images using DepthAnythingV2 model. 

This project uses Depth-Anything-V2 to generate depth maps for pothole dataset images.

## Folder Structure (important parts)

```text
CVCSL7360/
├── Depth-Anything-V2/                  # clone this repo here
└── Vision-Based-Pothole-Detection/
	├── data/
	│   ├── train/images/
	│   ├── valid/images/
	│   └── test/images/
	├── depth/
	│   └── generate_depth.py
	└── depth_maps/                     # output depth .npy files
```

## 1) Clone Depth-Anything-V2 in the correct location

From `CVCSL7360` root:

```powershell
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

Expected path after cloning:

`CVCSL7360/Depth-Anything-V2`

Also make sure the checkpoint exists at:

`Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth`

## 2) Run depth map generation

Go to the project folder and run:

```powershell
cd Vision-Based-Pothole-Detection
python depth/generate_depth.py
```

The script reads images from:

- `data/train/images`
- `data/valid/images`
- `data/test/images`

And saves depth maps to:

- `depth_maps/train`
- `depth_maps/valid`
- `depth_maps/test`