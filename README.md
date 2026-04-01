# Vision-Based-Pothole-Detection

This project estimates pothole severity from road images by combining:

- YOLOv8 instance segmentation (pothole mask)
- Depth-Anything-V2 depth estimation (depth map)
- Rule-based severity classification (Shallow, Moderate, Deep, or No pothole)

Dataset source: https://www.kaggle.com/datasets/farzadnekouei/pothole-image-segmentation-dataset

## Updated Project Structure (important parts)

```text
Vision-Based-Pothole-Detection/
├── main.py
├── segmentation.py
├── features.py
├── classifier.py
├── depth/
│   └── generate_depth.py
├── data1/
│   ├── train/images/
│   └── valid/images/
├── yolo-segmentation/
│   └── model/best.pt
├── Depth-Anything-V2/
│   └── checkpoints/depth_anything_v2_vits.pth
└── output/
```

## Pipeline Files (one-line explanation)

- `segmentation.py`: Loads YOLOv8 segmentation weights and returns the largest pothole mask.
- `main.py`: Runs the full pipeline from image input to visualization and saved output.
- `features.py`: Extracts normalized depth-based features inside the pothole mask.
- `classifier.py`: Applies rule-based thresholds on depth and area to assign severity.
- `depth/generate_depth.py`: Batch-generates and stores `.npy` depth maps for dataset folders.

## 1) Setup

Run from project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy torch torchvision opencv-python matplotlib ultralytics dill tqdm
```

## 2) Get Depth-Anything-V2 from GitHub

Clone Depth-Anything-V2 from GitHub:

```powershell
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

The scripts support either of these locations:

- `Vision-Based-Pothole-Detection/Depth-Anything-V2`
- `../Depth-Anything-V2` (sibling folder of this project)

## 3) Get dataset from Kaggle

Download the dataset from:

- https://www.kaggle.com/datasets/farzadnekouei/pothole-image-segmentation-dataset

Place it in this project as `data1/` so images are available at:

- `data1/train/images`
- `data1/valid/images`

## 4) Required model files

Make sure these files exist:

- `Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth`
- `yolo-segmentation/model/best.pt`

## 5) Run the pipeline on 1 image

```powershell
python main.py data1/train/images/pic-1-_jpg.rf.49882cdb272111f43a6656b1494a4918.jpg --output_dir output --no_show
```

This prints key values:

- `mean_depth`
- `max_depth`
- `area`
- `severity`

And saves a result image to `output/`.

## 6) Run the pipeline on 2 images

```powershell
python main.py data1/train/images/pic-1-_jpg.rf.49882cdb272111f43a6656b1494a4918.jpg --output_dir output --no_show
python main.py data1/train/images/pic-1-_jpg.rf.8d95dd1d29760a2634a45cc7fdd84b31.jpg --output_dir output --no_show
```

## 7) Optional: show plots interactively

Remove `--no_show` to display the visualization window.

## 8) Optional: batch depth-map generation for dataset

`depth/generate_depth.py` expects dataset folders under `data/`:

- `data/train/images`
- `data/valid/images`
- `data/test/images`

If your dataset is in `data1/`, either rename `data1` to `data` or update `DATA_PATHS` in `depth/generate_depth.py`.

Then run:

```powershell
python depth/generate_depth.py
```

Generated depth files are saved to:

- `depth_maps/train`
- `depth_maps/valid`
- `depth_maps/test`
