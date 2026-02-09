# Defect Detector (YOLO Segmentation POC)

POC-grade defect segmentation pipeline using Python, Ultralytics YOLO, OpenCV, Albumentations, FastAPI, and Docker.

## Stack

- Python
- PyTorch + Ultralytics YOLO (segmentation)
- OpenCV
- Albumentations
- FastAPI (optional serving layer)
- Docker

## Repository Layout

```text
data/
  raw/        # untouched originals
  labeled/    # YOLO-seg dataset export (images + labels)
  splits/     # train/val/test .txt lists
configs/
  dataset.yaml
  train.yaml
src/
  train.py
  infer.py
  postprocess.py
  calibration.py
  schemas.py
  prepare_dataset.py
  build_splits.py
models/
  weights/    # best.pt, model.onnx
demo/
  app.py      # streamlit demo
api/
  main.py     # fastapi app
```

## Data Format Expected

Training expects YOLO segmentation format under:

```text
data/labeled/
  images/
    *.jpg|*.png|...
  labels/
    *.txt  # normalized class + polygon points
```

Notes:
- This repo already contains `data/images` + `data/annotations` (VOC XML). `src.prepare_dataset` converts them into YOLO-seg labels.
- For this bootstrap conversion, each VOC box is emitted as a 4-point rectangle polygon (good for POC flow validation, not mask-quality ground truth).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Prepare Dataset (from included VOC XML)

```bash
python -m src.prepare_dataset ^
  --annotations-dir data/annotations ^
  --images-dir data/images ^
  --out-images-dir data/labeled/images ^
  --out-labels-dir data/labeled/labels ^
  --splits-dir data/splits ^
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

If your labels already exist in YOLO-seg format (Roboflow/Label Studio export), only build splits:

```bash
python -m src.build_splits ^
  --images-dir data/labeled/images ^
  --labels-dir data/labeled/labels ^
  --out-dir data/splits
```

## Train

```bash
python -m src.train --config configs/train.yaml
```

Artifacts:
- `models/weights/best.pt`
- `models/weights/model.onnx` (if export enabled)

## Inference (CLI)

```bash
python -m src.infer --weights models/weights/best.pt --image path\to\image.jpg --pretty
```

Optional calibration:
- Fixed scale:
```bash
python -m src.infer --weights models/weights/best.pt --image img.jpg --fixed-mm-per-px 0.05 --pretty
```
- Marker-based scale (known marker side in mm):
```bash
python -m src.infer --weights models/weights/best.pt --image img.jpg --marker-size-mm 20 --pretty
```

JSON output per defect:
- `id`
- `class`
- `confidence`
- `bbox` (`[x1, y1, x2, y2]`)
- `mask_area_px`
- `area_mm2` (nullable)
- `calibration` (`method`, `mm_per_px`) or `null`

## API

Run:

```bash
uvicorn api.main:app --reload
```

Endpoints:
- `GET /health`
- `POST /infer` (multipart upload, optional `fixed_mm_per_px`, `marker_size_mm`)

`/health` returns `degraded` when local weights are configured but missing, so startup no longer fails silently.

## Demo

```bash
streamlit run demo/app.py
```

## Docker

Build and run:

```bash
docker build -t defect-detector .
docker run --rm -p 8000:8000 defect-detector
```

## POC Evaluation Checklist

- Track recall on defects.
- Track false positives per image.
- Verify area ordering sanity (bigger defects should map to bigger areas).
- Spot-check 30-50 unseen images.
- Keep an error gallery:
  - `data/error_gallery/missed`
  - `data/error_gallery/false_positive`
  - `data/error_gallery/bad_area`
