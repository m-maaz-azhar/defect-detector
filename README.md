# Defect Detector MVP (Setup)

Simple pothole/defect detection app using YOLO segmentation.

This guide is focused on one goal: get the app running fast.

## 1. Prerequisites

- Windows PowerShell
- Python 3.11 recommended
- NVIDIA GPU optional (works on CPU too, slower)

## 2. Install

From repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Prepare dataset (one time)

Use included XML + images and convert to YOLO-seg format:

```powershell
python -m src.prepare_dataset --annotations-dir data/annotations --images-dir data/images --out-images-dir data/labeled/images --out-labels-dir data/labeled/labels --splits-dir data/splits
```

## 4. Train model (skip if you already have weights)

```powershell
python -m src.train --config configs/train.yaml
```

Expected output files:

- `models/weights/best.pt`
- `models/weights/model.onnx` (if export is enabled)

## 5. Start the UI app

```powershell
python -m streamlit run demo/app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

In the app sidebar:

- Set `Weights` to `models/weights/best.pt`
- Select `Device` (`0` for GPU, `cpu` for CPU)
- Upload image and run detection

## 6. Optional CLI check

```powershell
python -m src.infer --weights models/weights/best.pt --image data/images/potholes600.png --conf 0.40 --pretty
```

## 7. Optional API mode

```powershell
uvicorn api.main:app --reload
```

Endpoints:

- `GET /health`
- `POST /infer`

## Troubleshooting

- `ModuleNotFoundError: No module named 'src'`
  Run Streamlit with:
  `python -m streamlit run demo/app.py`

- No GPU detected:
  Verify with:
  `python -c "import torch; print(torch.cuda.is_available())"`

- Missing weights:
  Train first or point app to an existing `.pt` file.

## Project layout (minimal)

```text
configs/         # train + dataset config
data/            # raw, labeled, splits
demo/app.py      # streamlit UI
api/main.py      # fastapi server
src/train.py     # training
src/infer.py     # inference
models/weights/  # best.pt, model.onnx
```
