from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Ensure project root is importable when Streamlit runs from demo/ context.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer import DefectInferencer

try:
    import torch
except Exception:
    torch = None


PAGE_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root {
  --bg: #f5f1e8;
  --surface: #fff8eb;
  --card: #fffdf7;
  --ink: #0f172a;
  --muted: #4b5563;
  --line: #d7c4a5;
  --accent: #d97706;
  --accent-strong: #b45309;
  --ok: #0f766e;
}

.stApp {
  background:
    radial-gradient(circle at 10% 10%, #ffe8bf 0%, rgba(255, 232, 191, 0) 45%),
    radial-gradient(circle at 90% 0%, #d9f7ef 0%, rgba(217, 247, 239, 0) 40%),
    linear-gradient(180deg, #fdf8ef 0%, #f5efe5 100%);
  color: var(--ink);
  font-family: "Manrope", sans-serif;
}

.block-container {
  max-width: 1180px;
  padding-top: 1.1rem;
  padding-bottom: 2rem;
}

.hero {
  border: 1px solid var(--line);
  background: linear-gradient(120deg, #fff7e6, #fffdf7 52%, #ecfdf5);
  border-radius: 18px;
  padding: 1rem 1.2rem;
  box-shadow: 0 10px 32px rgba(15, 23, 42, 0.08);
}

.hero h1 {
  margin: 0;
  font-size: 1.55rem;
  letter-spacing: -0.02em;
}

.hero p {
  margin: 0.35rem 0 0 0;
  color: var(--muted);
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
  margin-top: 0.85rem;
}

.kpi {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--card);
  padding: 0.75rem 0.85rem;
}

.kpi .label {
  color: var(--muted);
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 700;
}

.kpi .value {
  margin-top: 0.2rem;
  font-size: 1.35rem;
  font-weight: 800;
}

.chip-row {
  margin-top: 0.75rem;
}

.chip {
  display: inline-block;
  border: 1px solid var(--line);
  color: var(--ink);
  background: var(--card);
  padding: 0.25rem 0.5rem;
  border-radius: 999px;
  font-size: 0.76rem;
  margin-right: 0.35rem;
  margin-bottom: 0.35rem;
}

.json-note {
  color: var(--muted);
  font-size: 0.9rem;
}

@media (max-width: 900px) {
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 520px) {
  .kpi-grid { grid-template-columns: 1fr; }
  .hero h1 { font-size: 1.25rem; }
}
</style>
"""


@st.cache_resource
def load_inferencer(weights_path: str, device: str) -> DefectInferencer:
    return DefectInferencer(weights_path=weights_path, device=device)


def list_devices() -> list[str]:
    options = ["cpu"]
    if torch is not None and torch.cuda.is_available():
        options = ["0", "cpu"]
    return options


def confidence_color(confidence: float) -> tuple[int, int, int]:
    if confidence >= 0.8:
        return (25, 170, 95)  # Green
    if confidence >= 0.6:
        return (16, 185, 129)  # Teal
    if confidence >= 0.4:
        return (8, 145, 178)  # Cyan
    return (37, 99, 235)  # Blue


def draw_detections(image_bgr: np.ndarray, defects: list[dict]) -> np.ndarray:
    for idx, defect in enumerate(defects, start=1):
        x1, y1, x2, y2 = defect["bbox"]
        conf = float(defect["confidence"])
        color = confidence_color(conf)

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        label = f'#{idx} {defect["class"]} {conf:.2f}'
        area_px = defect.get("mask_area_px")
        if area_px is not None:
            label += f" | {area_px}px"

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_top = max(0, y1 - text_h - 12)
        label_bottom = max(text_h + baseline + 8, y1 - 2)
        cv2.rectangle(
            image_bgr,
            (x1, label_top),
            (min(image_bgr.shape[1] - 1, x1 + text_w + 10), label_bottom),
            color,
            thickness=-1,
        )
        cv2.putText(
            image_bgr,
            label,
            (x1 + 5, label_bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image_bgr


def summarize(defects: list[dict]) -> dict[str, str]:
    total = len(defects)
    avg_conf = sum(float(item["confidence"]) for item in defects) / total if total else 0.0
    total_px = sum(int(item["mask_area_px"]) for item in defects)
    area_values = [item.get("area_mm2") for item in defects if item.get("area_mm2") is not None]
    severity_counts = Counter([item.get("severity") for item in defects if item.get("severity")])

    summary = {
        "count": str(total),
        "avg_conf": f"{avg_conf:.2f}",
        "total_px": f"{total_px:,}",
        "total_mm2": f"{sum(area_values):.2f}" if area_values else "N/A",
        "severity": ", ".join([f"{key}:{value}" for key, value in severity_counts.items()]) if severity_counts else "N/A",
    }
    return summary


def to_table_rows(defects: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for idx, defect in enumerate(defects, start=1):
        bbox = defect["bbox"]
        rows.append(
            {
                "idx": idx,
                "class": defect["class"],
                "confidence": round(float(defect["confidence"]), 4),
                "bbox": f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]",
                "mask_area_px": int(defect["mask_area_px"]),
                "area_mm2": defect.get("area_mm2"),
                "severity": defect.get("severity"),
            }
        )
    return rows


def render_summary(model_name: str, summary: dict[str, str], chips: list[str]) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>Pothole Detection MVP</h1>
          <p>Model: <b>{model_name}</b> | Operational preview for image upload, scoring, and export.</p>
          <div class="kpi-grid">
            <div class="kpi"><div class="label">Defects</div><div class="value">{summary["count"]}</div></div>
            <div class="kpi"><div class="label">Avg Confidence</div><div class="value">{summary["avg_conf"]}</div></div>
            <div class="kpi"><div class="label">Mask Area (px)</div><div class="value">{summary["total_px"]}</div></div>
            <div class="kpi"><div class="label">Area (mm²)</div><div class="value">{summary["total_mm2"]}</div></div>
          </div>
          <div class="chip-row">
            {"".join([f'<span class="chip">{chip}</span>' for chip in chips])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def calibration_inputs() -> tuple[str, float | None, float | None]:
    mode = st.radio(
        "Calibration",
        options=["none", "fixed-scale", "marker"],
        horizontal=True,
        help="Use fixed mm/px or a known marker size to estimate mm².",
    )
    fixed_mm_per_px: float | None = None
    marker_size_mm: float | None = None
    if mode == "fixed-scale":
        value = st.number_input("Fixed mm/px", min_value=0.001, max_value=20.0, value=0.05, step=0.001)
        fixed_mm_per_px = float(value)
    if mode == "marker":
        value = st.number_input("Marker side (mm)", min_value=1.0, max_value=500.0, value=20.0, step=1.0)
        marker_size_mm = float(value)
    return mode, fixed_mm_per_px, marker_size_mm


def main() -> None:
    st.set_page_config(page_title="Pothole Detector MVP", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    st.caption("Upload image -> detect potholes -> inspect detections -> export JSON and annotated image.")

    with st.sidebar:
        st.subheader("Inference Controls")
        weights = st.text_input("Weights", value="models/weights/best.pt")
        device_options = list_devices()
        default_idx = 0 if "0" in device_options else device_options.index("cpu")
        device = st.selectbox("Device", device_options, index=default_idx)
        conf = st.slider("Confidence", min_value=0.05, max_value=0.95, value=0.40, step=0.01)
        iou = st.slider("IoU", min_value=0.10, max_value=0.95, value=0.50, step=0.01)
        include_severity = st.checkbox("Severity buckets", value=True)
        mode, fixed_mm_per_px, marker_size_mm = calibration_inputs()
        st.divider()
        st.markdown("`Note:` GTX 1660 Super runs with AMP disabled by Ultralytics for stability.")

    uploaded = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("Drop one image to run pothole detection.")
        return

    inferencer = load_inferencer(weights, device)
    inferencer.conf = conf
    inferencer.iou = iou

    image_bytes = uploaded.getvalue()
    if not image_bytes:
        st.error("Uploaded file is empty.")
        return

    try:
        with st.spinner("Running detection..."):
            response = inferencer.infer_bytes(
                image_bytes=image_bytes,
                fixed_mm_per_px=fixed_mm_per_px,
                marker_size_mm=marker_size_mm,
                include_severity=include_severity,
            )
    except Exception as exc:
        st.error(str(exc))
        return

    payload = response.model_dump(by_alias=True)
    defects = payload["defects"]
    summary = summarize(defects)
    chips = [
        f"conf={conf:.2f}",
        f"iou={iou:.2f}",
        f"calibration={mode}",
        f"severity={include_severity}",
    ]
    if summary["severity"] != "N/A":
        chips.append(f"severity-mix {summary['severity']}")
    render_summary(payload["model"], summary, chips)

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not decode the uploaded image.")
        return

    annotated = draw_detections(image_bgr.copy(), payload["defects"])
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    view_col, data_col = st.columns([1.45, 1])
    with view_col:
        tab1, tab2 = st.tabs(["Annotated", "Original"])
        with tab1:
            st.image(annotated_rgb, use_container_width=True)
        with tab2:
            st.image(original_rgb, use_container_width=True)

    with data_col:
        st.subheader("Detections")
        if defects:
            st.dataframe(to_table_rows(defects), hide_index=True, use_container_width=True)
        else:
            st.warning("No potholes detected at this threshold.")
        if mode == "marker" and all(item.get("area_mm2") is None for item in defects):
            st.info("Marker calibration requested, but no marker was detected in this image.")
        st.markdown('<p class="json-note">Export results for API/regression checks.</p>', unsafe_allow_html=True)
        json_payload = json.dumps(payload, indent=2)
        st.download_button(
            "Download JSON",
            data=json_payload,
            file_name=f"{Path(uploaded.name).stem}_detections.json",
            mime="application/json",
            use_container_width=True,
        )
        ok, png = cv2.imencode(".png", annotated)
        if ok:
            st.download_button(
                "Download Annotated PNG",
                data=png.tobytes(),
                file_name=f"{Path(uploaded.name).stem}_annotated.png",
                mime="image/png",
                use_container_width=True,
            )

    with st.expander("Raw Response JSON"):
        st.json(payload)


if __name__ == "__main__":
    main()
