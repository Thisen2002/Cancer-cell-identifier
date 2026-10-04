"""
Streamlit prototype for the Canine TVT vs SCC cytology classifier.

Run from the repository root with:
    streamlit run app/app.py

This interface is for research/prototype use. It must not be presented as a validated
clinical diagnostic tool unless future evidence supports that claim.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# Ensure src/ is importable when Streamlit executes this file from app/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from explainability import generate_gradcam
from inference import TVTSCCPredictor


st.set_page_config(page_title="TVT vs SCC AI", page_icon="🔬", layout="centered")

st.title("Canine TVT vs SCC Cytology Classifier")
st.caption("Research prototype — not a substitute for veterinary pathological diagnosis.")

checkpoint_text = st.sidebar.text_input(
    "Model checkpoint",
    value=str(PROJECT_ROOT / "models" / "baseline" / "best_model.pt"),
)

uploaded_file = st.file_uploader(
    "Upload a cytology image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

if uploaded_file is None:
    st.info("Upload an image after a trained model checkpoint is available.")
    st.stop()

checkpoint_path = Path(checkpoint_text)
if not checkpoint_path.exists():
    st.error(
        "The model checkpoint does not exist yet. Train a model first, then select "
        "the generated best_model.pt file."
    )
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded cytology image", use_container_width=True)

try:
    predictor = TVTSCCPredictor(checkpoint_path)
    result = predictor.predict_pil(image)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

st.subheader("Model output")
st.metric("Prediction", result["predicted_label"])
st.metric("Model confidence", f"{result['confidence']:.1%}")

st.write("Class probabilities")
st.progress(result["probabilities"]["TVT"], text=f"TVT: {result['probabilities']['TVT']:.1%}")
st.progress(result["probabilities"]["SCC"], text=f"SCC: {result['probabilities']['SCC']:.1%}")

st.subheader("Grad-CAM")
try:
    # The Grad-CAM helper currently accepts file paths. A temporary copy is used,
    # and the original uploaded image is never modified.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        input_path = tmp_dir / "uploaded.png"
        output_path = tmp_dir / "gradcam.png"
        image.save(input_path)

        generate_gradcam(
            checkpoint_path=checkpoint_path,
            image_path=input_path,
            output_path=output_path,
        )
        cam_image = Image.open(output_path).copy()

    st.image(
        cam_image,
        caption="Grad-CAM visualization of regions influencing the prediction",
        use_container_width=True,
    )
except Exception as exc:
    st.warning(f"Grad-CAM could not be generated: {exc}")

st.warning(
    "Research use only. Model probability is not equivalent to diagnostic certainty. "
    "Final interpretation should remain with a qualified veterinary professional."
)
