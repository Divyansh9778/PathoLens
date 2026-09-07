"""
Optional Streamlit demo: upload a patch or large image, view prediction
and heatmap. Run with: streamlit run app/demo_app.py
"""

import streamlit as st
import numpy as np
import cv2
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import build_model
from tiling_inference import tile_and_predict, overlay_and_save

st.title("Histopathology Tumor Detection")
st.write("Upload a histopathology image to run tumor detection with a spatial heatmap.")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded image", use_column_width=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load("outputs/model.pth", map_location=device))

    with st.spinner("Running tiled inference..."):
        heatmap = tile_and_predict(model, img_rgb, device)

    st.write(f"Mean tumor probability across image: {heatmap.mean():.3f}")
    st.image(heatmap, caption="Tumor probability heatmap", use_column_width=True, clamp=True)
