import streamlit as st
import cv2
import torch
import supervision as sv
import numpy as np
from pathlib import Path
import tempfile

# Constants
VIDEO_PATH = "traffic_video.mp4"  # Replace with your video path
MODEL_PATH = "yolov5s.pt"  # Replace with your custom model path
YOLOV5_PATH = "C:/Users/dharm/OneDrive/Desktop/TEAM L3D/yolov5"

# Load YOLOv5 model from local repo
model = torch.hub.load(YOLOV5_PATH, "custom", path=MODEL_PATH, source="local")

# Congestion thresholds
THRESHOLDS = {
    'Low': 10,
    'Moderate': 20,
    'High': 30
}

@st.cache_data

def extract_vehicle_data(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    vehicle_counts = []
    class_ids = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Inference
        results = model(frame)
        st.write("[DEBUG] Type of prediction result:", type(results))

        if not hasattr(results, 'pred'):
            st.warning(f"[DEBUG] Skipping frame {frame_count} due to invalid prediction object: {type(results)}")
            continue

        pred_tensor = results.pred[0]

        if pred_tensor is None or len(pred_tensor) == 0 or pred_tensor.ndim != 2 or pred_tensor.shape[1] != 6:
            st.warning(f"[DEBUG] Invalid prediction shape at frame {frame_count}: {pred_tensor.shape}")
            continue

        try:
            detections = sv.Detections.from_yolov5(results)
        except Exception as e:
            st.warning(f"[DEBUG] Supervision conversion failed at frame {frame_count}: {e}")
            continue

        count = len(detections)
        vehicle_counts.append(count)

        for det in detections:
            class_ids.append(det.class_id)

    cap.release()
    return vehicle_counts, class_ids, fps

# --- Example Analysis Code ---

st.title("🚦 Congestion-Aware Traffic Advisory")
uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Analyzing video..."):
        vehicle_counts, class_ids, fps = extract_vehicle_data(tmp_path)

    if not vehicle_counts:
        st.error("No vehicles detected in the video.")
    else:
        avg_count = np.mean(vehicle_counts)
        unique_classes = set(class_ids)
        st.success("Analysis Complete!")
        st.write(f"**Average Vehicles per Frame:** {avg_count:.2f}")
        st.write(f"**Unique Vehicle Classes Detected:** {unique_classes}")
