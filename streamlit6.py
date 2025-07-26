import streamlit as st
import cv2
import torch
import numpy as np
import os
import supervision as sv
import sys

# Append the base YOLOv5 path
YOLOV5_PATH = r"C:\Users\dharm\OneDrive\Desktop\TEAM L3D\yolov5"
sys.path.append(YOLOV5_PATH)

# ✅ Import YOLOv5 modules directly from local repo
from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.torch_utils import select_device

# Set up Streamlit page
st.set_page_config(page_title="Smart Transit Optimization Platform", layout="wide")
st.title("🚦 SMART TRANSIT OPTIMIZATION PLATFORM")

# Sidebar configuration
st.sidebar.header("🔧 Configure Settings")
user_position = st.sidebar.number_input("Start Position (m)", value=0)
congestion_position = st.sidebar.number_input("Congestion Start Position (m)", value=500)
end_position = st.sidebar.number_input("End Position (m)", value=3000)
chosen_speed_limit = st.sidebar.slider("Set Speed Limit (km/h)", 20, 100, 60)

# 🎥 Sample video selection
st.subheader("🎥 Select a Sample Traffic Video")
video_options = {
    "sample1": "sample1.mp4",
    "sample2": "sample2.mp4",
    "sample3": "sample3.mp4"
}
selected_video_label = st.radio("Choose a video to analyze: ", list(video_options.keys()))
VIDEO_PATH = video_options[selected_video_label]

# Display the selected video
if os.path.exists(VIDEO_PATH):
    st.video(VIDEO_PATH)
else:
    st.warning(f"⚠ File not found: {VIDEO_PATH}")

# Load your custom YOLOv5 model
CUSTOM_MODEL_PATH = r"C:\Users\dharm\OneDrive\Desktop\TEAM L3D\yolov5\yolov5\runs\train\vehicle_model_v1\weights\best.pt"
st.info("Loading detection model...")

# ✅ Load YOLOv5 custom model directly
device = select_device("")
model = DetectMultiBackend(CUSTOM_MODEL_PATH, device=device)
imgsz = check_img_size((640, 640), s=model.stride)

# Init tracker
tracker = sv.ByteTrack()

# Define vehicle impact weights (by class ID)
vehicle_weights = {
    0: 0.5,  # bike
    1: 1.0,  # car
    2: 1.2,  # van
    3: 2.0   # truck
}

# Function to extract vehicle detection info from video
def extract_vehicle_data(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    data = []

    for _ in range(min(frame_count, 200)):
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.resize(frame, imgsz)
        img = torch.from_numpy(img).to(device).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)

        pred = model(img, augment=False, visualize=False)
        detections = sv.Detections.from_yolov5(pred)
        ids = detections.class_id if hasattr(detections, 'class_id') else np.array([])

        impact = sum(vehicle_weights.get(int(vid), 0) for vid in ids)
        data.append({"vehicle_impact": impact})
        tracker.update_with_detections(detections)

    cap.release()
    return data, fps

# Logic to classify congestion
def classify_congestion(impact):
    if impact > 25:
        return "Heavy"
    elif impact > 12:
        return "Moderate"
    else:
        return "Light"

# Estimate clearance time (in minutes)
def estimate_clearance_time(impact):
    return round(1 + max(0, impact - 10) * 0.25, 1)

# Run button
if st.button("🚀 Run Congestion Analysis"):
    if not os.path.exists(VIDEO_PATH):
        st.error(f"❌ Video not found: {VIDEO_PATH}")
    else:
        st.info("Analyzing video... This may take a moment.")
        vehicle_data, fps = extract_vehicle_data(VIDEO_PATH)

        avg_impact = np.mean([x['vehicle_impact'] for x in vehicle_data])
        congestion = classify_congestion(avg_impact)
        clearance_time = estimate_clearance_time(avg_impact)

        distance_to_congestion = congestion_position - user_position
        time_to_congestion = distance_to_congestion / (chosen_speed_limit / 3.6) / 60  # in minutes

        MIN_SPEED = 20
        if time_to_congestion < clearance_time:
            time_needed = clearance_time
            recommended_speed = distance_to_congestion / (time_needed * 60) * 3.6
            recommended_speed = max(recommended_speed, MIN_SPEED)
            delay = clearance_time - time_to_congestion
        else:
            recommended_speed = chosen_speed_limit
            delay = 0

        # Show results
        st.success("✅ Analysis Complete!")
        st.markdown(f"""
        ### 📜 Congestion Summary
        - *Congestion Level:* {congestion}
        - *Avg Vehicle Impact:* {avg_impact:.2f}
        - *Estimated Clearance Time:* {clearance_time} minutes
        - *Recommended Speed:* {recommended_speed:.1f} km/h
        - *Expected Delay if Speed Unchanged:* {delay:.2f} minutes

        ---
        *Positions:*
        - Start: {user_position} m
        - Congestion Starts At: {congestion_position} m
        - End: {end_position} m
        """)