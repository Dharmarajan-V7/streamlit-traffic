import streamlit as st
import torch
import supervision as sv
import cv2
import numpy as np
import tempfile
import os

# Title and layout
st.set_page_config(page_title="Smart Transit Optimization Platform", layout="wide")
st.title("🚦 SMART TRANSIT OPTIMIZATION PLATFORM")

# Sidebar - user inputs
st.sidebar.header("🔧 Configure Settings")
user_position = st.sidebar.number_input("Start Position (m)", value=0)
congestion_position = st.sidebar.number_input("Congestion Start Position (m)", value=500)
end_position = st.sidebar.number_input("End Position (m)", value=3000)
chosen_speed_limit = st.sidebar.slider("Set Speed Limit (km/h)", 20, 100, 60)

# Sample video selection
st.subheader("🎥 Select a Sample Traffic Video")
video_options = {
    "Urban Morning Traffic": "sample1",
    "Evening Highway Congestion": "sample2",
    "Weekend Flow": "sample3"
}
selected_video_label = st.radio("Choose a video to analyze: ", list(video_options.keys()))
VIDEO_PATH = video_options[selected_video_label]

# Show video
st.video(VIDEO_PATH)

# Model loading
CUSTOM_MODEL_PATH = r"C:\Users\dharm\OneDrive\Desktop\TEAM L3D\yolov5\yolov5\runs\train\vehicle_model_v1\weights\best.pt"

st.info("Loading model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=CUSTOM_MODEL_PATH, force_reload=True)
tracker = sv.ByteTrack()

# Vehicle weights
vehicle_weights = {0: 0.5, 1: 1.0, 2: 1.2, 3: 2.0}  # bike, car, van, truck

# Extract vehicle data
def extract_vehicle_data(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    data = []

    for _ in range(min(frame_count, 200)):
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        detections = sv.Detections.from_yolov5(results)
        ids = detections.class_id if hasattr(detections, 'class_id') else np.array([])

        impact = sum(vehicle_weights.get(int(vid), 0) for vid in ids)
        data.append({"vehicle_impact": impact})
        tracker.update_with_detections(detections)

    cap.release()
    return data, fps

# Congestion logic
def classify_congestion(impact):
    if impact > 25:
        return "Heavy"
    elif impact > 12:
        return "Moderate"
    else:
        return "Light"

def estimate_clearance_time(impact):
    return round(1 + max(0, impact - 10) * 0.25, 1)

# Run processing
if st.button("🚀 Run Congestion Analysis"):
    st.info("Analyzing video... This may take a moment.")
    vehicle_data, fps = extract_vehicle_data(VIDEO_PATH)

    avg_impact = np.mean([x['vehicle_impact'] for x in vehicle_data])
    congestion = classify_congestion(avg_impact)
    clearance_time = estimate_clearance_time(avg_impact)

    distance_to_congestion = congestion_position - user_position
    time_to_congestion = distance_to_congestion / (chosen_speed_limit / 3.6) / 60

    MIN_SPEED = 20
    if time_to_congestion < clearance_time:
        time_needed = clearance_time
        recommended_speed = distance_to_congestion / (time_needed * 60) * 3.6
        recommended_speed = max(recommended_speed, MIN_SPEED)
        delay = clearance_time - time_to_congestion
    else:
        recommended_speed = chosen_speed_limit
        delay = 0

    # Display results
    st.success("✅ Analysis Complete!")
    st.markdown("""
    ### 📝 Congestion Summary
    - **Congestion Level:** {0}
    - **Avg Vehicle Impact:** {1:.2f}
    - **Estimated Clearance Time:** {2} minutes
    - **Recommended Speed:** {3:.1f} km/h
    - **Expected Delay if Speed Unchanged:** {4:.2f} minutes

    ---
    **Positions:**
    - Start: {5} m
    - Congestion Starts At: {6} m
    - End: {7} m
    """.format(congestion, avg_impact, clearance_time, recommended_speed, delay, user_position, congestion_position, end_position))
