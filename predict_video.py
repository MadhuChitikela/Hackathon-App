import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import torch
if not hasattr(torch.classes, "__path__"):
    torch.classes.__path__ = type('', (), {"_path": []})()

from ultralytics import YOLO
import streamlit as st
import tempfile
import shutil
import glob
import cv2

# Load YOLOv8 model
model = YOLO("runs/detect/train2/weights/best.pt")

# Streamlit UI
st.set_page_config(page_title="YOLOv8 Video Detection", layout="centered")
st.title("YOLOv8 Safety Detection")

# Upload video
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
if uploaded_video:
    # Save uploaded video to a temp file
    temp_input_path = os.path.join(tempfile.gettempdir(), uploaded_video.name)
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(temp_input_path)
    st.write("Running detection...")

    # Use fixed output directory
    output_dir = "runs/detect/streamlit"
    if os.path.exists(output_dir):
        try:
            files = glob.glob(os.path.join(output_dir, "*"))
            for f in files:
                os.remove(f)
        except Exception as e:
            st.warning(f"Could not clear output folder: {e}")

    # Run detection
    with st.spinner("Processing..."):
        results = model.predict(
            source=temp_input_path,
            save=True,
            save_txt=False,
            save_conf=False,
            stream=False,
            conf=0.25,
            project="runs/detect",
            name="streamlit",
            exist_ok=True,
            vid_stride=1
        )

    # Find annotated video
    annotated_video = None
    for file in os.listdir(output_dir):
        if file.endswith((".mp4", ".avi", ".mov")):
            annotated_video = os.path.join(output_dir, file)
            break

    if annotated_video and os.path.exists(annotated_video):
        # Re-encode video to ensure compatibility
        cap = cv2.VideoCapture(annotated_video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output_path = os.path.join(tempfile.gettempdir(), "reencoded_output.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

        st.success("Detection complete. Annotated video:")
        st.video(temp_output_path)
    else:
        st.error("Annotated video not found or could not be generated.")