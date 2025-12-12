# app.py
"""
Streamlit app for Space Station Safety Objects Detection
Full-featured UI with image and video detection capabilities
"""

import io
import os
from pathlib import Path
import tempfile
from typing import Optional
from datetime import datetime
import json
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# ------------ CONFIG ------------
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Try multiple possible paths for the model
POSSIBLE_MODEL_PATHS = [
    SCRIPT_DIR / "runs" / "detect" / "train5" / "weights" / "best.pt",  # Relative to script
    Path("scripts/Hackathon2_scripts/runs/detect/train5/weights/best.pt"),  # From workspace root
    Path("runs/detect/train5/weights/best.pt"),  # From script directory
]

# Find the first existing model path
MODEL_PATH = None
for path in POSSIBLE_MODEL_PATHS:
    if path.exists():
        MODEL_PATH = str(path)
        break

# If no model found, set to first option (will show error later)
if MODEL_PATH is None:
    MODEL_PATH = str(POSSIBLE_MODEL_PATHS[0])

SAVED_DIR = SCRIPT_DIR / "saved_detections"
SAVED_DIR.mkdir(exist_ok=True, parents=True)
HISTORY_FILE = SAVED_DIR / "history.json"

CONFIDENCE_THRESHOLD = 0.25
IMG_SIZE = 1280
SHOW_CONFIDENCE = True

# Space station object classes
CLASSES = {
    0: "OxygenTank",
    1: "NitrogenTank",
    2: "FirstAidBox",
    3: "FireAlarm",
    4: "SafetySwitchPanel",
    5: "EmergencyPhone",
    6: "FireExtinguisher"
}
# ---------------------------------

st.set_page_config(
    page_title="Space Station Safety Objects Detection",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- CSS ----------
PAGE_CSS = """
<style>
/* Main background with galaxy effect */
.stApp {
    background: 
        radial-gradient(ellipse at top, rgba(138, 43, 226, 0.2) 0%, transparent 50%),
        radial-gradient(ellipse at bottom right, rgba(0, 191, 255, 0.15) 0%, transparent 50%),
        radial-gradient(ellipse at bottom left, rgba(255, 20, 147, 0.15) 0%, transparent 50%),
        linear-gradient(180deg, #0a0e1a 0%, #050608 100%);
    background-attachment: fixed;
}

/* Remove Streamlit default elements */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Container styling */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Top navigation */
.top-nav {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    align-items: center;
    margin-bottom: 15px;
    padding: 5px 0;
}

/* Glowing blue buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.5), 
                0 0 40px rgba(0, 212, 255, 0.3),
                0 4px 15px rgba(0, 212, 255, 0.4) !important;
    position: relative !important;
    overflow: hidden !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.8), 
                0 0 60px rgba(0, 212, 255, 0.5),
                0 6px 20px rgba(0, 212, 255, 0.6) !important;
    background: linear-gradient(135deg, #00e5ff 0%, #00b3e6 100%) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.6), 
                0 0 30px rgba(0, 212, 255, 0.4),
                0 2px 10px rgba(0, 212, 255, 0.5) !important;
}

/* Primary button styling */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
    border-radius: 30px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 14px 32px !important;
    box-shadow: 0 0 25px rgba(0, 212, 255, 0.6), 
                0 0 50px rgba(0, 212, 255, 0.4),
                0 6px 20px rgba(0, 212, 255, 0.5) !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 35px rgba(0, 212, 255, 0.9), 
                0 0 70px rgba(0, 212, 255, 0.6),
                0 8px 25px rgba(0, 212, 255, 0.7) !important;
    background: linear-gradient(135deg, #00e5ff 0%, #00b3e6 100%) !important;
}

/* Card styling */
.main-card {
    border-radius: 20px;
    padding: 30px;
    background: linear-gradient(180deg, rgba(15, 25, 40, 0.6) 0%, rgba(10, 15, 25, 0.4) 100%);
    border: 1px solid rgba(0, 212, 255, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px);
    margin-bottom: 10px;
}

/* Title styling */
.main-title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #ffffff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 10px;
    line-height: 1.2;
}

.subtitle-text {
    color: rgba(200, 230, 255, 0.8);
    text-align: center;
    font-size: 16px;
    padding: 10px 0 15px 0;
    line-height: 1.6;
}

/* Upload area */
.upload-area {
    border: 2px dashed rgba(0, 212, 255, 0.3);
    border-radius: 15px;
    padding: 30px 20px;
    text-align: center;
    background: rgba(0, 212, 255, 0.05);
    transition: all 0.3s ease;
    margin: 15px 0;
}

.upload-area:hover {
    border-color: rgba(0, 212, 255, 0.6);
    background: rgba(0, 212, 255, 0.1);
}

/* Buttons */
.detect-btn {
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
    color: white;
    padding: 14px 32px;
    border-radius: 30px;
    font-weight: 700;
    font-size: 16px;
    border: none;
    width: 100%;
    margin-top: 20px;
    box-shadow: 0 6px 20px rgba(0, 212, 255, 0.3);
    transition: all 0.3s ease;
}

.detect-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.5);
}

/* Results area */
.results-card {
    border-radius: 15px;
    padding: 20px;
    background: rgba(15, 25, 40, 0.4);
    border: 1px solid rgba(0, 212, 255, 0.2);
    margin-top: 10px;
}

.stats-box {
    display: flex;
    gap: 20px;
    margin: 20px 0;
    flex-wrap: wrap;
}

.stat-item {
    flex: 1;
    min-width: 150px;
    padding: 15px;
    background: rgba(0, 212, 255, 0.1);
    border-radius: 10px;
    border: 1px solid rgba(0, 212, 255, 0.2);
    text-align: center;
}

.stat-value {
    font-size: 28px;
    font-weight: 700;
    color: #00d4ff;
}

.stat-label {
    font-size: 12px;
    color: rgba(200, 230, 255, 0.7);
    margin-top: 5px;
}

/* Detection boxes */
.detection-list {
    margin-top: 20px;
}

.detection-item {
    padding: 12px;
    margin: 8px 0;
    background: rgba(0, 212, 255, 0.05);
    border-left: 3px solid #00d4ff;
    border-radius: 5px;
}

/* Footer */
.footer-note {
    position: fixed;
    bottom: 20px;
    left: 20px;
    color: rgba(255, 255, 255, 0.2);
    font-size: 11px;
}

/* Model status */
.model-status {
    padding: 15px;
    border-radius: 10px;
    margin: 20px 0;
}

.model-status.success {
    background: rgba(0, 255, 136, 0.1);
    border: 1px solid rgba(0, 255, 136, 0.3);
    color: #00ff88;
}

.model-status.warning {
    background: rgba(255, 193, 7, 0.1);
    border: 1px solid rgba(255, 193, 7, 0.3);
    color: #ffc107;
}

.model-status.error {
    background: rgba(255, 82, 82, 0.1);
    border: 1px solid rgba(255, 82, 82, 0.3);
    color: #ff5252;
}
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ---------- Helper Functions ----------
@st.cache_resource(ttl=3600)
def try_load_ultralytics_model(path: Optional[str] = None):
    """Try to load a YOLO model from the given path."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        return None, f"Ultralytics not installed: {e}"

    if path:
        p = Path(path)
        if not p.exists():
            return None, f"Model not found at: {path}"
        try:
            model = YOLO(str(p))
            return model, None
        except Exception as e:
            return None, f"Failed to load model at {path}: {e}"
    else:
        return None, None

def load_pretrained_yolov8n():
    """Load pretrained yolov8n model."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        return model, None
    except Exception as e:
        return None, f"Failed to load pretrained yolov8n: {e}"

def draw_detections_on_pil(image: Image.Image, detections: list):
    """Draw bounding boxes and labels on PIL image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except Exception:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=20)
        except Exception:
            font = ImageFont.load_default()
    
    w, h = image.size
    width_scale = max(2, int(round((w + h) / 800)))
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 212, 255), width=width_scale)
        
        # Prepare label
        if SHOW_CONFIDENCE:
            lbl = f"{det['label']} {det['conf']:.2f}"
        else:
            lbl = det['label']
        
        # Get text size (using textbbox for newer PIL versions)
        try:
            bbox = draw.textbbox((0, 0), lbl, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older PIL versions
            text_w, text_h = draw.textsize(lbl, font=font)
        
        pad = 8
        text_y = max(0, y1 - text_h - pad)
        text_x = x1
        
        # Draw label background
        draw.rectangle(
            [text_x, text_y, text_x + text_w + pad * 2, text_y + text_h + pad],
            fill=(0, 212, 255)
        )
        
        # Draw label text
        draw.text(
            (text_x + pad, text_y + pad // 2),
            lbl,
            fill=(0, 0, 0),
            font=font
        )
    
    return image

def run_inference_on_pil(model, pil_image: Image.Image, confidence_threshold: float = None, fast_mode: bool = False):
    """Run YOLO inference on PIL image and return detections."""
    if confidence_threshold is None:
        confidence_threshold = CONFIDENCE_THRESHOLD
    img_np = np.asarray(pil_image)
    
    # Use smaller image size for live detection to reduce lag
    inference_size = 640 if fast_mode else IMG_SIZE
    
    res = model.predict(
        source=img_np,
        imgsz=inference_size,
        conf=confidence_threshold,
        verbose=False
    )
    r = res[0]
    boxes = getattr(r, "boxes", None)
    detections = []
    
    if boxes is None or len(boxes) == 0:
        return detections
    
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        xyxy = np.array(boxes.xyxy)
        confs = np.array(boxes.conf)
        cls_ids = np.array(boxes.cls).astype(int)
    
    names = model.names if hasattr(model, "names") else CLASSES
    orig_w, orig_h = pil_image.size
    
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, cls_ids):
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(orig_w - 1, x2))
        y2 = int(min(orig_h - 1, y2))
        
        label = names.get(int(cls), str(cls)) if isinstance(names, dict) else names[int(cls)]
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "conf": float(conf),
            "label": str(label)
        })
    
    return detections

def save_detection_history(image_path: str, detections: list, file_name: str):
    """Save detection to history JSON file."""
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
    
    # Determine file type
    file_type = "video" if str(image_path).endswith(('.mp4', '.avi', '.mov', '.mkv')) else "image"
    
    history.append({
        "timestamp": datetime.now().isoformat(),
        "file_name": file_name,
        "file_path": str(image_path),
        "image_path": str(image_path),  # Keep for backward compatibility
        "file_type": file_type,
        "detections": detections,
        "count": len(detections)
    })
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def load_detection_history():
    """Load detection history from JSON file."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

# ---------- Session State ----------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "model_obj" not in st.session_state:
    st.session_state.model_obj = None
if "model_msg" not in st.session_state:
    st.session_state.model_msg = None
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = CONFIDENCE_THRESHOLD
if "live_detection" not in st.session_state:
    st.session_state.live_detection = False
if "live_source" not in st.session_state:
    st.session_state.live_source = None
if "live_rtsp" not in st.session_state:
    st.session_state.live_rtsp = None

# Try to pre-load model
if st.session_state.model_obj is None and st.session_state.model_msg is None:
    model_obj, msg = try_load_ultralytics_model(MODEL_PATH)
    st.session_state.model_obj = model_obj
    st.session_state.model_msg = msg

# ---------- Navigation ----------
st.markdown("""
<div style="text-align: center; margin: 0 0 20px 0;">
    <h2 style="background: linear-gradient(135deg, #00d4ff 0%, #00ff88 50%, #00d4ff 100%); 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent; 
               background-clip: text;
               font-size: 32px;
               font-weight: 800;
               margin: 0;
               text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);">
        🚀 Autonomous Space Station Safety Object Detection
    </h2>
</div>
""", unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    if st.button("🏠 Home", key="nav_home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
with col2:
    if st.button("🔴 Live Detection", key="nav_live", use_container_width=True):
        st.session_state.page = "live"
        st.rerun()
with col3:
    if st.button("📸 Image Upload", key="nav_image", use_container_width=True):
        st.session_state.page = "image"
        st.rerun()
with col4:
    if st.button("🎥 Video Upload", key="nav_video", use_container_width=True):
        st.session_state.page = "video"
        st.rerun()
with col5:
    if st.button("📊 History", key="nav_history", use_container_width=True):
        st.session_state.page = "history"
        st.rerun()
with col6:
    if st.button("ℹ️ About", key="nav_about", use_container_width=True):
        st.session_state.page = "about"
        st.rerun()

# ---------- Page Functions ----------
def page_home():
    """Home page with modern website-style landing page."""
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px 40px 20px; background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 0, 0, 0.3) 100%); border-radius: 20px; margin-bottom: 40px;">
        <h1 style="font-size: 56px; font-weight: 900; background: linear-gradient(135deg, #00d4ff 0%, #ffffff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 20px; line-height: 1.2;">
            🚀 AI Space Station<br/>Safety Detection
        </h1>
        <p style="font-size: 22px; color: rgba(200, 230, 255, 0.9); margin-bottom: 30px; line-height: 1.6;">
            Advanced AI-powered object detection for space station safety monitoring<br/>
            <span style="font-size: 18px; color: rgba(0, 212, 255, 0.8);">Real-time detection • High accuracy • Mission-critical safety</span>
        </p>
        <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; margin-top: 40px;">
            <a href="#features" style="text-decoration: none;">
                <button style="background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%); color: white; padding: 15px 35px; border-radius: 30px; font-weight: 700; font-size: 16px; border: none; cursor: pointer; box-shadow: 0 0 25px rgba(0, 212, 255, 0.6), 0 0 50px rgba(0, 212, 255, 0.4); transition: all 0.3s ease;">
                    Get Started →
                </button>
            </a>
            <a href="#about" style="text-decoration: none;">
                <button style="background: rgba(0, 212, 255, 0.1); color: #00d4ff; padding: 15px 35px; border-radius: 30px; font-weight: 700; font-size: 16px; border: 2px solid rgba(0, 212, 255, 0.5); cursor: pointer; transition: all 0.3s ease;">
                    Learn More
                </button>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown('<div id="features"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin: 50px 0;">
        <h2 style="text-align: center; font-size: 42px; font-weight: 800; color: #00d4ff; margin-bottom: 50px;">
            What We Detect
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Grid
    features = [
        ("🫁", "Oxygen Tank", "Life support systems"),
        ("🫁", "Nitrogen Tank", "Environmental control"),
        ("🏥", "First Aid Box", "Medical emergency supplies"),
        ("🔔", "Fire Alarm", "Fire detection systems"),
        ("🔌", "Safety Switch Panel", "Emergency controls"),
        ("📞", "Emergency Phone", "Communication systems"),
        ("🧯", "Fire Extinguisher", "Fire suppression equipment"),
    ]
    
    cols = st.columns(4)
    for idx, (icon, name, desc) in enumerate(features):
        with cols[idx % 4]:
            st.markdown(f"""
            <div style="background: linear-gradient(180deg, rgba(15, 25, 40, 0.6) 0%, rgba(10, 15, 25, 0.4) 100%); 
                        border: 1px solid rgba(0, 212, 255, 0.2); 
                        border-radius: 15px; 
                        padding: 25px; 
                        text-align: center; 
                        margin-bottom: 20px;
                        transition: all 0.3s ease;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);">
                <div style="font-size: 48px; margin-bottom: 15px;">{icon}</div>
                <h3 style="color: #00d4ff; font-size: 20px; font-weight: 700; margin-bottom: 10px;">{name}</h3>
                <p style="color: rgba(200, 230, 255, 0.7); font-size: 14px; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Technology Section
    st.markdown("""
    <div style="margin: 60px 0; padding: 40px; background: linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, rgba(0, 0, 0, 0.2) 100%); border-radius: 20px; border: 1px solid rgba(0, 212, 255, 0.2);">
        <h2 style="text-align: center; font-size: 42px; font-weight: 800; color: #00d4ff; margin-bottom: 30px;">
            Powered by YOLO AI
        </h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 30px; margin-top: 40px;">
            <div style="text-align: center;">
                <div style="font-size: 36px; color: #00d4ff; font-weight: 700;">⚡</div>
                <h3 style="color: #ffffff; margin: 15px 0 10px 0;">Real-Time Processing</h3>
                <p style="color: rgba(200, 230, 255, 0.8);">Instant detection results</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 36px; color: #00d4ff; font-weight: 700;">🎯</div>
                <h3 style="color: #ffffff; margin: 15px 0 10px 0;">High Accuracy</h3>
                <p style="color: rgba(200, 230, 255, 0.8);">State-of-the-art detection</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 36px; color: #00d4ff; font-weight: 700;">🔍</div>
                <h3 style="color: #ffffff; margin: 15px 0 10px 0;">Multi-Object Detection</h3>
                <p style="color: rgba(200, 230, 255, 0.8);">Detect multiple objects simultaneously</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Status Section
    st.markdown("""
    <div style="margin: 50px 0;">
        <h2 style="text-align: center; font-size: 36px; font-weight: 800; color: #00d4ff; margin-bottom: 30px;">
            System Status
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model_obj is None:
        st.markdown(
            '<div class="model-status warning" style="text-align: center; padding: 20px;">⚠️ Model not loaded</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <p style="color: rgba(200, 230, 255, 0.8); font-size: 16px; margin-bottom: 20px;">Load a model to start detecting objects</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📥 Load YOLOv8n (Pretrained)", key="load_yolo8n", use_container_width=True):
                with st.spinner("Loading pretrained model..."):
                    model, msg = load_pretrained_yolov8n()
                    st.session_state.model_obj = model
                    st.session_state.model_msg = msg
                    st.rerun()
        
        with col_b:
            uploaded_model = st.file_uploader(
                "Upload Model (.pt)",
                type=["pt"],
                key="model_upload",
                label_visibility="collapsed"
            )
            if uploaded_model is not None:
                tpath = SAVED_DIR / f"uploaded_model_{uploaded_model.name}"
                with open(tpath, "wb") as f:
                    f.write(uploaded_model.read())
                with st.spinner("Loading uploaded model..."):
                    model, msg = try_load_ultralytics_model(str(tpath))
                    st.session_state.model_obj = model
                    st.session_state.model_msg = msg
                    st.rerun()
        
        if st.session_state.model_msg:
            st.error(f"Error: {st.session_state.model_msg}")
    else:
        st.markdown(
            '<div class="model-status success" style="text-align: center; padding: 20px;">✅ Model loaded and ready</div>',
            unsafe_allow_html=True
        )
        if hasattr(st.session_state.model_obj, 'names'):
            st.markdown(f"""
            <div style="text-align: center; margin-top: 20px;">
                <p style="color: rgba(200, 230, 255, 0.9); font-size: 16px;">
                    <strong>Detectable classes:</strong> {', '.join(list(st.session_state.model_obj.names.values()))}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("""
    <div style="text-align: center; margin: 60px 0; padding: 40px; background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 0, 0, 0.3) 100%); border-radius: 20px;">
        <h2 style="font-size: 36px; font-weight: 800; color: #00d4ff; margin-bottom: 20px;">Ready to Start?</h2>
        <p style="font-size: 18px; color: rgba(200, 230, 255, 0.9); margin-bottom: 30px;">
            Upload images or videos to detect safety objects in real-time
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Functional buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📸 Upload Image", key="home_image_btn", use_container_width=True, type="primary"):
            st.session_state.page = "image"
            st.rerun()
    with col_btn2:
        if st.button("🎥 Upload Video", key="home_video_btn", use_container_width=True, type="primary"):
            st.session_state.page = "video"
            st.rerun()

def page_image_upload():
    """Image upload and detection page."""
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #00d4ff;">📸 Image Detection</h2>', unsafe_allow_html=True)
    
    # Model status check
    if st.session_state.model_obj is None:
        st.warning("⚠️ No model loaded. Please load a model on the Home page first.")
    
    # File uploader
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp"],
        key="uploader",
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    detect_button = st.button(
        "🔍 DETECT OBJECTS",
        key="detect_btn",
        use_container_width=True,
        type="primary"
    )
    
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            
            if detect_button:
                if st.session_state.model_obj is None:
                    st.error("❌ No model loaded. Please load a model first.")
                    st.image(pil_img)
                else:
                    with st.spinner("🔍 Running detection..."):
                        dets = run_inference_on_pil(
                            st.session_state.model_obj,
                            pil_img,
                            confidence_threshold=CONFIDENCE_THRESHOLD
                        )
                        out_img = pil_img.copy()
                        out_img = draw_detections_on_pil(out_img, dets)
                        
                        # Save output
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_name = f"det_{Path(uploaded_file.name).stem}_{timestamp}.jpg"
                        out_path = SAVED_DIR / out_name
                        out_img.save(out_path)
                        
                        # Save to history
                        save_detection_history(str(out_path), dets, uploaded_file.name)
                        
                        # Display results
                        st.image(out_img)
                        
                        # Statistics
                        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="stat-item"><div class="stat-value">{len(dets)}</div><div class="stat-label">Objects Detected</div></div>',
                            unsafe_allow_html=True
                        )
                        
                        # Count by class
                        class_counts = {}
                        for det in dets:
                            label = det['label']
                            class_counts[label] = class_counts.get(label, 0) + 1
                        
                        for label, count in class_counts.items():
                            st.markdown(
                                f'<div class="stat-item"><div class="stat-value">{count}</div><div class="stat-label">{label}</div></div>',
                                unsafe_allow_html=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Detection details
                        if dets:
                            st.markdown("### Detection Details")
                            for i, det in enumerate(dets, 1):
                                st.markdown(
                                    f"""
                                    <div class="detection-item">
                                    <strong>{i}. {det['label']}</strong><br>
                                    Confidence: {det['conf']:.2%} | 
                                    Position: ({det['bbox'][0]}, {det['bbox'][1]}) → ({det['bbox'][2]}, {det['bbox'][3]})
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.warning("No objects detected. Try lowering the confidence threshold.")
                        
                        st.success(f"✅ Detection complete! Saved to: `{out_path}`")
            else:
                st.image(pil_img)
                st.info("👆 Press 'DETECT OBJECTS' to run detection on this image.")
                
        except Exception as e:
            st.error(f"❌ Error reading image: {e}")
    else:
        st.info("👆 Upload an image above to start detection.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def page_history():
    """History page showing saved detections."""
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #00d4ff;">📊 Detection History</h2>', unsafe_allow_html=True)
    
    history = load_detection_history()
    files = sorted(SAVED_DIR.glob("*.jpg"), key=os.path.getmtime, reverse=True)
    
    if not files and not history:
        st.info("📭 No saved detections yet. Run a detection on the Home page to create saved images.")
    else:
        # Show history from JSON
        if history:
            st.markdown("### Recent Detections")
            for entry in reversed(history[-10:]):  # Show last 10
                # Safely get timestamp and other fields
                timestamp = entry.get('timestamp', 'Unknown')
                if isinstance(timestamp, str) and len(timestamp) > 19:
                    timestamp_display = timestamp[:19]
                else:
                    timestamp_display = str(timestamp)
                
                file_name = entry.get('file_name', 'Unknown')
                count = entry.get('count', 0)
                
                with st.expander(f"🕒 {timestamp_display} - {file_name} ({count} objects)"):
                    # Support both old format (image_path) and new format (file_path)
                    file_path = entry.get('file_path', entry.get('image_path', ''))
                    file_type = entry.get('file_type', 'image')
                    
                    if file_path and Path(file_path).exists():
                        # Check if it's a video file
                        if file_type == "video" or file_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            st.video(file_path)
                            st.markdown(f"**Video File:** `{file_path}`")
                        else:
                            st.image(file_path)
                            st.markdown(f"**Image File:** `{file_path}`")
                    detections = entry.get('detections', [])
                    if detections:
                        st.markdown("**Detections:**")
                        st.json(detections)
        
        # Show image files
        if files:
            st.markdown("### Saved Images")
            cols_per_row = 3
            for i in range(0, len(files), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(files):
                        p = files[i + j]
                        with col:
                            st.image(str(p))
                            st.caption(p.name)
                            if st.button(f"🗑️ Delete", key=f"del_{p.name}"):
                                p.unlink()
                                if HISTORY_FILE.exists():
                                    try:
                                        history = load_detection_history()
                                        history = [h for h in history if h['image_path'] != str(p)]
                                        with open(HISTORY_FILE, 'w') as f:
                                            json.dump(history, f, indent=2)
                                    except Exception:
                                        pass
                                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def page_video():
    """Video detection page with video output."""
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #00d4ff;">🎥 Video Detection</h2>', unsafe_allow_html=True)
    
    if st.session_state.model_obj is None:
        st.warning("⚠️ No model loaded. Please load a model on the Home page first.")
    
    video_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_uploader"
    )
    
    if video_file is not None:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix)
        t.write(video_file.read())
        t.flush()
        t.close()
        
        st.video(str(t.name))
        
        run_vid = st.button("🔍 RUN VIDEO DETECTION", key="run_video", use_container_width=True, type="primary")
        
        if run_vid:
            if st.session_state.model_obj is None:
                st.error("❌ No model loaded. Please load a model first.")
            else:
                with st.spinner("🎬 Processing video and generating annotated output (this may take a while)..."):
                    try:
                        import cv2
                        
                        # Run detection on video and save annotated video
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_video_name = f"video_det_{Path(video_file.name).stem}_{timestamp}.mp4"
                        out_video_path = SAVED_DIR / out_video_name
                        
                        # Use YOLO to predict and save annotated video
                        output_dir = SAVED_DIR / f"video_output_{timestamp}"
                        output_dir.mkdir(exist_ok=True, parents=True)
                        
                        results = st.session_state.model_obj.predict(
                            source=str(t.name),
                            imgsz=IMG_SIZE,
                            conf=CONFIDENCE_THRESHOLD,
                            save=True,
                            project=str(SAVED_DIR),
                            name=f"video_output_{timestamp}",
                            verbose=False
                        )
                        
                        # Find the saved video file - check multiple possible locations
                        video_files = []
                        
                        # Check in the specified output directory
                        if output_dir.exists():
                            video_files.extend(list(output_dir.glob("*.mp4")))
                            video_files.extend(list(output_dir.glob("*.avi")))
                        
                        # Check in runs/detect/predict (YOLO default location)
                        runs_predict = Path("runs/detect/predict")
                        if runs_predict.exists():
                            video_files.extend(list(runs_predict.glob("*.mp4")))
                            video_files.extend(list(runs_predict.glob("*.avi")))
                        
                        # Check in saved_detections directly
                        video_files.extend(list(SAVED_DIR.glob("*.mp4")))
                        video_files.extend(list(SAVED_DIR.glob("*.avi")))
                        
                        if video_files:
                            # Use the most recent video file
                            video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            detected_video = video_files[0]
                            
                            # Copy to final location
                            final_video_path = SAVED_DIR / out_video_name
                            if detected_video != final_video_path:
                                shutil.copy2(str(detected_video), str(final_video_path))
                            
                            # Get detection stats from first frame
                            cap = cv2.VideoCapture(str(t.name))
                            ret, frame = cap.read()
                            if ret:
                                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(img)
                                
                                # Get detections from first result
                                r = results[0]
                                boxes = getattr(r, "boxes", None)
                                detections = []
                                
                                if boxes is not None and len(boxes) > 0:
                                    try:
                                        xyxy = boxes.xyxy.cpu().numpy()
                                        confs = boxes.conf.cpu().numpy()
                                        cls_ids = boxes.cls.cpu().numpy().astype(int)
                                    except Exception:
                                        xyxy = np.array(boxes.xyxy)
                                        confs = np.array(boxes.conf)
                                        cls_ids = np.array(boxes.cls).astype(int)
                                    
                                    names = st.session_state.model_obj.names if hasattr(st.session_state.model_obj, "names") else CLASSES
                                    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, cls_ids):
                                        detections.append({
                                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                            "conf": float(conf),
                                            "label": names.get(int(cls), str(cls)) if isinstance(names, dict) else names[int(cls)]
                                        })
                                
                                cap.release()
                                
                                # Convert video to web-compatible format if needed
                                video_path_abs = str(final_video_path.resolve())
                                
                                if Path(video_path_abs).exists():
                                    # Check if video needs conversion for web compatibility
                                    web_video_path = SAVED_DIR / f"web_{out_video_name}"
                                    
                                    # Try to convert to web-compatible format (silent - no error messages)
                                    try:
                                        cap_test = cv2.VideoCapture(video_path_abs)
                                        if cap_test.isOpened():
                                            fps = int(cap_test.get(cv2.CAP_PROP_FPS)) or 30
                                            width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
                                            height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                            cap_test.release()
                                            
                                            # Try H.264 codec for web compatibility
                                            cap_convert = cv2.VideoCapture(video_path_abs)
                                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                                            out_convert = cv2.VideoWriter(str(web_video_path), fourcc, fps, (width, height))
                                            
                                            if out_convert.isOpened():
                                                while True:
                                                    ret, frame = cap_convert.read()
                                                    if not ret:
                                                        break
                                                    out_convert.write(frame)
                                                cap_convert.release()
                                                out_convert.release()
                                                video_path_abs = str(web_video_path.resolve())
                                            else:
                                                cap_convert.release()
                                                video_path_abs = str(final_video_path.resolve())
                                        else:
                                            cap_test.release()
                                    except Exception:
                                        video_path_abs = str(final_video_path.resolve())
                                    
                                    # Show preview frame first
                                    if detections:
                                        preview_img = draw_detections_on_pil(pil_img, detections)
                                        st.markdown("### 📸 Preview Frame with Detections")
                                        st.image(preview_img, use_container_width=True)
                                    
                                    st.success(f"✅ Annotated video saved: `{final_video_path}`")
                                    st.info(f"🎯 Detected {len(detections)} object(s) in the first frame. Video contains annotations for all frames.")
                                    
                                    # Save to history with video path
                                    save_detection_history(str(final_video_path), detections, video_file.name)
                                    
                                    # Display annotated video at the bottom
                                    st.markdown("### 📹 Annotated Video Output with Predictions")
                                    st.markdown("**Play the video below to see all detections:**")
                                    
                                    # Display video - try multiple methods for compatibility
                                    video_displayed = False
                                    try:
                                        # Method 1: Read as bytes
                                        with open(video_path_abs, 'rb') as video_file_bytes:
                                            video_bytes = video_file_bytes.read()
                                            if len(video_bytes) > 0:
                                                st.video(video_bytes, format="video/mp4", start_time=0)
                                                video_displayed = True
                                    except Exception:
                                        pass
                                    
                                    if not video_displayed:
                                        try:
                                            # Method 2: Path-based display
                                            st.video(video_path_abs, format="video/mp4", start_time=0)
                                        except Exception:
                                            try:
                                                # Method 3: Simple path display
                                                st.video(video_path_abs)
                                            except Exception:
                                                pass
                            else:
                                cap.release()
                        else:
                            # Fallback: Create video manually from frames
                            st.info("📹 Creating annotated video from frames...")
                            cap = cv2.VideoCapture(str(t.name))
                            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            # Use H.264 codec for web compatibility (avc1)
                            final_video_path = SAVED_DIR / out_video_name
                            
                            # Try H.264 codec first (best for web)
                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                            out_video = cv2.VideoWriter(str(final_video_path), fourcc, fps, (width, height))
                            
                            # If H.264 doesn't work, fallback to mp4v
                            if not out_video.isOpened():
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                out_video = cv2.VideoWriter(str(final_video_path), fourcc, fps, (width, height))
                            
                            frame_count = 0
                            detections = []
                            progress_bar = st.progress(0)
                            
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(frame_rgb)
                                
                                # Run detection on frame
                                r = results[frame_count] if frame_count < len(results) else results[0]
                                boxes = getattr(r, "boxes", None)
                                frame_dets = []
                                
                                if boxes is not None and len(boxes) > 0:
                                    try:
                                        xyxy = boxes.xyxy.cpu().numpy()
                                        confs = boxes.conf.cpu().numpy()
                                        cls_ids = boxes.cls.cpu().numpy().astype(int)
                                    except Exception:
                                        xyxy = np.array(boxes.xyxy)
                                        confs = np.array(boxes.conf)
                                        cls_ids = np.array(boxes.cls).astype(int)
                                    
                                    names = st.session_state.model_obj.names if hasattr(st.session_state.model_obj, "names") else CLASSES
                                    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, cls_ids):
                                        frame_dets.append({
                                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                            "conf": float(conf),
                                            "label": names.get(int(cls), str(cls)) if isinstance(names, dict) else names[int(cls)]
                                        })
                                
                                if frame_count == 0:
                                    detections = frame_dets
                                
                                annotated_img = draw_detections_on_pil(pil_img, frame_dets)
                                annotated_frame = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)
                                out_video.write(annotated_frame)
                                
                                frame_count += 1
                                progress_bar.progress(min(frame_count / 100, 1.0))
                            
                            cap.release()
                            out_video.release()
                            progress_bar.empty()
                            
                            video_path_abs = str(final_video_path.resolve())
                            if Path(video_path_abs).exists():
                                # Convert to web-compatible format (silent conversion)
                                web_video_path = SAVED_DIR / f"web_{out_video_name}"
                                
                                try:
                                    cap_convert = cv2.VideoCapture(video_path_abs)
                                    if cap_convert.isOpened():
                                        fps = int(cap_convert.get(cv2.CAP_PROP_FPS)) or 30
                                        width = int(cap_convert.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        height = int(cap_convert.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                        
                                        fourcc = cv2.VideoWriter_fourcc(*'avc1')
                                        out_convert = cv2.VideoWriter(str(web_video_path), fourcc, fps, (width, height))
                                        
                                        if out_convert.isOpened():
                                            while True:
                                                ret, frame = cap_convert.read()
                                                if not ret:
                                                    break
                                                out_convert.write(frame)
                                            cap_convert.release()
                                            out_convert.release()
                                            video_path_abs = str(web_video_path.resolve())
                                        else:
                                            cap_convert.release()
                                except Exception:
                                    pass
                                
                                st.success(f"✅ Annotated video created: `{final_video_path}`")
                                st.info(f"🎯 Processed {frame_count} frames. Detected {len(detections)} object(s) in the first frame.")
                                
                                # Display annotated video at the bottom
                                st.markdown("### 📹 Annotated Video Output with Predictions")
                                st.markdown("**Play the video below to see all detections:**")
                                
                                # Read video file as bytes for proper display
                                video_displayed = False
                                try:
                                    with open(video_path_abs, 'rb') as video_file_bytes:
                                        video_bytes = video_file_bytes.read()
                                        if len(video_bytes) > 0:
                                            st.video(video_bytes, format="video/mp4", start_time=0)
                                            video_displayed = True
                                except Exception:
                                    pass
                                
                                if not video_displayed:
                                    try:
                                        st.video(video_path_abs, format="video/mp4", start_time=0)
                                    except Exception:
                                        try:
                                            st.video(video_path_abs)
                                        except Exception:
                                            pass
                            else:
                                st.error(f"Video file not found at: {video_path_abs}")
                            save_detection_history(str(final_video_path), detections, video_file.name)
                            
                    except Exception as e:
                        st.error(f"❌ Video detection failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)

def page_live():
    """Live webcam page - browser camera for hosting compatibility."""
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #00d4ff;">🔴 Live Webcam</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 30px; text-align: center; background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 0, 0, 0.2) 100%); border-radius: 15px; margin: 20px 0;">
        <h3 style="color: #00d4ff; font-size: 28px; margin-bottom: 20px;">📹 Live Webcam Feed</h3>
        <p style="color: rgba(200, 230, 255, 0.9); font-size: 18px; margin-bottom: 30px;">
            Connect your webcam for live video feed (works in browser - perfect for hosting)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera source selection
    source = st.radio(
        "Select Camera Source:",
        ["Browser Camera (Recommended for Web)", "CCTV (Coming Soon)"],
        key="camera_source"
    )
    
    # CCTV placeholder
    if source == "CCTV (Coming Soon)":
        st.markdown("""
        <div style="background: rgba(255, 193, 7, 0.1); border: 2px solid rgba(255, 193, 7, 0.5); 
                    border-radius: 15px; padding: 40px; text-align: center; margin: 20px 0;">
            <div style="font-size: 64px; margin-bottom: 20px;">📹</div>
            <h3 style="color: #ffc107; margin-bottom: 15px;">CCTV Integration</h3>
            <p style="color: rgba(255, 255, 255, 0.9); font-size: 18px;">
                CCTV camera integration will be available soon. Stay tuned!
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Start/Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ START WEBCAM", key="start_webcam", use_container_width=True, type="primary"):
            st.session_state.live_detection = True
            st.success("🟢 Webcam started!")
            st.rerun()
    
    with col2:
        if st.button("⏹️ STOP WEBCAM", key="stop_webcam", use_container_width=True):
            st.session_state.live_detection = False
            st.info("🔴 Webcam stopped.")
            st.rerun()
    
    # Show status
    if st.session_state.get('live_detection', False):
        st.markdown("""
        <div style="background: rgba(0, 255, 136, 0.1); border: 1px solid rgba(0, 255, 136, 0.3); 
                    border-radius: 10px; padding: 15px; margin: 15px 0; text-align: center;">
            <p style="color: #00ff88; margin: 0; font-weight: 600;">
                🟢 <strong>Webcam Active</strong> - Camera feed is live
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(255, 193, 7, 0.1); border: 1px solid rgba(255, 193, 7, 0.3); 
                    border-radius: 10px; padding: 15px; margin: 15px 0; text-align: center;">
            <p style="color: #ffc107; margin: 0;">
                ⏸️ <strong>Webcam Stopped</strong> - Click "START WEBCAM" to begin
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Browser camera - works for hosting (no detection, just webcam feed)
    if st.session_state.get('live_detection', False):
        frame_placeholder = st.empty()
        
        try:
            # Use browser camera input - works in hosted environments
            camera_image = st.camera_input(
                "Live Webcam Feed",
                key="live_camera",
                help="Your webcam feed will appear here. Grant camera permission when prompted."
            )
            
            if camera_image is not None:
                # Just display the webcam feed - no detection needed
                pil_img = Image.open(camera_image).convert("RGB")
                frame_placeholder.image(pil_img, caption="📹 LIVE WEBCAM", channels="RGB", use_container_width=True)
                
                # Auto-refresh for continuous feed
                if st.session_state.get('live_detection', False):
                    st.rerun()
            else:
                frame_placeholder.info("👆 Click the camera button above to start webcam feed. Grant permission when prompted.")
                
        except Exception as e:
            # Silent error handling for hosting
            frame_placeholder.info("💡 Please grant camera permission to start webcam feed.")
            if st.session_state.get('live_detection', False):
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def page_about():
    """About page."""
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #00d4ff;">ℹ️ About</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🚀 Space Station Safety Objects Detection
    
    This application uses YOLO (You Only Look Once) deep learning models to detect 
    critical safety objects inside a space station interior.
    
    **Detectable Objects:**
    - 🫁 Oxygen Tank
    - 🫁 Nitrogen Tank
    - 🏥 First Aid Box
    - 🔔 Fire Alarm
    - 🔌 Safety Switch Panel
    - 📞 Emergency Phone
    - 🧯 Fire Extinguisher
    
    **Features:**
    - 📸 Image detection with bounding boxes and confidence scores
    - 🎥 Video frame detection
    - 📊 Detection history and statistics
    - ⚙️ Adjustable confidence threshold
    - 💾 Automatic saving of detection results
    
    **Model Information:**
    """)
    
    st.write(f"**Default model path:** `{MODEL_PATH}`")
    
    if st.session_state.model_obj is None:
        st.warning("⚠️ Model is not currently loaded.")
        if st.session_state.model_msg:
            st.write(f"**Last error:** {st.session_state.model_msg}")
    else:
        st.success("✅ Model loaded and ready for inference.")
        if hasattr(st.session_state.model_obj, 'names'):
            st.write("**Detectable classes:**", list(st.session_state.model_obj.names.values()))
    
    st.markdown("""
    ---
    **Built with:** Streamlit · Ultralytics YOLO · Python
    
    *For space station safety monitoring and object detection*
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Page Router ----------
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "image":
    page_image_upload()
elif st.session_state.page == "video":
    page_video()
elif st.session_state.page == "live":
    page_live()
elif st.session_state.page == "history":
    page_history()
elif st.session_state.page == "about":
    page_about()
else:
    page_home()

st.markdown(
    '<div class="footer-note">Built with Streamlit · YOLO Detection · Space Station Safety</div>',
    unsafe_allow_html=True
)
