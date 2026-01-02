"""
Streamlit dashboard: Automated Toll Tax Monitoring System
High-End UI Overhaul - Final Polish (Dark Uploader, Animations, HUD Header)
"""

import streamlit as st
import cv2
import numpy as np
import time
from collections import defaultdict
import tempfile
import os
from ultralytics import YOLO

# ---------------------------
# CONFIGURATION
# ---------------------------
st.set_page_config(
    page_title="Smart Toll AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config.py if present
DEFAULT_TOLL = {"car": 50, "truck": 200, "van": 100, "bus": 150}
DEFAULT_WEIGHTS = None
DEFAULT_TRACKER = "botsort.yaml"

try:
    import config as cfg

    TOLL = getattr(cfg, "TOLL_RATES", DEFAULT_TOLL)
    DEFAULT_WEIGHTS = getattr(cfg, "YOLO_WEIGHTS", DEFAULT_WEIGHTS)
    DEFAULT_TRACKER = getattr(cfg, "TRACKER_CONFIG", DEFAULT_TRACKER)
except Exception:
    TOLL = DEFAULT_TOLL

CLASS_NAMES = ["car", "truck", "van", "bus"]

# ---------------------------
# CUSTOM CSS (THE WOW FACTOR)
# ---------------------------
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700;800&family=Inter:wght@400;600&display=swap');

    /* GLOBAL THEME */
    .stApp {
        background-color: #020617; /* Very Dark Navy */
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* ========================================= */
    /* HEADER STYLING (HUD STYLE)           */
    /* ========================================= */
    .hud-header-container {
        text-align: center;
        padding: 20px 0 40px 0;
        animation: fadeIn 1.2s ease-out;
    }
    .hud-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 5rem;
        font-weight: 800;
        margin: 0;
        line-height: 1;
        background: linear-gradient(to bottom, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(56, 189, 248, 0.3);
        letter-spacing: 2px;
    }
    .hud-subtitle {
        font-family: 'Rajdhani', sans-serif;
        color: #38bdf8; /* Sky Blue */
        font-size: 1.5rem;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin-top: 10px;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
    }
    .hud-line {
        height: 2.25px;
        width: 150px;
        background: linear-gradient(90deg, transparent, #38bdf8, transparent);
        margin: 20px auto;
    }

    /* ========================================= */
    /* GLASS CARDS & ANIMATIONS             */
    /* ========================================= */
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }

    /* HOVER EFFECT - THE "POP" */
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(56, 189, 248, 0.5); /* Blue Glow Border */
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4),
                    0 0 20px rgba(56, 189, 248, 0.2); /* Outer Glow */
        background: rgba(30, 41, 59, 0.8);
    }

    /* REVENUE CARD SPECIAL STYLE */
    .revenue-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 78, 59, 0.3) 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .revenue-card:hover {
        border-color: #10b981;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.3);
    }

    .metric-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.1;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        margin-top: 5px;
    }

    /* ========================================= */
    /* SIDEBAR & FILE UPLOADER FIX          */
    /* ========================================= */
    [data-testid="stSidebar"] {
        background-color: #E6E6E6;
        border-right: 2px solid #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #2E2E2E !important;
    }

    /* FIX FOR FILE UPLOADER - BLACK BACKGROUND */
    [data-testid='stFileUploader'] {
        width: 100%;
    }
    [data-testid='stFileUploader'] section {
        padding: 0;
        background-color: #E6E6E6 !important;
        color: 2E2E2E !important;
    }
    /* The dropzone container */
    .stFileUploaderDropzone {
        background-color: #E6E6E6 !important; /* Dark Grey */
        border: 2px dashed #3b82f6 !important; /* Blue dashed border */
        border-radius: 10px;

    }
    .stFileUploaderDropzone:hover {
        background-color: #E6E6E6 !important; /* Lighter grey on hover */
        border-color: #60a5fa !important;
    }
    /* The small instruction text inside dropzone */
    .stFileUploaderDropzone div, .stFileUploaderDropzone span, .stFileUploaderDropzone small {
        color: 2E2E2E !important;
    }
    /* The "Browse files" button */
    .stFileUploaderDropzone button {
        border: 1px solid #3b82f6;
        background-color: #E6E6E6;
        color: 2E2E2E;
    }

    /* RADIO BUTTONS SCALING */
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: 22px !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        transform: scale(1.3);
        border-color: #38bdf8 !important;
    }

    /* INPUT FIELDS */
    .stTextInput > div > div > input {
        background-color: #E6E6E6 !important;
        color: 2E2E2E !important;
        border: 1px solid #334155 !important;
        font-size: 18px !important;
        border-radius: 8px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.3) !important;
    }

    /* BUTTONS */
    div.stButton > button {
        background: linear-gradient(92deg, #2563eb 0%, #3b82f6 100%);
        border: none;
        color: 2E2E2E;
        padding: 16px 28px;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.6); /* Glow Effect */
    }

    /* ANIMATIONS */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes popIn { from { opacity: 0; transform: scale(0.8); } to { opacity: 1; transform: scale(1); } }

    /* FOOTER */
    .custom-footer {
        background: rgba(11, 17, 32, 0.9);
        border-top: 1px solid rgba(56, 189, 248, 0.2);
        margin-top: 80px;
        padding: 60px 20px 30px 20px;
        text-align: center;
        border-radius: 30px 30px 0 0;
        box-shadow: 0 -20px 50px rgba(0,0,0,0.3);
    }
    .footer-links a {
        color: #94a3b8;
        font-weight: 600;
        text-decoration: none;
        margin: 0 20px;
        font-size: 1.5rem;
        transition: all 0.3s ease;
        border-bottom: 2px solid transparent;
        padding-bottom: 5px;
    }
    .footer-links a:hover {
        color: #38bdf8;
        border-bottom-color: #38bdf8;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
<div class="hud-header-container">
    <div class="hud-line"></div>
    <h1 class="hud-title">SMART TOLL VISION</h1>
    <div class="hud-subtitle">Autonomous Vehicle Detection, Tracking & Taxation</div>
    <div class="hud-subtitle">The system provides a foundation for operational transparency & efficiency, enforcing strict accountability in daily tax collection</div>
    <div class="hud-line" style="width: 80px; margin-top: 15px; opacity: 0.5;"></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
with st.sidebar:
    st.markdown("### 🎛️ CONTROL CENTER")
    st.write("")  # Spacer

    st.markdown("**SOURCE SELECTION**")
    source_choice = st.radio(
        "Source",
        ("🎥 Upload Video", "📡 Live RTSP Stream"),
        label_visibility="collapsed"
    )

    rtsp_url = ""
    uploaded_file = None

    if source_choice == "📡 Live RTSP Stream":
        rtsp_url = st.text_input("RTSP Stream URL", placeholder="rtsp://192.168.1.5:554/stream")
    else:
        # File uploader (Style fixed in CSS)
        uploaded_file = st.file_uploader("Drop MP4 File Here", type=["mp4"])

    st.markdown("---")
    st.markdown("**MODEL CONFIGURATION**")
    weights_input = st.text_input("YOLO Weights Path", value=(DEFAULT_WEIGHTS or ""), placeholder="yolov8n.pt")
    tracker_input = st.text_input("Tracker Config", value=(DEFAULT_TRACKER or ""), placeholder="botsort.yaml")

    st.markdown("---")

    # Session State Init
    if "running" not in st.session_state: st.session_state.running = False
    if "counts" not in st.session_state: st.session_state.counts = defaultdict(int)
    if "total_toll" not in st.session_state: st.session_state.total_toll = 0
    if "counted_ids" not in st.session_state: st.session_state.counted_ids = set()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ START"):
            st.session_state.running = True
    with col2:
        if st.button("⏹ STOP"):
            st.session_state.running = False

    st.write("")
    if st.button("♻ RESET DATA"):
        st.session_state.counts = defaultdict(int)
        st.session_state.total_toll = 0
        st.session_state.counted_ids = set()
        st.toast("System Counters Reset", icon="🗑️")

# ---------------------------
# MAIN DASHBOARD AREA
# ---------------------------

# 1. METRICS ROW
st.markdown("### 📊 Live Traffic Data")
m_cols = st.columns(5)


def render_card(column, emoji, label, value, is_revenue=False):
    css_class = "glass-card revenue-card" if is_revenue else "glass-card"
    column.markdown(f"""
        <div class="{css_class}">
            <div style="font-size: 3rem; margin-bottom: 10px;">{emoji}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)


# Create placeholders
placeholders = {
    "car": m_cols[0].empty(),
    "truck": m_cols[1].empty(),
    "van": m_cols[2].empty(),
    "bus": m_cols[3].empty(),
    "toll": m_cols[4].empty()
}

# Initial Render: Only Revenue shows 0
render_card(placeholders['toll'], "💰", "Total Revenue", "Rs 0", is_revenue=True)
# Others are empty by default

# 2. VIDEO FEED
st.markdown("### 👁️ Surveillance Feed")
video_placeholder = st.empty()
video_placeholder.markdown(
    """
    <div style="background: rgba(15, 23, 42, 0.4); border: 2px dashed #334155; border-radius: 16px; height: 500px; display: flex; align-items: center; justify-content: center; box-shadow: inset 0 0 50px rgba(0,0,0,0.5);">
        <div style="text-align: center; color: #475569;">
            <i class="fas fa-video-slash" style="font-size: 4rem; margin-bottom: 20px; color: #334155;"></i><br>
            <div style="font-family: 'Rajdhani'; font-size: 1.5rem; letter-spacing: 2px;">AWAITING SIGNAL</div>
            <span style="font-size: 0.9rem; color: #64748b;">Select source and initialize system</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# LOGIC
# ---------------------------
def frame_to_jpeg_bytes(frame_bgr):
    ret, buf = cv2.imencode(".jpg", frame_bgr)
    return buf.tobytes() if ret else None


@st.cache_resource(show_spinner=False)
def load_yolo_model(path):
    return YOLO(path)


# Source Path Logic
selected_source_path = None
if source_choice == "📡 Live RTSP Stream" and rtsp_url:
    selected_source_path = rtsp_url.strip()
elif source_choice == "🎥 Upload Video" and uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    selected_source_path = tfile.name

if st.session_state.running:
    if not selected_source_path:
        st.error("⚠️ Source not selected.")
        st.session_state.running = False
    else:
        # load model
        try:
            weights_val = weights_input.strip() if weights_input.strip() else DEFAULT_WEIGHTS
            if not weights_val: raise ValueError("Weights not specified")

            model = load_yolo_model(weights_val)
            tracker_val = tracker_input.strip() if tracker_input.strip() else DEFAULT_TRACKER

            # Start Stream
            stream = model.track(source=selected_source_path, tracker=tracker_val, stream=True, verbose=False)

            toll_line_y = None
            direction = "down"
            previous_centroids = {}

            for r in stream:
                if not st.session_state.running: break

                frame = getattr(r, "orig_img", None)
                if frame is None: continue

                if toll_line_y is None:
                    toll_line_y = int(frame.shape[0] * 0.75)

                boxes = getattr(r, "boxes", None)
                if boxes:
                    xyxys = boxes.xyxy.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)

                    ids = None
                    if hasattr(boxes, 'id') and boxes.id is not None:
                        ids = boxes.id.cpu().numpy().astype(int)

                    if ids is not None:
                        for i, xyxy in enumerate(xyxys):
                            x1, y1, x2, y2 = map(int, xyxy)
                            cls_id = int(clss[i])
                            track_id = int(ids[i])
                            label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                            # Counting Logic
                            prev = previous_centroids.get(track_id, None)
                            if prev and track_id not in st.session_state.counted_ids:
                                prev_y = prev[1]
                                if direction == "down":
                                    if prev_y < toll_line_y <= cy:
                                        st.session_state.counted_ids.add(track_id)
                                        st.session_state.counts[label] += 1
                                        st.session_state.total_toll += int(TOLL.get(label, 0))

                            previous_centroids[track_id] = (cx, cy)

                            # Draw Bounding Box
                            # Neon Green Box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
                            # Label Background
                            (w, h), _ = cv2.getTextSize(f"{label}", 0, 0.6, 2)
                            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 127), -1)
                            cv2.putText(frame, f"{label}", (x1, y1 - 5), 0, 0.6, (0, 0, 0), 2)

                # Draw Line (Red -> Blue for high tech)
                cv2.line(frame, (0, toll_line_y), (frame.shape[1], toll_line_y), (0, 80, 255), 3)

                # DYNAMIC UPDATE OF WIDGETS
                # We check counts. If > 0, we render the card.

                if st.session_state.counts['car'] > 0:
                    render_card(placeholders['car'], "🚗", "Cars", st.session_state.counts['car'])

                if st.session_state.counts['truck'] > 0:
                    render_card(placeholders['truck'], "🚚", "Trucks", st.session_state.counts['truck'])

                if st.session_state.counts['van'] > 0:
                    render_card(placeholders['van'], "🚐", "Vans", st.session_state.counts['van'])

                if st.session_state.counts['bus'] > 0:
                    render_card(placeholders['bus'], "🚌", "Buses", st.session_state.counts['bus'])

                # Revenue always updates
                render_card(placeholders['toll'], "💰", "Total Revenue", f"Rs {st.session_state.total_toll}",
                            is_revenue=True)

                # Update Video
                jpeg = frame_to_jpeg_bytes(frame)
                if jpeg:
                    video_placeholder.image(jpeg, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown(
    """
    <div class="custom-footer">
        <div style="font-size: 2.5rem; font-weight: 800; font-family: 'Rajdhani', sans-serif; color: #f8fafc; margin-bottom: 10px; letter-spacing: 2px;">
            SMART TOLL TAX SYSTEM
        </div>
        <div class="footer-links" style="margin-bottom: 25px;">
            <a href="https://github.com/SobanHM/Automated-Toll-Tax-System" target="_blank"><i class="fab fa-github"></i> GitHub</a>
            <a href="https://www.linkedin.com/in/soban-hussaain-java-software-web-developer/" target="_blank"><i class="fab fa-linkedin"></i> LinkedIn</a>
            <a href="mailto:sobanhussainmahesar@gmail.com" target="_blank"><i class="fas fa-envelope"></i> Email</a>
        </div>
        <div style="color: #94a3b8; font-size: 1.5rem; line-height: 1.6; margin-bottom: 15px;">
            Developed by <strong>Soban Hussain & Muskan</strong><br>
            Instructor: <strong>Prof. Dr. Sher Muhammad</strong>
        </div>
        <div style="margin-top: 25px; color: #cream; font-size: 1rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 20px;">
            © 2025 Sukkur IBA University, Sindh, Pakistan. All Rights Reserved.
        </div>
    </div>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)
