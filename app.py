# Importing Required Libraries
from pathlib import Path
from PIL import Image, ImageEnhance
import streamlit as st
import settings
import helper

# Setting Page Layout
st.set_page_config(
    page_title="Object Detection 🤖",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for the Top Bar
st.markdown("""
    <style>
    .top-bar { 
        display: flex; 
        justify-content: flex-start; 
        align-items: center; 
        background-color: #f0f2f6; 
        padding: 10px 20px; 
        border-bottom: 1px solid #ccc;
    }
    .top-bar a { 
        text-decoration: none; 
        color: #4CAF50; 
        font-weight: bold; 
        margin-right: 20px;
    }
    .top-bar a:hover { color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

# Adding a top bar with Login option
st.markdown("""
    <div class="top-bar">
        <a href="/login" target="_self">🔒 Login<https://peppy-meringue-6406b8.netlify.app/>
    </div>
""", unsafe_allow_html=True)

# Main Page Title
st.markdown(
    '<h1>🤖 Object Detection and Tracking</h1>',
    unsafe_allow_html=True
)

# Sidebar Configuration
st.sidebar.markdown('<h2>⚙️ Configuration Panel</h2>', unsafe_allow_html=True)

# Sidebar: Model Settings
st.sidebar.markdown('<h3>📊 Model Settings</h3>', unsafe_allow_html=True)
model_type = st.sidebar.radio(
    "Task Type 🧩:",
    options=['Detection', 'Segmentation'],
    help="Choose the type of task your ML model will perform."
)

confidence = st.sidebar.slider(
    "Confidence Threshold 🎯:",
    min_value=25,
    max_value=100,
    value=40,
    help="Set the minimum confidence level for object detection."
) / 100

# Sidebar: Source Settings
st.sidebar.markdown('<h3>📥 Source Settings</h3>', unsafe_allow_html=True)
source_radio = st.sidebar.radio(
    "Input Source 📡:",
    options=settings.SOURCES_LIST,
    help="Choose the source of the image or video for detection."
)

# Conditional Sidebar Inputs Based on Source
source_img = None
video_path = None
youtube_url = None

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Upload Image 🖼️:", type=("jpg", "jpeg", "png", "bmp", "webp"),
        help="Upload an image for object detection."
    )
elif source_radio == settings.VIDEO:
    video_path = st.sidebar.text_input(
        "Video Path 🎥:", placeholder="Enter video file path...",
        help="Provide the path to a video file for detection."
    )
elif source_radio == settings.WEBCAM:
    st.sidebar.info("📸 Webcam source will use your device's camera.")
elif source_radio == settings.YOUTUBE:
    youtube_url = st.sidebar.text_input(
        "YouTube URL 📺:", placeholder="Paste YouTube video URL...",
        help="Provide the URL of a YouTube video."
    )

# Sidebar: Image Enhancements
if source_img:
    st.sidebar.markdown('<h3>🎨 Image Enhancements</h3>', unsafe_allow_html=True)
    brightness = st.sidebar.slider("Brightness ☀️", 0.5, 2.0, 1.0)
    contrast = st.sidebar.slider("Contrast 🎛️", 0.5, 2.0, 1.0)

# Load Pre-trained ML Model
st.sidebar.markdown('<h3>🚀 Model Status</h3>', unsafe_allow_html=True)
model_path = None

if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

try:
    with st.spinner("Loading the model... ⏳"):
        model = helper.load_model(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
except Exception as ex:
    st.sidebar.error("❌ Error loading the model.")
    st.sidebar.error(ex)

# Main Page: Tabs for Image and Video Analysis
tab1, tab2 = st.tabs(["📷 Image Analysis", "📹 Video Analysis"])

# Tab 1: Image Analysis
with tab1:
    st.markdown('<h2>🖼️ Image Detection</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            try:
                uploaded_image = Image.open(source_img)
                # Apply Enhancements
                uploaded_image = ImageEnhance.Brightness(uploaded_image).enhance(brightness)
                uploaded_image = ImageEnhance.Contrast(uploaded_image).enhance(contrast)

                st.image(uploaded_image, caption="Uploaded Image 🖼️", use_column_width=True)
            except Exception as ex:
                st.error("❌ Error occurred while opening or enhancing the image.")
                st.error(ex)
        else:
            st.info("📥 Please upload an image to proceed.")

    with col2:
        if source_img and st.button("Detect Objects 🔍"):
            with st.spinner("Detecting objects... 🛠️"):
                try:
                    res = model.predict(uploaded_image, conf=confidence)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image ✅', use_column_width=True)

                    # Display Results
                    with st.expander("📋 Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.error("❌ Error occurred during object detection.")
                    st.error(ex)

# Tab 2: Video Analysis
with tab2:
    st.markdown('<h2>🎥 Video Detection</h2>', unsafe_allow_html=True)
    if source_radio == settings.VIDEO and video_path:
        st.video(video_path)
        if st.button("Start Video Detection ▶️"):
            helper.play_stored_video(confidence, model)
    elif source_radio == settings.WEBCAM:
        if st.button("Start Webcam 📹"):
            helper.play_webcam(confidence, model)
    elif source_radio == settings.RTSP:
        if st.button("Start RTSP Stream 🌐"):
            helper.play_rtsp_stream(confidence, model)
    elif source_radio == settings.YOUTUBE and youtube_url:
        if st.button("Start YouTube Stream 📺"):
            helper.play_youtube_video(confidence, model)
    else:
        st.info("📡 Please provide a valid source for video analysis.")

# Footer with Icons
st.markdown("""
    ---
    <div style="text-align: center;">
        <p>🌟 Developed by <b>Your Name/Team</b> | Contact: <a href="mailto:your-email@example.com">your-email@example.com</a></p>
    </div>
""", unsafe_allow_html=True)
