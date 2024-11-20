# Importing Required Libraries
from pathlib import Path
from PIL import Image, ImageEnhance
import streamlit as st
import settings
import helper

# Setting Page Layout
st.set_page_config(
    page_title="Object Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .stSidebar {background-color: #f0f2f6;}
    .stButton > button {background-color: #4CAF50; color: white;}
    </style>
""", unsafe_allow_html=True)

# Main Page Title
st.title("Object Detection and Tracking 🤖")

# Sidebar Configuration
st.sidebar.title("Configuration Panel")

# Sidebar: Model Settings
st.sidebar.subheader("Model Settings")
model_type = st.sidebar.radio(
    "Task Type:",
    options=['Detection', 'Segmentation'],
    help="Choose the type of task your ML model will perform."
)

confidence = st.sidebar.slider(
    "Confidence Threshold:",
    min_value=25,
    max_value=100,
    value=40,
    help="Set the minimum confidence level for object detection."
) / 100

# Sidebar: Source Settings
st.sidebar.subheader("Source Settings")
source_radio = st.sidebar.radio(
    "Input Source:",
    options=settings.SOURCES_LIST,
    help="Choose the source of the image or video for detection."
)

# Conditional Sidebar Inputs Based on Source
source_img = None
video_path = None
youtube_url = None

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Upload Image:", type=("jpg", "jpeg", "png", "bmp", "webp"),
        help="Upload an image for object detection."
    )
elif source_radio == settings.VIDEO:
    video_path = st.sidebar.text_input(
        "Video Path:", placeholder="Enter video file path...",
        help="Provide the path to a video file for detection."
    )
elif source_radio == settings.WEBCAM:
    st.sidebar.info("Webcam source will use your device's camera.")
elif source_radio == settings.YOUTUBE:
    youtube_url = st.sidebar.text_input(
        "YouTube URL:", placeholder="Paste YouTube video URL...",
        help="Provide the URL of a YouTube video."
    )

# Sidebar: Image Enhancements
if source_img:
    st.sidebar.subheader("Image Enhancements")
    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)

# Load Pre-trained ML Model
st.sidebar.subheader("Model Status")
model_path = None

if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

try:
    with st.spinner("Loading the model..."):
        model = helper.load_model(model_path)
        st.sidebar.success("Model loaded successfully!")
except Exception as ex:
    st.sidebar.error("Error loading the model.")
    st.sidebar.error(ex)

# Main Page: Tabs for Image and Video Analysis
tab1, tab2 = st.tabs(["📷 Image Analysis", "📹 Video Analysis"])

# Tab 1: Image Analysis
with tab1:
    st.header("Image Detection")
    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            try:
                uploaded_image = Image.open(source_img)
                # Apply Enhancements
                uploaded_image = ImageEnhance.Brightness(uploaded_image).enhance(brightness)
                uploaded_image = ImageEnhance.Contrast(uploaded_image).enhance(contrast)

                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening or enhancing the image.")
                st.error(ex)
        else:
            st.info("Please upload an image to proceed.")

    with col2:
        if source_img and st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                try:
                    res = model.predict(uploaded_image, conf=confidence)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image', use_column_width=True)

                    # Display Results
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.error("Error occurred during object detection.")
                    st.error(ex)

# Tab 2: Video Analysis
with tab2:
    st.header("Video Detection")
    if source_radio == settings.VIDEO and video_path:
        st.video(video_path)
        if st.button("Start Video Detection"):
            helper.play_stored_video(confidence, model)
    elif source_radio == settings.WEBCAM:
        if st.button("Start Webcam"):
            helper.play_webcam(confidence, model)
    elif source_radio == settings.RTSP:
        if st.button("Start RTSP Stream"):
            helper.play_rtsp_stream(confidence, model)
    elif source_radio == settings.YOUTUBE and youtube_url:
        if st.button("Start YouTube Stream"):
            helper.play_youtube_video(confidence, model)
    else:
        st.info("Please provide a valid source for video analysis.")

# Footer
st.markdown("""
    ---
    **Developed by [Your Name/Team]** | For inquiries, contact: [your-email@example.com]
""")
