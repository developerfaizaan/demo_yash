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

# Initialize Session State for Authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Function to Handle Login
def login():
    st.session_state.logged_in = True

# Function to Handle Logout
def logout():
    st.session_state.logged_in = False

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
    .top-bar button { 
        background: none; 
        border: none; 
        color: #4CAF50; 
        font-size: 16px; 
        font-weight: bold; 
        cursor: pointer; 
    }
    .top-bar button:hover { color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

# Adding a top bar with Login/Logout option
if st.session_state.logged_in:
    st.markdown("""
        <div class="top-bar">
            <button onclick="window.location.reload();">🔓 Logout</button>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="top-bar">
            <a href="https://detectifilogin.netlify.app/" target="_self" onclick="window.parent.location.reload();">🔒 Login</a>
        </div>
    """, unsafe_allow_html=True)

# Simulate Login for Testing (Replace with actual login logic)
if not st.session_state.logged_in:
    st.sidebar.button("Simulate Login", on_click=login)
else:
    st.sidebar.button("Simulate Logout", on_click=logout)

# Main Page Title
st.markdown('<h1>🤖 Object Detection and Tracking</h1>', unsafe_allow_html=True)

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
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Upload Image 🖼️:", type=("jpg", "jpeg", "png", "bmp", "webp"),
        help="Upload an image for object detection."
    )

# Sidebar: Image Adjustment
st.sidebar.markdown('<h3>🎨 Image Adjustments</h3>', unsafe_allow_html=True)

# Contrast Slider
contrast = st.sidebar.slider(
    "Adjust Contrast 🌟:",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="Increase or decrease the contrast of the uploaded image."
)

# Brightness Slider
brightness = st.sidebar.slider(
    "Adjust Brightness 🌞:",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="Increase or decrease the brightness of the uploaded image."
)

# Load Pre-trained ML Model
st.sidebar.markdown('<h3>🚀 Model Status</h3>', unsafe_allow_html=True)
model_path = Path(settings.DETECTION_MODEL if model_type == 'Detection' else settings.SEGMENTATION_MODEL)
try:
    with st.spinner("Loading the model... ⏳"):
        model = helper.load_model(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
except Exception as ex:
    st.sidebar.error("❌ Error loading the model.")
    st.sidebar.error(ex)

# Main Page: Image Analysis
st.markdown('<h2>🖼️ Image Detection</h2>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    if source_img:
        try:
            uploaded_image = Image.open(source_img)
            
            # Adjust Contrast
            enhancer_contrast = ImageEnhance.Contrast(uploaded_image)
            adjusted_image = enhancer_contrast.enhance(contrast)
            
            # Adjust Brightness
            enhancer_brightness = ImageEnhance.Brightness(adjusted_image)
            adjusted_image = enhancer_brightness.enhance(brightness)
            
            st.image(adjusted_image, caption="Adjusted Image 🖼️", use_column_width=True)
        except Exception as ex:
            st.error("❌ Error occurred while adjusting the image.")
            st.error(ex)
    else:
        st.info("📥 Please upload an image to proceed.")

with col2:
    if source_img and st.button("Detect Objects 🔍"):
        with st.spinner("Detecting objects... 🛠️"):
            try:
                # Perform detection using adjusted image
                res = model.predict(adjusted_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image ✅', use_column_width=True)

                # Display results...
            except Exception as ex:
                st.error("❌ Error occurred during object detection.")
                st.error(ex)

# Footer with Icons
st.markdown("""
    ---
    <div style="text-align: center;">
        <p>🌟 Developed by <b>Your Name/Team</b> | Contact: <a href="mailto:your-email@example.com">your-email@example.com</a></p>
    </div>
""", unsafe_allow_html=True)
