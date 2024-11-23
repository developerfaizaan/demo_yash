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

# Manage login/logout state using st.session_state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Function to handle logout
def logout():
    st.session_state.logged_in = False
    st.experimental_set_query_params()  # Clear any login-related query parameters
    st.experimental_rerun()

# Check for login state in URL query parameters
query_params = st.experimental_get_query_params()
if "login_success" in query_params:
    st.session_state.logged_in = True

# Custom Styling for the Top Bar
st.markdown("""
    <style>
    .top-bar { 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        background-color: #f0f2f6; 
        padding: 10px 20px; 
        border-bottom: 1px solid #ccc;
    }
    .top-bar a, .top-bar button { 
        text-decoration: none; 
        color: #4CAF50; 
        font-weight: bold; 
        margin-right: 20px;
        border: none;
        background: none;
        cursor: pointer;
    }
    .top-bar a:hover, .top-bar button:hover { color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

# Top Bar: Login or Logout
if st.session_state.logged_in:
    st.markdown("""
        <div class="top-bar">
            <span>Welcome, User! 🎉</span>
            <button onclick="window.location.href='/';">🔓 Logout</button>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="top-bar">
            <a href="https://detectifilogin.netlify.app/" target="_self">🔒 Login</a>
        </div>
    """, unsafe_allow_html=True)

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

source_img = None
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Upload Image 🖼️:", type=("jpg", "jpeg", "png", "bmp", "webp"),
        help="Upload an image for object detection."
    )

# Sidebar: Image Adjustment
st.sidebar.markdown('<h3>🎨 Image Adjustments</h3>', unsafe_allow_html=True)

contrast = st.sidebar.slider("Adjust Contrast 🌟:", 0.5, 3.0, 1.0, 0.1)
brightness = st.sidebar.slider("Adjust Brightness 🌞:", 0.5, 3.0, 1.0, 0.1)

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
            
            enhancer_contrast = ImageEnhance.Contrast(uploaded_image)
            adjusted_image = enhancer_contrast.enhance(contrast)
            
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
                res = model.predict(adjusted_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image ✅', use_column_width=True)

                object_counts = {}
                for box in boxes:
                    label = box.label if hasattr(box, 'label') else "Object in Image"
                    object_counts[label] = object_counts.get(label, 0) + 1

                st.markdown("### Detection Summary 📝")
                for label, count in object_counts.items():
                    st.write(f"- **{label.capitalize()}**: {count}")
            except Exception as ex:
                st.error("❌ Error occurred during object detection.")
                st.error(ex)

# Footer
st.markdown("""
    ---
    <div style="text-align: center;">
        <p>🌟 Developed by <b>Your Name/Team</b> | Contact: <a href="mailto:your-email@example.com">your-email@example.com</a></p>
    </div>
""", unsafe_allow_html=True)
