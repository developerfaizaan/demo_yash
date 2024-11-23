# Importing Required Libraries
from pathlib import Path
from PIL import Image
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
        <a href="/login" target="_self">🔒 Login</a>
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
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Upload Image 🖼️:", type=("jpg", "jpeg", "png", "bmp", "webp"),
        help="Upload an image for object detection."
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
            st.image(uploaded_image, caption="Uploaded Image 🖼️", use_column_width=True)
        except Exception as ex:
            st.error("❌ Error occurred while opening the image.")
            st.error(ex)
    else:
        st.info("📥 Please upload an image to proceed.")

with col2:
    if source_img and st.button("Detect Objects 🔍"):
        with st.spinner("Detecting objects... 🛠️"):
            try:
                # Perform detection
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image ✅', use_column_width=True)

                # Define mapping for tensor types
                tensor_type_mapping = {
                    0: "Person",
                    7: "Truck",
                    2: "car" ,
                    63: "laptop" ,
                    3: "Bike" ,
                    41: "Cup"
                    
                }

                # Categorize detected objects
                object_counts = {}
                tensor_counts = {}  # For counting by tensor types

                for box in boxes:  # Loop through each detected box
                    # st.write(f"Detected: {box}")  # Display the box details

                    # Extract label and tensor type
                    label = box.label if hasattr(box, 'label') else "Object in Image"
                    tensor_type = int(box.cls[0])  # Extract tensor type as integer

                    # Map tensor type to a name if available
                    tensor_name = tensor_type_mapping.get(tensor_type, f"Tensor {tensor_type}")

                    # Update object counts by label
                    object_counts[label] = object_counts.get(label, 0) + 1

                    # Update counts by tensor type (mapped name)
                    tensor_counts[tensor_name] = tensor_counts.get(tensor_name, 0) + 1

                # Display Detection Summary
                st.markdown("### Detection Summary 📝")
                if object_counts:
                    st.write("**Counts of detected objects by label:**")
                    for label, count in object_counts.items():
                        st.write(f"- **{label.capitalize()}**: {count}")
                else:
                    st.write("No objects detected.")

                # Display Tensor Counts
                st.markdown("### Object Count Summary 🧮")
                if tensor_counts:
                    st.write("**Counts of detected objects by type:**")
                    for tensor_name, count in tensor_counts.items():
                        st.write(f"- **{tensor_name}**: {count}")
                else:
                    st.write("No tensor types detected.")

                # Display Detailed Results
                with st.expander("📋 Detailed Detection Results"):
                    for box in boxes:
                        st.write(box.data)  # Display raw detection data for each box

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
