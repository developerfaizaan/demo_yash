import streamlit as st
from pathlib import Path
from PIL import Image, ImageEnhance
import settings
import helper

# Setting Page Layout
st.set_page_config(
    page_title="Object Detection 🤖",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Login and Sign Up Section
st.sidebar.markdown('<h3>🔑 Sign Up / Login</h3>', unsafe_allow_html=True)

# Store users' credentials temporarily in session (for demo purposes)
if "users" not in st.session_state:
    st.session_state.users = {}

# Toggle between Login and Sign Up
auth_mode = st.sidebar.radio("Choose Action", ("Login", "Sign Up"))

if auth_mode == "Sign Up":
    # Sign Up Form
    st.sidebar.markdown("### Create a new account 📋")

    new_username = st.sidebar.text_input("Username 👤:")
    new_password = st.sidebar.text_input("Password 🔒:", type="password")
    confirm_password = st.sidebar.text_input("Confirm Password 🔒:", type="password")

    # Check if Sign Up information is valid
    if st.sidebar.button("Sign Up 📝"):
        if new_username and new_password == confirm_password:
            if new_username not in st.session_state.users:
                st.session_state.users[new_username] = new_password
                st.sidebar.success(f"Account created successfully for {new_username}! Please log in.")
            else:
                st.sidebar.warning("❌ Username already exists. Please try another.")
        elif new_password != confirm_password:
            st.sidebar.warning("❌ Passwords do not match. Please try again.")
        else:
            st.sidebar.warning("❌ Please fill all fields.")

elif auth_mode == "Login":
    # Login Form
    st.sidebar.markdown("### Login to your account 🔑")

    username = st.sidebar.text_input("Username 👤:")
    password = st.sidebar.text_input("Password 🔒:", type="password")

    demo_username = "demo_user"
    demo_password = "demo_pass"

    # Check Login Credentials
    if st.sidebar.button("Login 🔑"):
        if username == demo_username and password == demo_password:
            st.session_state.logged_in = True
            st.sidebar.success("✅ Login successful! Welcome to the demo.")
        elif username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.sidebar.success(f"✅ Welcome back, {username}!")
        else:
            st.session_state.logged_in = False
            st.sidebar.warning("❌ Incorrect username or password. Please try again.")

# Main Page Only Accessible After Successful Login
if "logged_in" in st.session_state and st.session_state.logged_in:
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

                    # Define mapping for tensor types
                    tensor_type_mapping = {
                        0: "Person",
                        7: "Truck",
                        2: "Car",
                        63: "Laptop",
                        3: "Bike",
                        41: "Cup",
                        16: "Dog",
                        58: "Potted Plant"
                    }

                    # Categorize detected objects
                    object_counts = {}
                    tensor_counts = {}  # For counting by tensor types

                    for box in boxes:  # Loop through each detected box
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

else:
    st.warning("⚠️ Please login to access the app.")
    st.stop()  # Prevent further execution if not logged in
