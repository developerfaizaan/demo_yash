from pathlib import Path
import PIL
import streamlit as st
import settings
import helper

# Page Configurations
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Page Header
st.title("🔍 Object Detection and Tracking")
st.markdown("A simple tool for object detection and segmentation using ML models. Configure the settings in the sidebar and upload your image or video to get started.")

# Sidebar - Model Configuration
st.sidebar.header("⚙️ Model Configuration")
st.sidebar.write("Adjust the settings below to configure the detection/segmentation task.")

# Model Options
model_type = st.sidebar.radio(
    "Task Type:", ['Detection', 'Segmentation'], index=0, help="Choose the ML task you want to perform."
)

confidence = st.sidebar.slider(
    "Model Confidence:", 25, 100, 40, step=5, help="Set the minimum confidence threshold for detection results."
) / 100

# Determine model path based on task
model_path = Path(settings.DETECTION_MODEL if model_type == 'Detection' else settings.SEGMENTATION_MODEL)

# Load Model
st.sidebar.write("---")
st.sidebar.subheader("📂 Load Model")
try:
    model = helper.load_model(model_path)
    st.sidebar.success(f"{model_type} model loaded successfully!")
except Exception as ex:
    st.sidebar.error("Unable to load model. Check the file path.")
    st.sidebar.error(ex)

# Sidebar - Source Configuration
st.sidebar.header("🎥 Source Configuration")
st.sidebar.write("Select the input source for object detection.")

source_radio = st.sidebar.radio(
    "Source Type:", settings.SOURCES_LIST, help="Choose the input source (image, video, webcam, etc.)"
)

# Main Content
if source_radio == settings.IMAGE:
    st.header("🖼️ Image Input")
    st.markdown("Upload an image to detect objects. If no image is uploaded, a default image will be used.")

    source_img = st.sidebar.file_uploader(
        "Upload an image file:", type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image, caption="Default Image", use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        except Exception as ex:
            st.error("Error loading the image.")
            st.error(ex)

    with col2:
        st.subheader("Detected Results")
        if source_img and st.sidebar.button("🔍 Detect Objects"):
            try:
                res = model.predict(uploaded_image, conf=confidence)
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption="Detected Image", use_column_width=True)

                with st.expander("Detection Results"):
                    for box in res[0].boxes:
                        st.write(box.data)
            except Exception as ex:
                st.error("Detection failed. Ensure a valid image is uploaded.")
                st.error(ex)
        elif source_img is None:
            st.info("Upload an image to see detection results.")

elif source_radio == settings.VIDEO:
    st.header("🎥 Video Input")
    st.markdown("Upload a video or use another source for object detection.")
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    st.header("📸 Webcam Input")
    st.markdown("Use your webcam to perform object detection in real-time.")
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    st.header("📡 RTSP Stream")
    st.markdown("Provide an RTSP stream URL for real-time detection.")
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    st.header("🎥 YouTube Video")
    st.markdown("Enter a YouTube video URL to detect objects from the video stream.")
    helper.play_youtube_video(confidence, model)

else:
    st.error("⚠️ Please select a valid source type!")

# Footer
st.write("---")
st.markdown(
    "Developed with ❤️ using [Streamlit](https://streamlit.io/)."
)
