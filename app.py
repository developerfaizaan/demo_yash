import streamlit as st

# Initialize session state for user authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# Page Configuration
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header with Login and Sign Up
with st.container():
    col1, col2 = st.columns([4, 1])

    with col1:
        st.title("Object Detection and Tracking")

    with col2:
        if st.session_state["authenticated"]:
            st.markdown(f"**Welcome, {st.session_state['user']}!**")
            if st.button("Logout"):
                st.session_state["authenticated"] = False
                st.session_state["user"] = None
                st.experimental_rerun()
        else:
            if st.button("Login / Sign Up"):
                st.session_state["show_modal"] = True

# Login and Sign Up Modal
if "show_modal" in st.session_state and st.session_state["show_modal"]:
    with st.expander("🔑 Login / Sign Up", expanded=True):
        st.markdown("### Login")
        login_email = st.text_input("Email", placeholder="Enter your email")
        login_password = st.text_input("Password", type="password", placeholder="Enter your password")
        if st.button("Login"):
            if login_email == "user@example.com" and login_password == "password123":  # Replace with actual auth logic
                st.success("Login successful!")
                st.session_state["authenticated"] = True
                st.session_state["user"] = login_email
                st.session_state["show_modal"] = False
                st.experimental_rerun()
            else:
                st.error("Invalid email or password.")

        st.markdown("### Sign Up")
        signup_email = st.text_input("New Email", placeholder="Enter a new email")
        signup_password = st.text_input("New Password", type="password", placeholder="Create a password")
        if st.button("Sign Up"):
            st.success("Sign up successful! You can now log in.")
            st.session_state["show_modal"] = False
