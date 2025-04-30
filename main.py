import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from datetime import datetime

# ------------------------------
# Load pre-trained model and embeddings
# ------------------------------
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# ------------------------------
# User Storage Utilities
# ------------------------------
USERS_FILE = "users.pkl"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "rb") as f:
            return pickle.load(f)
    else:
        return {}

def save_users(users):
    with open(USERS_FILE, "wb") as f:
        pickle.dump(users, f)

# ------------------------------
# Login or Register
# ------------------------------
def login():
    st.set_page_config(page_title="Fashion Recommender Login", layout="centered")
    st.markdown("<h2 style='text-align: center;'>👗 Fashion Recommendation System</h2>", unsafe_allow_html=True)

    users = load_users()

    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        with st.form(key="login_form"):
            st.markdown("### Enter username and password")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            new_account = st.checkbox("Create new account")
            submit_button = st.form_submit_button("Continue")

            if submit_button:
                if not username or not password:
                    st.error("Please enter both username and password.")
                elif new_account:
                    if username in users:
                        st.warning("Username already exists. Try logging in instead.")
                    else:
                        users[username] = {"password": password, "name": username.title(), "history": []}
                        save_users(users)
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        st.success(f"Account created! Welcome, {username.title()}!")
                        st.rerun()
                else:
                    if username in users and users[username]["password"] == password:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        st.success(f"Welcome back, {username.title()}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

# ------------------------------
# Save Uploaded File
# ------------------------------
def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# ------------------------------
# Feature Extraction
# ------------------------------
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# ------------------------------
# Recommendation Function
# ------------------------------
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# ------------------------------
# Main Recommender App
# ------------------------------
def main_app():
    st.title('👗 Fashion Recommender System')

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=False)

            # Feature extraction
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

            # Get recommendations
            indices = recommend(features, feature_list)

            # Show recommended images
            st.markdown("### Recommended Fashion Items")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    st.image(filenames[indices[0][i]], use_container_width=True)

            # Save to user history
            users = load_users()
            username = st.session_state["username"]

            if "history" not in users[username]:
                users[username]["history"] = []

            users[username]["history"].append({
                "uploaded_image": os.path.join("uploads", uploaded_file.name),
                "recommendations": [filenames[indices[0][i]] for i in range(5)],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Add timestamp
            })

            save_users(users)

    # Show recent uploads and recommendations
    st.markdown("---")
    st.markdown("## 🕘 Your Recent Uploads & Recommendations")

    users = load_users()
    history = users[st.session_state["username"]].get("history", [])

    if not history:
        st.info("You haven't uploaded anything yet.")
    else:
        for entry in reversed(history[-3:]):  # Show last 3 uploads
            st.image(entry["uploaded_image"], caption=f"Your Upload ({entry['timestamp']})", width=150)
            cols = st.columns(5)
            for i, img_path in enumerate(entry["recommendations"]):
                with cols[i]:
                    st.image(img_path, use_container_width=True)
            st.markdown("---")

    # Logout
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

# ------------------------------
# Main Entry Point
# ------------------------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state["logged_in"]:
        main_app()
    else:
        login()

if __name__ == "__main__":
    main()
