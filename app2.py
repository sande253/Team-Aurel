import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from numpy.linalg import norm
import pickle as pkl
from sklearn.neighbors import NearestNeighbors
import uuid

# Page config
st.set_page_config(page_title="Fashion Image Recommender", page_icon="👗", layout="centered")

st.title('👗 Fashion Image Recommender')

# Set paths dynamically
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGE_FEATURES_PATH = os.path.join(BASE_DIR, 'Images_features.pkl')
FILENAMES_PATH = os.path.join(BASE_DIR, 'filenames.pkl')
IMAGE_DIRECTORY = os.path.join(BASE_DIR, 'fashion-dataset', 'images')
TEMP_DIR = os.path.join(BASE_DIR, 'temp_files')

# Ensure the temporary directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Load precomputed image features and filenames
image_features = pkl.load(open(IMAGE_FEATURES_PATH, 'rb'))
filenames = pkl.load(open(FILENAMES_PATH, 'rb'))

# Load pre-trained ResNet50 model for feature extraction
@st.cache_resource
def load_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])
    return model

model = load_model()

# Initialize NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(image_features)

def extract_features_from_image(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    features = model.predict(img_preprocess).flatten()
    return features / norm(features)

uploaded_file = st.file_uploader("Upload an image for fashion recommendations", type=["png", "jpg", "jpeg"])

if uploaded_file:
    temp_image_filename = f"temp_{uuid.uuid4().hex}_{uploaded_file.name}"
    temp_image_path = os.path.join(TEMP_DIR, temp_image_filename)

    # Save the uploaded file to a temporary location
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features from the uploaded image
    input_img_features = extract_features_from_image(temp_image_path, model)

    # Find the most similar images using NearestNeighbors
    distances, indices = neighbors.kneighbors([input_img_features])

    # Get recommended image paths (excluding the first one, which is the uploaded image itself)
    recommended_images = [filenames[idx] for idx in indices[0][1:6]]

    st.write("### Recommended Images")

    for img_path in recommended_images:
        image_url = os.path.join(IMAGE_DIRECTORY, os.path.basename(img_path))
        st.image(image_url, caption=os.path.basename(img_path), use_column_width=True)

    # Clean up the temporary image file
    os.remove(temp_image_path)
