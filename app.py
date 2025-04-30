import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load pre-trained ResNet50 model (excluding top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Wrap with a pooling layer to get feature vectors
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Function to extract normalized features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# List of image file paths
# Replace 'images' with the path to your dataset directory
dataset_path = r'C:\Users\Bolgamvaishnavi\Downloads\project\project\myntradataset\images'
filenames = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Extract features for all images
feature_list = []
for file in tqdm(filenames, desc="Extracting features"):
    feature_list.append(extract_features(file, model))

# Save features and filenames
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

print("✅ Feature extraction complete. Files saved:")
print(" - embeddings.pkl")
print(" - filenames.pkl")
