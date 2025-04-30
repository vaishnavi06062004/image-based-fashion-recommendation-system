 #  👗 Image-Based Fashion Recommendation System
An AI-powered web application that recommends visually similar fashion items based on an uploaded image. The app includes secure login/logout, recent upload tracking, and real-time visual similarity recommendations using deep learning. Built with Streamlit and powered by a Kaggle fashion image dataset.
# 📦 Dataset
We use the Fashion Product Images (Small) on Kaggle, which contains ~44,000 images of various fashion items such as t-shirts, dresses, shoes, etc.
# https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
# 🚀 Features
✅ Upload a fashion image and get similar product recommendations
✅ Deep learning-based visual similarity search
✅ Login and logout functionality with user session control
✅ Displays a list of recent uploads by the user
✅ Simple and fast Streamlit user interface
## 🛠️ Tech Stack
**Frontend**: Streamlit
**Backend**: Python-3.12.4
**Libraries**:
**Tensorflow**:2.19
**tensorflow.keras** – ResNet50 for feature extraction
**pandas, numpy** – Data handling
**scikit-learn** – Cosine similarity
**opencv, PIL** – Image processing
**streamlit_authenticator** – Authentication
## 🧪 How It Works
**User Login**: Secure login using username and password.
**Image Upload**: Upload an image of a fashion item.
**Feature Extraction**: Extract visual features using a pre-trained CNN (ResNet50).
**Similarity Matching**: Use cosine similarity to find the closest matches.
**Recent Uploads**: Display thumbnails of the user's recent uploads.
**Recommendations**: Show the top visually similar fashion items from the dataset.
# 📄 License
This project is licensed under the MIT License.
Feel free to use, adapt, and build upon it.



