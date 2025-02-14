import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import requests
from io import BytesIO
import time

# Configure page layout
st.set_page_config(
    page_title="Brain Tumor Detection App",
    page_icon="🧠",
    layout="wide"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 10px;
        text-align: center;
    }
    .sub-header {
        font-size: 20px;
        color: #6B7280;
        margin-bottom: 30px;
        text-align: center;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        color: #6B7280;
        font-size: 14px;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)


GITHUB_USERNAME = "Santhoshkumar099"
REPO_NAME = "Brain_Tumor_Detection"
BRANCH = "main"  

# Sample Images
SAMPLE_IMAGES = {
    "Sample 1 (No Tumor)": f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{REPO_NAME}/{BRANCH}/Sample 1.jpg",
    "Sample 2 (No Tumor)": f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{REPO_NAME}/{BRANCH}/Sample 2.jpg",
    "Sample 3 (Tumor)": f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{REPO_NAME}/{BRANCH}/Sample 3.jpg",
    "Sample 4 (Tumor)": f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{REPO_NAME}/{BRANCH}/Sample 4.jpg",
    "Sample 5 (Tumor)": f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{REPO_NAME}/{BRANCH}/Sample 5.jpg"
}


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("binary_brain_tumor_classifier.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def load_image_from_url(url):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None


def preprocess_image(image, img_size=128):
    image = image.resize((img_size, img_size))  # Resize image
    image = img_to_array(image) / 255.0  # Convert to array & normalize
    image = np.expand_dims(image, axis=0)  # Expand dims for batch
    return image


def predict_tumor(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]  # Get prediction
    
    # Return prediction label and confidence
    label = "Tumor" if prediction > 0.5 else "No Tumor"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, float(confidence)

# ------- Main App Logic --------
st.markdown('<h1 class="main-header">🧠 Brain Tumor Detection AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an MRI scan or choose a sample image to detect the presence of brain tumors</p>', 
            unsafe_allow_html=True)


model = load_model()

if model is None:
    st.warning("Failed to load model. Please check if the model file exists.")
    st.stop()


col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload or Select an MRI Scan")
    
    # File uploader with additional info
    uploaded_file = st.file_uploader(
        "Upload a brain MRI image (JPG, PNG, or JPEG format)",
        type=["jpg", "png", "jpeg"],
        help="Please upload a clear brain MRI scan image for accurate results"
    )
    

    st.markdown("### 🔍 Or choose a sample image")
    selected_sample = st.selectbox(
        "Select a sample MRI image",
        options=list(SAMPLE_IMAGES.keys()),
        index=None,
        help="Choose one of our pre-loaded sample images"
    )
    
    
    if selected_sample:
        with st.spinner("Loading sample image..."):
            sample_image = load_image_from_url(SAMPLE_IMAGES[selected_sample])
            if sample_image:
                st.image(sample_image, caption=selected_sample, width=300)
    
    
    with st.expander("ℹ️ How does this work?"):
        st.write("""
        This app uses a deep learning model trained on thousands of MRI scans to detect the presence of brain tumors.
        The model processes your uploaded image through these steps:
        1. Resizing the image to 128x128 pixels
        2. Normalizing pixel values (0-1)
        3. Passing through a convolutional neural network
        4. Generating a probability score
        
        **Note:** This app is for educational purposes only and should not be used for medical diagnosis.
        Always consult with a qualified healthcare professional.
        """)

with col2:
    st.markdown("### 🔍 Analysis Results")
    
    # Process either the uploaded file or the selected sample
    image = None
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_source = "uploaded"
        except Exception as e:
            st.error(f"Error opening uploaded image: {e}")
    
    elif selected_sample:
        image = load_image_from_url(SAMPLE_IMAGES[selected_sample])
        if image:
            image_source = "sample"
    
    if image:
        # Display image if it hasn't been shown already
        if 'image_source' in locals() and image_source == 'uploaded':
            st.image(image, caption="Uploaded MRI Image", use_container_width=True)
        
        # Add processing animation
        with st.spinner("Analyzing image..."):
            # Simulate processing time
            time.sleep(1.5)
            
            # Predict tumor
            label, confidence = predict_tumor(image, model)
            
            # Determine color and icon based on prediction
            if label == "Tumor":
                color = "#EF4444"  # red
                icon = "⚠️"
            else:
                color = "#10B981"  # green
                icon = "✅"
            
            # Display prediction result with confidence
            st.markdown(
                f"""
                <div class="prediction-box" style="background-color: {color}20;">
                    <h2 style="color: {color}; text-align: center;">
                        {icon} {label} Detected {icon}
                    </h2>
                    <h3 style="text-align: center;">
                        Confidence: {confidence*100:.1f}%
                    </h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Add cautionary note
            st.markdown("""
            <div style="font-size: 14px; margin-top: 20px; padding: 10px; background-color: #FEF3C7; border-radius: 5px;">
                <strong>⚠️ Important:</strong> This is an AI-assisted analysis and should not be used as a substitute for professional medical diagnosis. Please consult with a healthcare provider for proper evaluation.
            </div>
            """, unsafe_allow_html=True)

# Add a footer with disclaimer
st.markdown("""
<div class="footer">
    <hr>
    <p>© 2025 Brain Tumor Detection AI | For Research Purposes Only</p>
    <p>Not intended for clinical use or medical diagnosis</p>
</div>
""", unsafe_allow_html=True)
