import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import os

# Import our preprocessing logic from the main script
from image_classifier import preprocess_image, extract_hog_features, CLASS_NAMES

st.set_page_config(page_title="Fashion AI", page_icon="👕", layout="centered")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler.pkl')

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

st.title("👕 Fashion Image Classifier")
st.write("Upload an image of clothing to see what the AI thinks it is!")
st.write("*(Supports: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)*")

model, scaler = load_models()

if model is None:
    st.error("⚠️ Model not found! Please run `python3 image_classifier.py` first to train and save the model.")
else:
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Take Photo"])
    
    input_image = None
    
    with tab1:
        uploaded_file = st.file_uploader("Choose an image (JPG/PNG)...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            
    with tab2:
        camera_file = st.camera_input("Take a picture of clothing")
        if camera_file is not None:
            input_image = Image.open(camera_file)

    if input_image is not None:
        # Display the input image
        st.image(input_image, caption='Input Image', width=300)

        with st.spinner("Analyzing image..."):
            # 1. Convert to grayscale numpy array
            img_array = np.array(input_image.convert('L'))
            
            # 2. Fashion MNIST uses white items on black backgrounds. 
            # If the uploaded image has a white background, we need to invert it.
            # We estimate background color by checking the edges.
            top_edge = img_array[0, :]
            bottom_edge = img_array[-1, :]
            left_edge = img_array[:, 0]
            right_edge = img_array[:, -1]
            edges = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
            edge_mean = np.mean(edges)

            if edge_mean > 127:  # Background is light
                img_array = cv2.bitwise_not(img_array)
                
            # 3. Resize to 28x28 (Fashion MNIST size)
            img_resized = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
            
            # 4. Flatten to pass into our original preprocessing function
            img_flat = img_resized.flatten()
            
            # 5. Apply our exact training pipeline
            processed_img = preprocess_image(img_flat)
            hog_feat = extract_hog_features(processed_img)
            pixel_feat = processed_img.flatten().astype(np.float64) / 255.0
            
            # 6. Combine features and scale
            combined = np.hstack([hog_feat, pixel_feat]).reshape(1, -1)
            combined_scaled = scaler.transform(combined)
            
            # 7. Predict
            prediction = model.predict(combined_scaled)
            predicted_class = CLASS_NAMES[prediction[0]]
            
        st.success(f"## 🎯 Prediction: **{predicted_class}**")
        
        # Display what the AI actually "sees"
        st.write("---")
        st.write("### 👁️ What the AI actually sees:")
        st.write("The AI squishes your image down to a 28x28 grayscale square. If the background isn't completely plain, or if the shape gets distorted, it gets confused!")
        
        # We need to scale the processed image back to 0-255 for display
        display_img = (processed_img * 255).astype(np.uint8) if processed_img.max() <= 1.0 else processed_img.astype(np.uint8)
        display_img = cv2.resize(display_img, (150, 150), interpolation=cv2.INTER_NEAREST)
        st.image(display_img, caption="28x28 Processed Vision", width=150)
        
        st.balloons()
