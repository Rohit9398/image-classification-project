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
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("### 👁️ AI Vision (28x28):")
            display_img = (processed_img * 255).astype(np.uint8) if processed_img.max() <= 1.0 else processed_img.astype(np.uint8)
            display_img = cv2.resize(display_img, (150, 150), interpolation=cv2.INTER_NEAREST)
            st.image(display_img, caption="What AI sees", width=150)
            
        with col2:
            st.write("### 💡 Is this prediction correct?")
            file_key = getattr(uploaded_file, 'name', 'camera_img') if uploaded_file else 'camera_img'
            feedback = st.radio(
                "Feedback:",
                ["Yes, correct! 👍", "No, correct prediction ✏️"],
                key=f"fb_radio_{file_key}"
            )
            
            if feedback == "Yes, correct! 👍":
                st.balloons()
            else:
                correct_class = st.selectbox(
                    "Select correct label:",
                    CLASS_NAMES,
                    index=int(prediction[0]),
                    key=f"fb_select_{file_key}"
                )
                
                if st.button("💾 Save Feedback & Update Model", key=f"fb_btn_{file_key}"):
                    with st.spinner("Teaching the AI your correct label..."):
                        correct_idx = CLASS_NAMES.index(correct_class)
                        
                        # 1. Save feedback image for dataset collection
                        fb_dir = os.path.join(OUTPUT_DIR, 'feedback_data', correct_class)
                        os.makedirs(fb_dir, exist_ok=True)
                        import time
                        ts = int(time.time())
                        input_image.save(os.path.join(fb_dir, f"feedback_{ts}.png"))
                        
                        # 2. Store & Boost feedback features
                        fb_feat_path = os.path.join(OUTPUT_DIR, 'feedback_features.pkl')
                        if os.path.exists(fb_feat_path):
                            fb_data = joblib.load(fb_feat_path)
                            X_fb = np.vstack([fb_data['X'], combined_scaled])
                            y_fb = np.append(fb_data['y'], correct_idx)
                        else:
                            X_fb = combined_scaled
                            y_fb = np.array([correct_idx])
                            
                        joblib.dump({'X': X_fb, 'y': y_fb}, fb_feat_path)
                        
                        # 3. Update model with weight boosting
                        # Mix feedback samples with base sample dataset for balanced multi-class retraining
                        X_boosted = np.repeat(X_fb, 25, axis=0)
                        y_boosted = np.repeat(y_fb, 25, axis=0)
                        
                        base_sample_path = os.path.join(OUTPUT_DIR, 'base_samples.pkl')
                        if os.path.exists(base_sample_path):
                            base_data = joblib.load(base_sample_path)
                            X_train_mix = np.vstack([base_data['X'], X_boosted])
                            y_train_mix = np.append(base_data['y'], y_boosted)
                        else:
                            X_train_mix = X_boosted
                            y_train_mix = y_boosted
                        
                        # Retrain model on feedback boosted samples
                        try:
                            if hasattr(model, 'fit'):
                                model.fit(X_train_mix, y_train_mix)
                                joblib.dump(model, MODEL_PATH)
                                st.cache_resource.clear()
                                st.success(f"✅ Success! AI has learned that this image is a **{correct_class}**.")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error updating model: {e}")

