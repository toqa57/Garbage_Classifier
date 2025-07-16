import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

# Set page config
st.set_page_config(page_title="Garbage Classifier", layout="wide")


# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/best_model.h5')
        # Warm-up the model
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()


# Load model
model = load_model()

# Get class names
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Default
try:
    CLASS_NAMES = sorted(os.listdir('data/train'))
except FileNotFoundError:
    st.warning("⚠️ Using default class names")

# Streamlit UI
st.title("♻️ Smart Waste Classifier")
st.markdown("Upload an image to classify waste materials")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"]
)

col1, col2 = st.columns(2)

if uploaded_file:
    try:
        # Display image
        img = Image.open(uploaded_file)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        with col1:
            st.image(img, caption="Uploaded Image", width=300)

        # Preprocess and predict
        if st.button("🔍 Classify Waste", type="primary"):
            start_time = time.time()

            # Resize and normalize
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array, verbose=0)[0]
            pred_time = time.time() - start_time

            # Get results
            top_class = CLASS_NAMES[np.argmax(preds)]
            confidence = np.max(preds) * 100

            # Show results
            with col2:
                st.success(f"**Predicted:** {top_class}")
                st.metric("Confidence", f"{confidence:.2f}%")
                st.write(f"Time taken: {pred_time:.2f}s")

                # Detailed probabilities
                with st.expander("📊 See all class probabilities"):
                    for i, cls in enumerate(CLASS_NAMES):
                        st.progress(
                            float(preds[i]),
                            text=f"{cls}: {preds[i] * 100:.2f}%"
                        )
                    st.bar_chart(dict(zip(CLASS_NAMES, preds)))

    except Exception as e:
        st.error(f"Error: {str(e)}")

st.caption("♻️ Sustainable Waste AI | v1.0")