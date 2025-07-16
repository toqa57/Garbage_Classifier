import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
model = tf.keras.models.load_model('models/best_model.h5')

# Get class names
try:
    CLASS_NAMES = sorted(os.listdir('data/train'))
except FileNotFoundError:
    CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Fallback


def predict_image(img):
    # Convert numpy array to PIL Image if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))

    # Preprocess
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array, verbose=0)[0]
    return {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}


# Create interface WITHOUT examples (or create the examples directory)
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(label="Upload Garbage Image"),
    outputs=gr.Label(label="Prediction Probabilities"),
    title="♻️ Garbage Classification",
    description="Upload an image to classify the type of waste",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share=True)