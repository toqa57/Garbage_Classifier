# Garbage_Classifier

# ♻️ Smart Garbage Classification System

A deep learning-based image classification system that identifies types of garbage (e.g., plastic, metal, glass, etc.) using a convolutional neural network. This project uses TensorFlow/Keras for training and Gradio for building a simple web interface. You can train the model locally or run the Gradio app on Hugging Face Spaces.

---

## 🧠 Model Architectures Supported
- Custom CNN
- MobileNetV2 (default)
- VGG16
- ResNet50

---

## 📁 Dataset Structure

Your dataset directory should follow this format:

data/original data/
├── battery/
├── biological/
├── cardboard/
├── clothes/
├── glass/
├── metal/
├── paper/
├── plastic/
├── shoes/
├── trash/

## 🚀 Features

- Optimized image data pipeline with augmentation
- Supports training with multiple CNN backbones
- Automatic class extraction and labeling
- Training history visualization (accuracy and loss)
- Confusion matrix and classification report
- Interactive Gradio interface for image prediction

##💡 Future Improvements
Add more classes or fine-tune on larger datasets

Export to ONNX or TFLite for mobile deployment

Use EfficientNet or ViT for improved accuracy

