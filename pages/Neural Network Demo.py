import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Title and instructions
st.title("การจำแนกประเภทรูปถ่ายทางดาวเทียม")
st.write("อัปโหลดไฟล์ภาพ และโมเดลจะทำการจำแนกว่าเป็น กลุ่มเมฆ น้ำ พื้นที่สีเขียว หรือทะเลทราย")

# File uploader for images
uploaded_file = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg", "jpeg", "png"])

# Function to load the model (caching for performance)
@st.cache_resource
def load_classification_model():
    model_path = "exported_model/rf/NN/NN_save.keras"  # Replace with your actual model file path
    model = tf.keras.models.load_model(model_path)
    return model

model = load_classification_model()

# Define class names (update if your order differs)
class_names = ["cloudy", "desert", "green_area", "water"]

def preprocess_image(image_file):
    """
    Preprocess the uploaded image:
    - Resize to 224x224,
    - Convert to array,
    - Normalize pixel values,
    - Expand dimensions for batch size.
    """
    img = load_img(image_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

if uploaded_file is not None:
    # Preprocess image and get the original image for display
    original_img, processed_img = preprocess_image(uploaded_file)
    
    # Get prediction from the model
    prediction = model.predict(processed_img)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # Display the uploaded image and prediction results using use_container_width
    st.image(original_img, caption="Uploaded Image", use_container_width=True)
    st.write(f"**Prediction:** {predicted_class} (Confidence: {confidence:.2f}%)")
