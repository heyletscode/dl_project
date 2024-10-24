import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import sys

# Configure stdout to handle UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set page configuration
st.set_page_config(page_title='Image Segmentation App', layout='wide')

# Define the target image dimensions
TARGET_WIDTH = 256
TARGET_HEIGHT = 256

# Load the model (cache to improve performance)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('unet_segmentation.keras')
    return model

model = load_model()

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert('RGB')
    image = image.resize((TARGET_WIDTH, TARGET_HEIGHT))
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # Normalize pixel values
    return image_array

# Title and description
st.title('Image Segmentation Prediction App')
st.write('Upload an image to perform segmentation prediction using the pre-trained U-Net model.')

# File uploader
uploaded_file = st.file_uploader('Choose an image file', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Preprocess the uploaded image
    input_image = preprocess_image(uploaded_file)

    # Expand dimensions to match model input
    input_image_expanded = np.expand_dims(input_image, axis=0)

    # Make prediction
    with st.spinner('Performing prediction...'):
        pred_mask = model.predict(input_image_expanded)

    # Post-process the predicted mask
    pred_mask = pred_mask[0]
    pred_mask = np.squeeze(pred_mask)

    # Display the original image and predicted mask
    st.subheader('Results')
    col1, col2 = st.columns(2)

    with col1:
        st.image(input_image, caption='Original Image', use_column_width=True)

    with col2:
        st.image(pred_mask, caption='Predicted Mask', use_column_width=True)
else:
    st.info('Please upload an image file to get started.')
