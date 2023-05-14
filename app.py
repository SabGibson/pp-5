# main.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import io

# load your model
model = tf.keras.models.load_model('models\cherry-picker-v1.h5')

def upload_images():
    st.title("Upload Images")
    uploaded_files = st.file_uploader("Choose images...", type="jpg", accept_multiple_files=True)
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        images.append(image)
    return uploaded_files, images

def show_predictions(files, images):
    st.title("Predictions")
    for i, file in enumerate(files):
        if file is not None:
            # Preprocess the image here as per your model's requirements
            img_array = np.array(images[i].resize((224, 224))) / 255.0  # example for model that takes 224x224x3 image normalized to [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # model expects images in batch
            predictions = model.predict(img_array)
            st.write("Image ", i)
            st.write("Class probabilities:", predictions)  # Display raw probabilities
            st.write("Predicted Class:", np.argmax(predictions))  # Display predicted class

# create a dictionary of pages
pages = {
    "Upload Images": upload_images,
    "Show Predictions": show_predictions,
}

# create a sidebar for navigation
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(pages.keys()))

# run the function for the selected page
files, images = pages[selection]()

if selection == "Show Predictions" and files is not None:
    show_predictions(files, images)
