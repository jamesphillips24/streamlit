import streamlit as st
import numpy as np
import keras
from PIL import Image

model = keras.models.load_model('my_model.keras')

# Set the title of the app
st.title("Skin Mark Detection AI")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image as needed (resize, normalize, etc.)
    image = image.resize((224, 224))  # Adjust size to fit your model
    image_array = np.array(image) / 255.0  # Normalize if needed
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)  # Adjust based on your output

    # Display the prediction
    st.write(f"Predicted Class: {predicted_class[0]}")
