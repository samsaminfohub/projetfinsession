import streamlit as st
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

BACKEND_URL = "http://backend:8000"

st.title("MNIST Digit Classification")
st.write("Upload an image of a handwritten digit (0-9) for classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        files = {"image": uploaded_file.getvalue()}
        response = requests.post(f"{BACKEND_URL}/predict/", files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Digit: {result['predicted_digit']}")
            st.write(f"Confidence: {result['confidence']:.2%}")
            
            # Show prediction probabilities
            fig, ax = plt.subplots()
            digits = list(range(10))
            ax.bar(digits, [0.1]*10, color='lightgray')
            ax.bar(result['predicted_digit'], result['confidence'], color='blue')
            ax.set_xticks(digits)
            ax.set_ylim(0, 1)
            ax.set_title('Prediction Probabilities')
            ax.set_xlabel('Digit')
            ax.set_ylabel('Probability')
            st.pyplot(fig)
        else:
            st.error(f"Error: {response.text}")

# Show recent predictions
if st.checkbox("Show recent predictions"):
    response = requests.get(f"{BACKEND_URL}/predictions/")
    if response.status_code == 200:
        predictions = response.json()
        st.write("Recent Predictions:")
        for pred in predictions:
            st.write(f"Digit: {pred['predicted_digit']}, Confidence: {pred['confidence']:.2%}, Time: {pred['created_at']}")
    else:
        st.error(f"Error fetching predictions: {response.text}")

# Show model info
if st.checkbox("Show model information"):
    response = requests.get(f"{BACKEND_URL}/model-info/")
    if response.status_code == 200:
        model_info = response.json()
        st.write("Model Information:")
        st.write(f"Name: {model_info['name']}")
        st.write(f"Version: {model_info['latest_version']}")
        st.write(f"Description: {model_info['description']}")
    else:
        st.error(f"Error fetching model info: {response.text}")