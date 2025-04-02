import streamlit as st
import pandas as pd
import cv2
import pytesseract
from transformers import pipeline
from PIL import Image
import numpy as np

# Load text generation model (GPT-2)
text_generator = pipeline("text-generation", model="gpt2")

# Function to summarize CSV data
def summarize_csv(file):
    df = pd.read_csv(file)
    summary = df.describe().to_string()
    generated_text = text_generator(summary, max_length=200)[0]['generated_text']
    return generated_text

# Function to analyze and summarize images
def analyze_image(image):
    image = Image.open(image)
    img_array = np.array(image)
    text = pytesseract.image_to_string(img_array)
    generated_text = text_generator(text, max_length=200)[0]['generated_text']
    return generated_text

# Streamlit UI
st.title("Automated Data & Visual Report Generator")

option = st.radio("Choose Data Type", ["CSV File", "Visual Data (Image/Chart)"])

if option == "CSV File":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        st.write("### Summary:")
        summary = summarize_csv(uploaded_file)
        st.text_area("Generated Summary", summary, height=200)

elif option == "Visual Data (Image/Chart)":
    uploaded_image = st.file_uploader("Upload Image or Chart", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write("### Extracted Insights:")
        insights = analyze_image(uploaded_image)
        st.text_area("Generated Report", insights, height=200)
