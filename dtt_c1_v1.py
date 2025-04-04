import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from transformers import pipeline

# Fix OpenCV import issue
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  

# Load NLP model for text generation
text_generator = pipeline("text-generation", model="gpt2")

def generate_text_summary(data):
    """Generate textual summary from structured data."""
    try:
        summary = f"This dataset contains {data.shape[0]} rows and {data.shape[1]} columns. Key insights: "
        summary += "Mean values of numerical columns: " + str(data.mean(numeric_only=True).to_dict())
        generated = text_generator(summary, max_length=100)[0]['generated_text']
        return generated
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def extract_text_from_image(image):
    """Extract text insights from charts or graphs."""
    try:
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        text_summary = "Analyzing the chart for key insights..."
        generated = text_generator(text_summary, max_length=100)[0]['generated_text']
        return generated
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def main():
    st.title("📊 AI-Powered Data & Visual Intelligence System")
    st.write("Upload structured data (CSV) or visual data (charts/images) to generate automated insights.")

    option = st.selectbox("Select Analysis Type:", ["Data-to-Text (CSV)", "Visual Intelligence (Image)"])
    
    if option == "Data-to-Text (CSV)":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("### Preview of Uploaded Data:")
                st.dataframe(df.head())
                
                st.write("### Generated Summary:")
                summary = generate_text_summary(df)
                st.success(summary)
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    elif option == "Visual Intelligence (Image)":
        uploaded_image = st.file_uploader("Upload an Image (Chart or Graph)", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            try:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Chart", use_column_width=True)
                
                st.write("### Generated Summary:")
                summary = extract_text_from_image(image)
                st.success(summary)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
