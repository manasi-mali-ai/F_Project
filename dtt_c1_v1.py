import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Function to generate text summary
def generate_text_summary(data):
    """Generate textual summary from structured data."""
    summary = f"This dataset contains {data.shape[0]} rows and {data.shape[1]} columns."
    summary += "\nKey Statistics:\n"
    summary += str(data.describe().to_dict())  # Simple statistical summary
    return summary

# Function to handle image analysis (Dummy function for now)
def extract_text_from_image(image):
    """Placeholder for image analysis."""
    return "Image uploaded. Text analysis is not implemented in this version."

def main():
    st.title("📊 AI-Powered Data & Visual Intelligence System")
    st.write("Upload structured data (CSV) or visual data (charts/images) to generate automated insights.")

    option = st.selectbox("Select Analysis Type:", ["Data-to-Text (CSV)", "Visual Intelligence (Image)"])
    
    if option == "Data-to-Text (CSV)":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### Preview of Uploaded Data:")
            st.dataframe(df.head())

            st.write("### Generated Summary:")
            summary = generate_text_summary(df)
            st.success(summary)
    
    elif option == "Visual Intelligence (Image)":
        uploaded_image = st.file_uploader("Upload an Image (Chart or Graph)", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Chart", use_column_width=True)

            st.write("### Generated Summary:")
            summary = extract_text_from_image(image)
            st.success(summary)

if __name__ == "__main__":
    main()
