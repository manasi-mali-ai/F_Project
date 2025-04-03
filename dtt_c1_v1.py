import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import io
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# NLG (Natural Language Generation) utility function for CSV Summary
def generate_csv_summary(df):
    # Basic statistical summary of numerical columns
    summary = df.describe().transpose()
    
    # Generating a text-based summary
    summary_text = f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n\n"
    summary_text += "Statistical Summary:\n"
    summary_text += summary.to_string()
    
    return summary_text

# Function to generate image captions (using a placeholder for simplicity)
def generate_image_caption(image):
    # In an actual implementation, we would use an image captioning model
    # For simplicity, we're using a placeholder text here.
    return "This is a placeholder caption for the uploaded image."

# Streamlit page configuration
st.set_page_config(page_title="NLG Data to Text", page_icon=":bar_chart:", layout="wide")

# App Title
st.title("NLG Data to Text")

# Sidebar - Upload section
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
uploaded_image = st.sidebar.file_uploader("Upload a Graph/Image", type=["png", "jpg", "jpeg", "svg"])

# Display file upload options
if uploaded_file:
    st.sidebar.success("File uploaded successfully!")

if uploaded_image:
    st.sidebar.success("Image uploaded successfully!")

# Main Body
st.header("Generate Report")

# If user uploads CSV
if uploaded_file:
    st.subheader("CSV Summary")
    
    # Read the CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the dataframe
    st.write("### Data Overview")
    st.write(df.head())
    
    # Generate and display summary
    st.write("### Data Summary")
    summary_text = generate_csv_summary(df)
    st.text_area("Summary Report", summary_text, height=300)

# If user uploads Image
elif uploaded_image:
    st.subheader("Image Captioning")
    
    # Display the uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Generate and display the caption
    caption = generate_image_caption(img)
    st.write("### Image Caption:")
    st.write(caption)

else:
    st.write("Please upload a CSV file or an image to generate a report.")
