import streamlit as st
import pandas as pd
from PIL import Image

# Function to generate a structured, human-like summary of the dataset
def generate_csv_summary(df):
    num_rows, num_cols = df.shape
    summary_text = f"📊 **Dataset Overview:**\n\n"
    summary_text += f"This dataset contains **{num_rows} rows** and **{num_cols} columns**.\n"

    for col in df.columns:
        unique_values = df[col].nunique()
        if df[col].dtype == 'object':  # Categorical Data
            example_values = df[col].dropna().unique()[:3]  # Show first 3 unique values
            summary_text += f"\n🔹 **{col}**: This column contains **categorical** data with {unique_values} unique values. Examples: {', '.join(map(str, example_values))}.\n"
        else:  # Numerical Data
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            summary_text += f"\n🔹 **{col}**: Numerical column with values ranging from **{min_val} to {max_val}**, and an average of **{mean_val:.2f}**.\n"

    return summary_text

# Function to generate a structured placeholder caption for uploaded images
def generate_image_caption():
    return """
    🖼 **Image Overview:**  
    - The uploaded image appears to be a **chart or visualization**.  
    - It likely represents **data trends, comparisons, or patterns**.  
    - If it is a **bar chart**, it may show category-wise performance.  
    - If it is a **line chart**, it might indicate trends over time.  
    """

# Streamlit Page Config
st.set_page_config(page_title="NLG Data to Text", page_icon="📊", layout="wide")

# Title
st.title("📊 NLG Data to Text")

# Sidebar Upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
uploaded_image = st.sidebar.file_uploader("Upload a Graph/Image", type=["png", "jpg", "jpeg", "svg"])

# If user uploads CSV
if uploaded_file:
    st.subheader("📑 CSV Summary")
    
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())
    
    summary_text = generate_csv_summary(df)
    st.write("### 🔍 Generated Summary:")
    st.markdown(summary_text)

# If user uploads Image
elif uploaded_image:
    st.subheader("🖼 Image Analysis")
    
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    caption = generate_image_caption()
    st.write("### 🔍 Image Interpretation:")
    st.markdown(caption)

else:
    st.write("📢 Please upload a CSV file or an image to generate a report.")
