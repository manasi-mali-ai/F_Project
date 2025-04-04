import streamlit as st
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP Model and Processor (for Image Captioning)
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Function to generate a simple text summary from CSV data
def generate_text_summary(data):
    """Generate an easy-to-understand summary of the dataset."""
    summary = []
    summary.append(f"- The dataset contains **{data.shape[0]} rows** and **{data.shape[1]} columns**.")

    for col in data.select_dtypes(include=["number"]).columns[:5]:  # Limit to 5 numerical columns
        summary.append(f"- The column **'{col}'** has an average (mean) of **{data[col].mean():.2f}**.")
        summary.append(f"- The median value of **'{col}'** is **{data[col].median():.2f}**.")
    
    return "\n".join(summary)

# Function to generate a simple caption for an image
def generate_image_summary(image):
    """Generate a simple and clear caption for the image."""
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    summary = [
        f"- **What is in the image?**: {caption}.",
        "- This is an AI-generated description and may not be 100% accurate.",
    ]
    
    return "\n".join(summary)

def main():
    st.title("📊 AI-Powered Data & Image Insights")
    st.write("Upload a dataset or an image, and AI will generate a simple summary.")

    option = st.selectbox("Select Analysis Type:", ["📊 Data Insights (CSV)", "🖼️ Image Insights"])
    
    if option == "📊 Data Insights (CSV)":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### 📋 Preview of Uploaded Data:")
            st.dataframe(df.head())

            st.write("### 🔍 AI-Generated Summary:")
            summary = generate_text_summary(df)
            st.success(summary)

    elif option == "🖼️ Image Insights":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            st.write("### 📝 AI-Generated Summary:")
            summary = generate_image_summary(image)
            st.success(summary)

if __name__ == "__main__":
    main()
