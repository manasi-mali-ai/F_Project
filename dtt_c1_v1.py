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

# Function to generate text summary for CSV data
def generate_text_summary(data):
    """Generate textual summary from structured data."""
    summary = f"This dataset contains {data.shape[0]} rows and {data.shape[1]} columns."
    summary += "\nKey Statistics:\n"
    summary += str(data.describe().to_dict())  # Simple statistical summary
    return summary

# Function to generate captions for an image
def generate_image_summary(image):
    """Generate an easy-to-understand description of the image."""
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def main():
    st.title("📊 AI-Powered Data & Visual Intelligence System")
    st.write("Upload structured data (CSV) or visual data (images) to generate automated insights.")

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
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            st.write("### Generated Summary:")
            summary = generate_image_summary(image)
            st.success(summary)

if __name__ == "__main__":
    main()
