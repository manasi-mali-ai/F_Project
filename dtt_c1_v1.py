import streamlit as st
import pandas as pd
import torch
import cv2
import pytesseract
from transformers import T5ForConditionalGeneration, T5Tokenizer
st.title("Automated Reporting System")

# File upload options
option = st.radio("Choose Input Type:", ["CSV File", "Visual Data (Chart/Image)"])

uploaded_file = st.file_uploader("Upload File", type=["csv", "png", "jpg", "jpeg"])

if uploaded_file:
    if option == "CSV File":
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded CSV:")
        st.dataframe(df)

        if st.button("Generate Summary"):
            summary = generate_text_from_csv(df)  # Function to process CSV
            st.write("Generated Summary:")
            st.text(summary)

    elif option == "Visual Data (Chart/Image)":
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Extract Insights"):
            insights = generate_text_from_image(uploaded_file)  # Function to process images
            st.write("Generated Report:")
            st.text(insights)
def generate_text_from_csv(df):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Convert data into text format
    text_input = "summarize: " + df.to_string()
    inputs = tokenizer.encode(text_input, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=150, num_return_sequences=1)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def generate_text_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OCR to extract text
    extracted_text = pytesseract.image_to_string(gray)

    # Convert extracted text into a summary
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    text_input = "summarize: " + extracted_text
    inputs = tokenizer.encode(text_input, return_tensors="pt", max_length=512, truncation=True)

    summary_ids = model.generate(inputs, max_length=150, num_return_sequences=1)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def refine_with_neural_planning(text):
    """Improve coherence and factual accuracy using predefined patterns."""
    structured_text = text.replace("Firstly,", "Step 1:") \
                          .replace("Secondly,", "Step 2:") \
                          .replace("Finally,", "Conclusion:")
    return structured_text
