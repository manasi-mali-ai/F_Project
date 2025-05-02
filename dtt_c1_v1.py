import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from transformers import pipeline

# Load summarizer model
summarizer = pipeline("summarization", model="t5-small")

# Summarization functions
def generate_technical_summary(text):
    return summarizer(text, max_length=80, min_length=20, do_sample=False)[0]['summary_text']

def generate_friendly_summary(text):
    prompt = f"Explain this to a non-technical person: {text}"
    return summarizer(prompt, max_length=80, min_length=20, do_sample=False)[0]['summary_text']

# File processing functions
def process_csv(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)

def process_json(file):
    df = pd.read_json(file)
    return df.to_string(index=False)

def process_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

# Streamlit UI
st.title("📊 Multimodal Data Summary AI System")
st.markdown("Upload **CSV / JSON / PDF** or **Graph/Image (PNG)** to get technical and friendly summaries.")

col1, col2 = st.columns(2)

with col1:
    data_file = st.file_uploader("Upload CSV / JSON / PDF", type=["csv", "json", "pdf"])

with col2:
    image_file = st.file_uploader("Upload PNG Image (e.g., Graph)", type=["png", "jpg", "jpeg"])

if st.button("Generate Summary"):

    input_text = ""

    # Handle structured files
    if data_file:
        if data_file.name.endswith('.csv'):
            input_text = process_csv(data_file)
        elif data_file.name.endswith('.json'):
            input_text = process_json(data_file)
        elif data_file.name.endswith('.pdf'):
            input_text = process_pdf(data_file)
    
    # Handle images
    elif image_file:
        input_text = process_image(image_file)

    if input_text:
        with st.spinner("Generating summaries..."):
            tech_summary = generate_technical_summary(input_text)
            friendly_summary = generate_friendly_summary(input_text)

        st.subheader("📌 Technical Summary")
        st.write(tech_summary)

        st.subheader("🧑‍🤝‍🧑 Friendly Summary")
        st.write(friendly_summary)
    else:
        st.warning("Please upload a file to summarize.")
