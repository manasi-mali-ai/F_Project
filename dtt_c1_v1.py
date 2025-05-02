import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from transformers import pipeline

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# CSV Summary Function (Structured + Friendly in one)
def generate_summary_from_dataframe(df):
    num_rows, num_cols = df.shape
    column_names = df.columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    insights = []

    # Analyze numeric columns
    for col in numeric_cols:
        col_mean = df[col].mean()
        col_sum = df[col].sum()
        col_max = df[col].max()
        col_min = df[col].min()
        insights.append(
            f"'{col}': avg={col_mean:.2f}, total={col_sum:.2f}, max={col_max}, min={col_min}."
        )

    # Analyze categorical columns
    for col in df.select_dtypes(include='object').columns.tolist():
        try:
            top_value = df[col].value_counts().idxmax()
            count = df[col].value_counts().max()
            insights.append(
                f"In column '{col}', the most frequent value is '{top_value}' appearing {count} times."
            )
        except Exception:
            continue

    # Construct smart prompt
    prompt = f"""
Dataset contains {num_rows} rows and {num_cols} columns: {', '.join(column_names)}.

Key insights:
{chr(10).join(insights)}

Now write a single, easy-to-understand paragraph in human-friendly language that summarizes the overall content of this dataset.
It should describe the type of data, patterns, statistics, and what a non-technical person can understand from this.
Length: at least 6 lines.
"""

    result = summarizer(prompt, max_length=350, min_length=120, do_sample=False)[0]['summary_text']
    return result

# Process PDF
def process_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Process image
def process_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

# General text summarization
def summarize_text(text):
    prompt = f"Summarize the following document or image in a simple and informative paragraph:\n{text}"
    return summarizer(prompt, max_length=350, min_length=120, do_sample=False)[0]['summary_text']

# Streamlit UI
st.set_page_config(page_title="AI Data Summary App", layout="centered")
st.title("📊 Smart Data Summarizer")
st.markdown("Upload your **CSV**, **JSON**, **PDF**, or **image (graph, chart, heatmap)** to get a detailed and easy-to-understand summary.")

col1, col2 = st.columns(2)

with col1:
    data_file = st.file_uploader("📄 Upload CSV / JSON / PDF", type=["csv", "json", "pdf"])

with col2:
    image_file = st.file_uploader("🖼️ Upload PNG / JPG / JPEG (Image of graph/chart)", type=["png", "jpg", "jpeg"])

if st.button("🔍 Generate Summary"):
    if data_file:
        try:
            if data_file.name.endswith('.csv'):
                df = pd.read_csv(data_file)
                summary = generate_summary_from_dataframe(df)
            elif data_file.name.endswith('.json'):
                df = pd.read_json(data_file)
                summary = generate_summary_from_dataframe(df)
            elif data_file.name.endswith('.pdf'):
                text = process_pdf(data_file)
                summary = summarize_text(text)
        except Exception as e:
            st.error(f"Error processing file: {e}")
            summary = None

    elif image_file:
        try:
            extracted_text = process_image(image_file)
            summary = summarize_text(extracted_text)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            summary = None

    else:
        st.warning("Please upload a file to summarize.")
        summary = None

    if summary:
        st.subheader("📄 Summary")
        st.write(summary)
