import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from transformers import pipeline

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Summarize structured tabular data
def summarize_dataframe(df):
    num_rows, num_cols = df.shape
    col_names = df.columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    insights = []

    for col in numeric_cols:
        try:
            mean = df[col].mean()
            total = df[col].sum()
            max_val = df[col].max()
            min_val = df[col].min()
            insights.append(
                f"In the column '{col}', values range from {min_val} to {max_val}, with an average of {mean:.2f} and a total sum of {total:.2f}."
            )
        except:
            continue

    for col in df.select_dtypes(include='object').columns.tolist():
        try:
            most_common = df[col].value_counts().idxmax()
            count = df[col].value_counts().max()
            insights.append(
                f"The column '{col}' frequently contains the value '{most_common}', which appears {count} times."
            )
        except:
            continue

    prompt = (
        f"This dataset has {num_rows} rows and {num_cols} columns. "
        f"Some important columns are: {', '.join(col_names)}. "
        f"Here are some insights: {' '.join(insights)} "
        f"Please write a detailed summary (at least 100 words) in simple, clear language for someone without a technical background."
    )

    result = summarizer(prompt, max_length=350, min_length=150, do_sample=False)[0]['summary_text']
    return result

# Summarize extracted text from PDFs or images
def summarize_text(text):
    prompt = (
        f"Please summarize the following information in one long paragraph (at least 100 words). "
        f"The summary should be simple and friendly so that even someone without a technical background can understand it. "
        f"\n\n{text}"
    )
    result = summarizer(prompt, max_length=350, min_length=150, do_sample=False)[0]['summary_text']
    return result

# Streamlit interface
st.set_page_config(page_title="📄 AI Data Summarizer", layout="centered")
st.title("🧠 Smart Data Summary Generator")
st.markdown("Upload a **CSV, JSON, PDF, or Image** and get a **single, detailed summary paragraph** in plain, human-friendly language.")

uploaded_file = st.file_uploader("📤 Upload File", type=["csv", "json", "pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            summary = summarize_dataframe(df)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
            summary = summarize_dataframe(df)
        elif uploaded_file.name.endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
            summary = summarize_text(text)
        elif uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
            summary = summarize_text(text)
        else:
            st.error("Unsupported file format.")
            summary = ""

        if summary:
            st.subheader("📘 Summary Output")
            st.write(summary)

    except Exception as e:
        st.error(f"An error occurred while processing: {e}")
