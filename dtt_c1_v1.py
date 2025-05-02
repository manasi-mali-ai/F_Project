import streamlit as st
import pandas as pd
import json
import pytesseract
from PIL import Image
import pdfplumber
from transformers import pipeline
import tempfile
import os

# Set path to tesseract (for Colab/Linux)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Load summarization pipeline (T5 or BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.set_page_config(page_title="Smart Summary Generator", layout="wide")
st.title("📊 Smart Data & Image Summary Generator")
st.markdown("Upload CSV, JSON, PDF or PNG/JPG to get a **detailed, easy-to-understand** summary (100+ words).")

uploaded_file = st.file_uploader("Upload CSV / JSON / PDF / PNG / JPG", type=['csv', 'json', 'pdf', 'png', 'jpg', 'jpeg'])

def read_data(file):
    if file.type == "text/csv":
        df = pd.read_csv(file)
    elif file.type == "application/json":
        data = json.load(file)
        df = pd.json_normalize(data)
    elif file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text
    elif file.type in ["image/png", "image/jpeg", "image/jpg"]:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return text
    else:
        return None
    return df

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
                f"The column '{col}' has values ranging from {min_val} to {max_val}, with an average of {mean:.2f} and a total of {total:.2f}."
            )
        except:
            continue

    for col in df.select_dtypes(include='object').columns.tolist():
        try:
            most_common = df[col].value_counts().idxmax()
            count = df[col].value_counts().max()
            insights.append(
                f"In column '{col}', the most common value is '{most_common}', appearing {count} times."
            )
        except:
            continue

    insight_text = f"This dataset has {num_rows} rows and {num_cols} columns. " \
                   f"Columns include: {', '.join(col_names)}. " \
                   f"{' '.join(insights)}"

    # Pass only the generated insight text to the summarizer
    result = summarizer(insight_text, max_length=350, min_length=150, do_sample=False)[0]['summary_text']
    return result

def summarize_text(text):
    prompt = (
        f"Summarize this text in a single paragraph (100+ words) in simple, clear language for a general audience. "
        f"\n\n{text[:3000]}"
    )
    result = summarizer(prompt, max_length=350, min_length=150, do_sample=False)[0]['summary_text']
    return result

if uploaded_file:
    file_type = uploaded_file.type
    data = read_data(uploaded_file)

    if isinstance(data, pd.DataFrame):
        st.subheader("📘 Summary Output")
        st.write(data.head(3))
        with st.spinner("Generating summary..."):
            summary = summarize_dataframe(data)
            st.markdown(f"### 📝 Summary\n{summary}")
    elif isinstance(data, str):
        st.subheader("📘 Summary Output")
        st.text_area("Extracted Text Preview", value=data[:1000], height=200)
        with st.spinner("Generating summary..."):
            summary = summarize_text(data)
            st.markdown(f"### 📝 Summary\n{summary}")
    else:
        st.error("Unsupported file or failed to extract data.")
