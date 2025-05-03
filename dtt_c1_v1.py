# ✅ Install required packages (first time only, especially for Colab or Streamlit Cloud)
# !pip install streamlit pandas pdfplumber easyocr transformers

import streamlit as st
import pandas as pd
import json
import pdfplumber
import easyocr
from PIL import Image
from transformers import pipeline

# ✅ Load OCR and summarizer
reader = easyocr.Reader(['en'])
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.set_page_config(page_title="Smart Summary Generator", layout="wide")
st.title("📊 Smart Data & Image Summary Generator")
st.markdown("Upload CSV, JSON, PDF or PNG/JPG to get a **detailed, easy-to-understand** summary (100+ words).")

uploaded_file = st.file_uploader("Upload CSV / JSON / PDF / PNG / JPG", type=['csv', 'json', 'pdf', 'png', 'jpg', 'jpeg'])

def read_data(file):
    if file.type == "text/csv":
        df = pd.read_csv(file)
        return df

    elif file.type == "application/json":
        data = json.load(file)
        df = pd.json_normalize(data)
        return df

    elif file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text

    elif file.type in ["image/png", "image/jpeg", "image/jpg"]:
        image = Image.open(file)
        results = reader.readtext(image, detail=0)
        return " ".join(results)

    return None

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
                f"The column '{col}' ranges from {min_val} to {max_val}, with an average of {mean:.2f} and a total of {total:.2f}."
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
                   f"Columns include: {', '.join(col_names[:6])}. " \
                   f"{' '.join(insights)}"

    result = summarizer(insight_text, max_length=350, min_length=150, do_sample=False)[0]['summary_text']
    return result

def summarize_text(text):
    prompt = (
        f"Summarize this text in a single paragraph (100+ words) in simple, clear language for a general audience. "
        f"\n\n{text[:3000]}"
    )
    result = summarizer(prompt, max_length=350, min_length=150, do_sample=False)[0]['summary_text']
    return result

# ✅ App UI logic
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
