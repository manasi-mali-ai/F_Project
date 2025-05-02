import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from transformers import pipeline

# Load the Hugging Face summarizer model
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
                f"For '{col}', the average is {mean:.2f}, total is {total:.2f}, maximum is {max_val}, and minimum is {min_val}."
            )
        except:
            continue

    for col in df.select_dtypes(include='object').columns.tolist():
        try:
            most_common = df[col].value_counts().idxmax()
            count = df[col].value_counts().max()
            insights.append(
                f"The column '{col}' most commonly contains '{most_common}', appearing {count} times."
            )
        except:
            continue

    prompt = f"""
This dataset has {num_rows} rows and {num_cols} columns. Key columns are: {', '.join(col_names)}.
Here are some basic insights:\n{chr(10).join(insights)}\n
Now write a **detailed summary (minimum 100 words)** in simple, clear language suitable for someone without a technical background.
"""
    result = summarizer(prompt, max_length=350, min_length=150, do_sample=False)[0]['summary_text']
    return result

# Summarize extracted text from PDFs or images
def summarize_text(text):
    prompt = f"""
Summarize this content in a **detailed, user-friendly paragraph** using **non-technical language**.
Make sure the paragraph is at least 100 words to help anyone understand easily:\n{text}
"""
    result = summarizer(prompt, max_length=350, min_length=150, do_sample=False)[0]['summary_text']
    return result

# Streamlit UI
st.set_page_config(page_title="🧠 Data Summarizer", layout="centered")
st.title("📊 AI-Powered Summary Generator")
st.markdown("Upload a CSV, JSON, PDF, or Image file and get a long, human-readable summary in simple words.")

uploaded_file = st.file_uploader("📤 Upload CSV / JSON / PDF / Image", type=["csv", "json", "pdf", "png", "jpg", "jpeg"])

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
            st.error("Unsupported file type.")
            summary = ""

        if summary:
            st.subheader("📝 Detailed Summary")
            st.write(summary)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
