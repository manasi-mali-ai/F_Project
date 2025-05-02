import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDFs
from PIL import Image
import pytesseract
from transformers import pipeline

# Load the summarization model from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize DataFrame content
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
                f"'{col}' most commonly contains '{most_common}' ({count} times)."
            )
        except:
            continue

    prompt = f"""
This dataset has {num_rows} rows and {num_cols} columns, including: {', '.join(col_names)}.
Technical insights:
{chr(10).join(insights)}

Now write a single, detailed, user-friendly paragraph (minimum 100 words), combining technical insights with non-technical explanation.
Use simple language for people from non-technical backgrounds.
"""
    result = summarizer(prompt, max_length=350, min_length=150, do_sample=False)[0]['summary_text']
    return result

# Function to summarize plain text (PDF/image)
def summarize_text(text):
    prompt = f"Summarize this content in simple, detailed language for non-technical people (minimum 100 words):\n{text}"
    result = summarizer(prompt, max_length=350, min_length=150, do_sample=False)[0]['summary_text']
    return result

# Streamlit UI
st.set_page_config(page_title="📊 Smart Data Summarizer", layout="centered")
st.title("📈 AI-Powered Data Summarizer")
st.markdown("Upload your **CSV**, **JSON**, **PDF**, or **image (chart, graph)** to get a detailed and easy-to-understand paragraph summary.")

uploaded_file = st.file_uploader("📤 Upload CSV / JSON / PDF / Image", type=["csv", "json", "pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            summary = summarize_dataframe(df)
            st.subheader("📝 Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    elif uploaded_file.name.endswith(".json"):
        try:
            df = pd.read_json(uploaded_file)
            st.dataframe(df.head())
            summary = summarize_dataframe(df)
            st.subheader("📝 Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Error reading JSON: {e}")

    elif uploaded_file.name.endswith(".pdf"):
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
            summary = summarize_text(text)
            st.subheader("📝 Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    elif uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
        try:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
            summary = summarize_text(text)
            st.subheader("📝 Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Error processing image: {e}")

    else:
        st.warning("❌ Unsupported file type.")
