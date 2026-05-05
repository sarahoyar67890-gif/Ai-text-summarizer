import streamlit as st
from transformers import pipeline

# =========================
# LOAD MODEL (cached)
# =========================
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        framework="pt"
    )

model = load_model()

# =========================
# UI DESIGN
# =========================
st.set_page_config(page_title="AI Summarizer", page_icon="🧠")

st.title("🧠 AI Text Summarizer")
st.write("Paste any article and get a clean AI-generated summary")

# Input box
text = st.text_area("Enter your text here")

# =========================
# SUMMARIZATION FUNCTION
# =========================
def summarize(text):
    if not text.strip():
        return "Please enter valid text."

    # limit input size
    text = text[:1200]

    result = model(
        text,
        max_length=80,
        min_length=20,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    return result[0]["summary_text"]

# =========================
# BUTTON ACTION
# =========================
if st.button("Generate Summary"):
    with st.spinner("Summarizing..."):
        output = summarize(text)

    st.success("Summary Generated 🎉")
    st.write("### 📌 Summary:")
    st.write(output)