import streamlit as st
import tempfile
import os
from ingestion_pipeline import ingest_documents_to_qdrant
from utils.ui_components import init_page

user_info = init_page("Document Ingestion")

st.set_page_config(page_title="Document Ingestion")

st.title("📄 Document Ingestion")
st.write("Upload PDF laws or company documents to add them to the AI knowledge base.")

with st.sidebar:
    if st.button("⬅ Back to Dashboard"):
        st.switch_page("main.py") # or your main dashboard file

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    if st.button("Start Ingestion", type="primary"):
        with st.spinner("Processing PDF and creating embeddings..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                ingest_documents_to_qdrant(tmp_path)
                st.success("Success! The knowledge base has been updated.")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)