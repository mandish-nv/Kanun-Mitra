import streamlit as st
import os
import tempfile

from langchain_community.embeddings import HuggingFaceEmbeddings
# Import the Qdrant configuration and the ingestion function from the service file
from qdrant_service import ingest_documents_to_qdrant, QDRANT_CONFIG

# --- Embedding Model Caching ---
# Use the HuggingFaceEmbeddings wrapper for sentence-transformers/all-MiniLM-L6-v2
@st.cache_resource
def get_dense_embedding_model():
    """Loads and caches the Hugging Face Sentence Transformer model."""
    # Note: Using 'cpu' for wider compatibility
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

## --- Streamlit UI ---

st.title("📚 PDF Ingestion to Qdrant Vector Store")
st.markdown("Uses `sentence-transformers/all-MiniLM-L6-v2` and LangChain/Qdrant.")

# Load the model once
embedding_model = get_dense_embedding_model()

# Display configuration in the sidebar
st.sidebar.markdown(f"**Loaded Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`")
st.sidebar.markdown(f"**Vector Size:** {QDRANT_CONFIG['VECTOR_SIZE']}")
st.sidebar.markdown(f"**Qdrant Endpoint:** {QDRANT_CONFIG['QDRANT_HOST']}:{QDRANT_CONFIG['QDRANT_PORT']}")
st.sidebar.markdown(f"**Target Collection:** {QDRANT_CONFIG['COLLECTION_NAME']}")
st.sidebar.markdown(f"**Chunk Size:** 500 tokens")


uploaded_file = st.file_uploader(
    "**1. Upload a PDF Document**",
    type="pdf"
)

if uploaded_file is not None:
    # Use a temporary file to save the uploaded PDF for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
        
    st.subheader(f"Processing File: **{uploaded_file.name}**")
    
    if st.button("🚀 Ingest & Embed"):
        # Call the separated Qdrant service function
        ingest_documents_to_qdrant(tmp_file_path, embedding_model)
        
    # Clean up the temporary file (handled within the Streamlit run cycle)
    try:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
    except:
        # Ignore errors during cleanup to keep the app running smoothly
        pass 

else:
    st.info("Please upload a PDF file to begin the ingestion process.")