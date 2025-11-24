import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient #

# Load environment variables
load_dotenv()

# ---------------- API KEYS ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------- QDRANT CONFIG ----------------
COLLECTION_NAME = "pdf_rag_hybrid_collection"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

# Dense Configuration (all-MiniLM-L6-v2)
VECTOR_SIZE = 384 
DENSE_VECTOR_NAME = "dense_vector"

# Sparse Configuration (SPLADE)
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"
SPARSE_VECTOR_NAME = "sparse_vector"

# ---------------- LLM CONFIG ----------------
LLM_MODEL = "gemini-2.5-flash"
LLM_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"

# ---------------- SYSTEM PROMPTS ----------------
RAG_SYSTEM_PROMPT = """
You are an expert AI assistant capable of answering questions based strictly on the provided context.
1. ANALYZE the Context provided below carefully.
2. SYNTHESIZE an answer that addresses the User Query.
3. CITE specific parts of the context if applicable (e.g., "According to page 2...").
4. If the answer is NOT in the context, explicitly state: "I cannot find the answer in the provided document."
"""

QUERY_GEN_PROMPT = """
You are a helpful assistant that generates search queries. 
Generate 3 specific, keyword-rich search queries based on the user's question to improve retrieval from a vector database.
Output ONLY the query. Do not add numbering or explanation.
"""

# ---------------- CACHED RESOURCES ----------------
_DENSE_MODEL = None
_SPARSE_MODEL = None

@st.cache_resource
def get_qdrant_client():
    """Return cached Qdrant Client instance."""
    try:
        return QdrantClient(url=QDRANT_URL)
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {e}")
        return None

def get_dense_model():
    """Return the initialized Dense embeddings model (LangChain wrapper)."""
    global _DENSE_MODEL
    if _DENSE_MODEL is None:
        try:
            _DENSE_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Error loading Dense Model: {e}")
            return None
    return _DENSE_MODEL

def get_sparse_model():
    """Return the initialized Sparse embeddings model (FastEmbed)."""
    global _SPARSE_MODEL
    if _SPARSE_MODEL is None:
        try:
            _SPARSE_MODEL = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        except Exception as e:
            st.error(f"Error loading Sparse Model (fastembed): {e}")
            return None
    return _SPARSE_MODEL