import os
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
SPARSE_MODEL_NAME = "Qdrant/bm25"
SPARSE_VECTOR_NAME = "sparse_vector"

# ---------------- RERANKING CONFIG ----------------
# Using a standard Cross-Encoder for high-accuracy re-ranking
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANK = 10  # Number of docs to pass to LLM after re-ranking

# ---------------- LLM CONFIG ----------------
LLM_MODEL = "gemini-2.5-flash" 

# Generation Configs exposed for control
GEN_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192, # not used
}

# ---------------- SYSTEM PROMPTS ----------------
RAG_SYSTEM_PROMPT = """
You are an **Expert Document Analyst** and helpful AI assistant. Your primary directive is to answer the user's query **EXCLUSIVELY** based on the provided `CONTEXT` below.

### 📜 Constraints & Requirements

1.  **STRICT ADHERENCE:** You **MUST NOT** use any prior knowledge or information outside of the provided `CONTEXT`.
2.  **CONTEXT MISSING:** If the answer to the User Query cannot be found in the provided `CONTEXT` (even after thorough analysis), you **MUST** respond with the exact phrase: **"I cannot find the answer in the provided document."**
3.  **DETAIL and EXPLANATION:** Provide a comprehensive and detailed explanation of the answer, synthesizing information from the relevant context passages.
4.  **CITATIONS:** For every statement derived from the context, include an inline citation in brackets, referencing the source (e.g., [Paragraph 3], [Page 2], [Source Document Title]). Use the most specific identifier available.

### Structure of Output

Follow this structure for your response:

1.  **Detailed Answer:** The full, explained, and cited response.
2.  **Summary/Key Takeaway:** A brief, single-paragraph summary of the main finding regarding the User Query, placed at the very end.
"""

QUERY_GEN_PROMPT = """
You are a helpful assistant that generates search queries to improve retrieval from a vector database.
1. Analyze the User's Question.
2. Generate EXACTLY 3 specific, keyword-rich search queries that explore different angles of the question.
3. Output ONLY the 3 new queries, separated by newlines. 
4. Do not number them or add bullet points. Just the text.
"""

# ---------------- CACHED RESOURCES ----------------
_DENSE_MODEL = None
_SPARSE_MODEL = None
_RERANK_MODEL = None

@st.cache_resource
def get_qdrant_client():
    """Return cached Qdrant Client instance."""
    try:
        return QdrantClient(url=QDRANT_URL)
    except Exception as e:
        logging.error(f"Error connecting to Qdrant: {e}")
        st.error(f"Error connecting to Qdrant: {e}")
        return None

def get_dense_model():
    """Return the initialized Dense embeddings model (LangChain wrapper)."""
    global _DENSE_MODEL
    if _DENSE_MODEL is None:
        try:
            _DENSE_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logging.error(f"Error loading Dense Model: {e}")
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
            logging.error(f"Error loading Sparse Model (fastembed): {e}")
            st.error(f"Error loading Sparse Model (fastembed): {e}")
            return None
    return _SPARSE_MODEL

def get_rerank_model():
    """Return the initialized CrossEncoder for re-ranking."""
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        try:
            # We use CrossEncoder from sentence_transformers
            _RERANK_MODEL = CrossEncoder(RERANK_MODEL_NAME)
        except Exception as e:
            logging.error(f"Error loading Rerank Model: {e}")
            st.error(f"Error loading Rerank Model: {e}")
            return None
    return _RERANK_MODEL