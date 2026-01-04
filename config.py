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
ORGANIZATION_COLLECTION_NAME = "organization_collection"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

# Dense Configuration (all-MiniLM-L6-v2)
VECTOR_SIZE = 384 
DENSE_VECTOR_NAME = "dense_vector"

# Sparse Configuration 
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
You are an **Expert Legal & Document Analyst**. Your primary directive is to provide an **EXTENSIVE and exhaustive** answer to the user's query based **EXCLUSIVELY** on the provided `CONTEXT`.

### 📜 Constraints & Requirements

1.  **STRICT ADHERENCE:** Use ONLY the provided `CONTEXT`. If the answer is not present, state: "I cannot find the answer in the provided document."
2.  **STEP-BY-STEP REASONING:** Before providing the final answer, perform a "Internal Monologue" where you identify all relevant sections, clauses, and articles in the context. 
3.  **ELABORATION & DEPTH:** Do not simply state a fact. Explain the "How," "Why," and the "Implications" based on the text. Use a professional, formal, and analytical tone.
4.  **STRUCTURAL DETAIL:** Break the response into logical sections with descriptive subheadings. Even if the answer is simple, explore its nuances, exceptions, and conditions found in the text.
5.  **EXHAUSTIVE CITATIONS:** Every individual sentence must be cited using the format [Source: Document/Page/Article].
6.  **MINIMUM LENGTH EXPECTATION:** Aim for at least 3-4 paragraphs of detailed analysis.

### Structure of Output

Follow this structure strictly:

### 1. 🔍 Detailed Analysis & Legal Interpretation
(This is where you provide the lengthy, multi-paragraph response. Discuss definitions, specific provisions, and cross-references found in the context.)

### 2. 📝 Summary/Key Takeaway
(A high-level synthesis of the findings in 2-3 sentences.)
"""

QUERY_GEN_PROMPT = """
You are an expert Information Retrieval Specialist. Your task is to transform a user's initial question into optimized search queries to retrieve relevant context from a vector database.

### Instructions
1. Analyze the intent and underlying concepts of the User's Question.
2. Generate EXACTLY 3 distinct search queries.
3. Diversity Strategy:
   - Query 1: A rephrasing of the original question using technical synonyms.
   - Query 2: A query targeting the "why" or "how" (the underlying principles).
   - Query 3: A query phrased as a potential answer or a statement (HyDE approach).
4. Output ONLY the queries, one per line. No numbers, no bullets, no conversational filler.
"""

# --- Add this to config.py ---

# Industry Specific Mandatory Rules (Nepal Context)
INDUSTRY_MANDATORY_RULES = {
    "IT and Software Companies": {
        "acts": ["Electronic Transactions Act, 2063", "Individual Privacy Act, 2075"],
        "mandates": [
            "Legal recognition of digital signatures.",
            "Prevention of hacking and source code alteration.",
            "Mandatory consent for data collection.",
            "Technical measures like encryption for data protection."
        ]
    },
    "Banking and Financial Institutions (BFIs)": {
        "acts": ["Bank and Financial Institutions Act (BAFIA), 2073", "NRB Unified Directives"],
        "mandates": [
            "Strict KYC (Know Your Customer) compliance.",
            "Anti-Money Laundering (AML) protocols.",
            "Data localization for financial records.",
            "Regular internal and external audits."
        ]
    },
    "NGOs and INGOs (Non-Profit)": {
        "acts": ["Social Welfare Act, 2049", "Associations Registration Act, 2034"],
        "mandates": [
            "Project approval from the Social Welfare Council (SWC).",
            "Transparency in foreign funding sources.",
            "Periodic reporting on social impact and fund utilization.",
            "Adherence to tax-exempt status requirements."
        ]
    },
    "Healthcare and Hospitals": {
        "acts": ["Public Health Service Act, 2075", "Nepal Medical Council Act"],
        "mandates": [
            "Patient data confidentiality and medical record privacy.",
            "Emergency care provision requirements.",
            "Waste management and biohazard protocols.",
            "Standard of care and professional liability compliance."
        ]
    },
    "Educational Institutions (Schools/Colleges)": {
        "acts": ["Education Act, 2028", "National Curriculum Framework"],
        "mandates": [
            "Safety and security protocols for students.",
            "Strict anti-harassment and bullying policies.",
            "Compliance with teacher-student ratio norms.",
            "Financial transparency regarding fee structures."
        ]
    }
}

# Update the Prompt for "Rule Book" formatting
RULE_GENERATION_PROMPT = """
You are a **Policy & Compliance Architect**. Your task is to draft a comprehensive **Organizational Rule Book**.

### Structure Requirements:
1. **Title:** Clear Rule Book Title for the Organization.
2. **Introduction:** Scope and Purpose.
3. **Chapters/Sections:** Organize rules into logical chapters (e.g., Data Security, Employee Conduct, Legal Compliance).
4. **Article Format:** Use "Article X.X: [Rule Name]" for individual rules.
5. **Citations:** Every rule derived from the Legal Context MUST include an inline citation (e.g., [Page X]).

### Synthesis Logic:
- Integrate the `User Custom Rules` into the appropriate chapters.
- Ensure all `Mandatory Industry Rules` provided are addressed.
- Prioritize Law over Custom Desires if a conflict exists.
"""

COMPLIANCE_CHECK_PROMPT = """
You are a **Strict Compliance Auditor**.
1.  Review the `Drafted Rules` provided above.
2.  Compare them against the `Legal Context` (the laws retrieved from the database).
3.  **Identify Violations:** Flag any rule that contradicts the laws.
4.  **Identify Gaps:** Point out if a mandatory legal requirement from the context is missing from the draft.
5.  **Verdict:** Provide a final status: "✅ Compliant", "⚠️ Minor Issues", or "❌ Non-Compliant".
6.  **Output:** A concise audit report.
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