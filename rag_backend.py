import streamlit as st
import requests
import time
import json
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document  # Added for object reconstruction
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ---------------- QDRANT CONFIG ----------------
COLLECTION_NAME = "pdf_rag_collection"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
VECTOR_SIZE = 384  # for all-MiniLM-L6-v2

# ---------------- LLM CONFIG ----------------
LLM_MODEL = "gemini-2.5-flash"
LLM_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"
API_KEY = api_key

# ---------------- EMBEDDINGS MODEL ----------------
try:
    EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error initializing HuggingFaceEmbeddings: {e}. Please ensure 'sentence-transformers' is installed.")
    EMBEDDINGS_MODEL = None


def get_embeddings_model():
    """Return the initialized embeddings model."""
    return EMBEDDINGS_MODEL


# ---------------------------------------------------------
# INGESTION PIPELINE
# ---------------------------------------------------------
def ingest_documents_to_qdrant(pdf_path):
    """Loads PDF, chunks it, creates Qdrant collection, and stores vectors."""

    if not EMBEDDINGS_MODEL:
        st.error("Embeddings model not loaded. Ingestion cannot proceed.")
        return

    st.info("Starting document ingestion...")

    # 1. Load PDF
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        st.success(f"Loaded {len(documents)} page(s).")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    st.success(f"Split into {len(chunks)} chunks.")

    # 3. Qdrant ingestion
    try:
        # Initialize Client Explicitly
        client = QdrantClient(url=QDRANT_URL)

        # Check and recreate collection
        if client.collection_exists(collection_name=COLLECTION_NAME):
            client.delete_collection(collection_name=COLLECTION_NAME)
            st.info(f"Deleted existing collection '{COLLECTION_NAME}'.")
        
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        st.success("Created new collection structure.")

        embeddings_model = get_embeddings_model()

        # FIX: Use direct Constructor + add_documents
        # This avoids the 'missing path' error in from_documents/from_existing_collection
        vector_store = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings_model
        )
        
        vector_store.add_documents(documents=chunks)

        st.balloons()
        info = client.get_collection(collection_name=COLLECTION_NAME)
        st.success(f"Stored {len(chunks)} vectors in Qdrant. Total vectors: {info.points_count}")

    except Exception as e:
        st.error(f"Qdrant ingestion error: {e}")


# ---------------------------------------------------------
# RAG QUERY PIPELINE
# ---------------------------------------------------------
def query_qdrant_rag(user_query: str):
    """Performs similarity search in Qdrant, then uses the retrieved context for RAG with Gemini."""

    if not EMBEDDINGS_MODEL:
        return "LLM Error (Missing Embeddings Model)", []

    st.info(f"Searching Qdrant for: {user_query}")

    try:
        embeddings_model = get_embeddings_model()
        
        # Initialize Client Explicitly
        client = QdrantClient(url=QDRANT_URL)

        if not client.collection_exists(collection_name=COLLECTION_NAME):
            st.warning("Collection does not exist. Please ingest a document first.")
            return "No documents ingested yet.", []

    except Exception as e:
        st.error(f"Error connecting to Qdrant: {e}")
        return f"Error connecting to Qdrant: {e}", []

    # Retrieve top-k documents using query_points
    try:
        # 1. Convert query text to vector manually
        query_vector = embeddings_model.embed_query(user_query)

        # 2. Use client.query_points instead of vector_store.similarity_search
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=4,
            with_payload=True
        )

        if not search_result.points:
            st.warning("No relevant documents found.")
            return "No matching content in the PDF.", []

        # 3. Map Qdrant points back to LangChain Document format
        # This ensures the rest of your code (using doc.page_content) works
        retrieved_docs = []
        for point in search_result.points:
            # LangChain typically stores content in 'page_content' inside payload
            content = point.payload.get("page_content", "")
            metadata = point.payload.get("metadata", {})
            retrieved_docs.append(Document(page_content=content, metadata=metadata))

        context = "\n---\n".join(doc.page_content for doc in retrieved_docs)
        st.success("Retrieved relevant document chunks.")

    except Exception as e:
        st.error(f"Similarity search error: {e}")
        return f"Search error: {e}", []

    # LLM Query
    system_instruction = (
        "You are an expert Q&A model. Only answer using the provided CONTEXT. "
        "If the answer is not in the context, say so clearly."
    )

    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {user_query}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
    }
    llm_api_endpoint = f"{LLM_API_URL}?key={API_KEY}"

    for attempt in range(3):
        try:
            response = requests.post(
                llm_api_endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            response.raise_for_status()

            result = response.json()
            candidate = result.get("candidates", [{}])[0]
            answer = candidate.get("content", {}).get("parts", [{}])[0].get("text")

            if answer:
                return answer, retrieved_docs
            else:
                st.warning("LLM returned an empty or malformed response.")
                if attempt < 2:
                    time.sleep(1.5 ** attempt)
                else:
                    st.error("LLM API failed after retries.")
                    return "LLM Error (Empty Response)", []

        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(1.5 ** attempt)
            else:
                st.error(f"LLM API failed after retries. Request Error: {e}")
                return "LLM Error (Request Failure)", []
        except Exception as e:
            if attempt < 2:
                time.sleep(1.5 ** attempt)
            else:
                st.error(f"An unexpected error occurred: {e}")
                return "LLM Error (Unexpected Exception)", []

    return "Failed to generate response after all attempts.", []