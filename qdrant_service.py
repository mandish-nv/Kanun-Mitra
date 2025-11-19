import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# --- Qdrant Configuration ---
COLLECTION_NAME = "pdf_rag_collection"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
# The vector size for 'sentence-transformers/all-MiniLM-L6-v2' is 384
VECTOR_SIZE = 384 

# Expose config for use in the Streamlit app
QDRANT_CONFIG = {
    "COLLECTION_NAME": COLLECTION_NAME,
    "QDRANT_HOST": QDRANT_HOST,
    "QDRANT_PORT": QDRANT_PORT,
    "VECTOR_SIZE": VECTOR_SIZE,
}

def ingest_documents_to_qdrant(pdf_path, embeddings_model):
    """
    1. Loads PDF, 2. Chunks text, and 3. Stores embeddings in Qdrant.
    
    This function includes Streamlit calls (st.info, st.success, etc.) 
    to provide real-time status updates to the user interface.
    
    Args:
        pdf_path (str): The file path to the uploaded PDF.
        embeddings_model: The loaded HuggingFaceEmbeddings model instance.
    """
    st.info("Starting document ingestion and processing...")

    # 1. Document Ingestion
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        st.success(f"Loaded {len(documents)} page(s) from PDF.")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    st.success(f"Chunked document into {len(chunks)} text snippets.")

    # 3. Vector Store: Store embeddings in Qdrant
    try:
        # Initialize the Qdrant Client (Connecting to persistent instance)
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # --- Check and Create Collection ---
        try:
            # Check if the collection exists
            client.get_collection(collection_name=COLLECTION_NAME)
            st.info(f"Collection **{COLLECTION_NAME}** found. Appending vectors...")
        except UnexpectedResponse:
            # If not found, create it
            st.warning(f"Collection **{COLLECTION_NAME}** not found. Creating it now...")
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            )
            st.success(f"Collection **{COLLECTION_NAME}** created successfully.")
        
        # 1. Initialize LangChain Qdrant object
        vector_store = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings_model
        )

        # 2. Add documents (embed and upload)
        vector_store.add_documents(
            documents=chunks,
        )

        st.balloons()
        st.success(f"Successfully stored {len(chunks)} vectors in Qdrant collection: **{COLLECTION_NAME}**")
        
        # Example check: Get the total collection point count
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        st.write(f"Total Vector Count in Collection: {collection_info.points_count}")
        
    except Exception as e:
        st.error(f"An error occurred during Qdrant connection or storage: {e}")