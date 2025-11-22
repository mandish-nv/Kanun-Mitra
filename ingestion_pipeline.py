import streamlit as st
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
import config

def ingest_documents_to_qdrant(pdf_path):
    """
    Processes PDF: Hybrid chunking, Sparse+Dense embedding, and manual Qdrant ingestion.
    """
    
    # 1. Initialize Models
    dense_model = config.get_dense_model()
    sparse_model = config.get_sparse_model() 

    if not dense_model or not sparse_model:
        st.error("Embedding models not loaded. Ingestion cannot proceed.")
        return

    st.info("Starting Hybrid ingestion...")

    # 2. Load PDF
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        st.success(f"Loaded {len(documents)} page(s).")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return

    # 3. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    st.info(f"Created {len(chunks)} content chunks.")

    # 4. Prepare Qdrant Collection (Hybrid Config)
    try:
        client = QdrantClient(url=config.QDRANT_URL)

        if client.collection_exists(collection_name=config.COLLECTION_NAME):
            client.delete_collection(collection_name=config.COLLECTION_NAME)
            st.info(f"Recreating collection '{config.COLLECTION_NAME}'...")
        
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config={
                config.DENSE_VECTOR_NAME: models.VectorParams(
                    size=config.VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                config.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            }
        )

        # Indexing 'page_number' for fast filtering
        client.create_payload_index(
            collection_name=config.COLLECTION_NAME,
            field_name="page_number",
            field_schema=models.PayloadSchemaType.INTEGER
        )
        st.success("Collection created with Hybrid support and Indexing.")

    except Exception as e:
        st.error(f"Qdrant initialization error: {e}")
        return

    # 5. Vectorization & Point Creation
    points = []
    texts = [doc.page_content for doc in chunks]
    
    with st.spinner("Generating Dense and Sparse embeddings..."):
        # Dense
        dense_embeddings = dense_model.embed_documents(texts)
        
        # Sparse (CORRECTION: Added batch_size=32 to prevent Memory Error)
        # The default batch_size=256 causes OOM with SPLADE models on some machines
        sparse_embeddings = list(sparse_model.embed(texts, batch_size=32))

    # 6. Construct Points
    for i, doc in enumerate(chunks):
        point_id = str(uuid.uuid4())
        
        # Extract indices and values from FastEmbed sparse object
        sparse_vector = models.SparseVector(
            indices=sparse_embeddings[i].indices.tolist(),
            values=sparse_embeddings[i].values.tolist()
        )

        payload = {
            "page_content": doc.page_content,
            "page_number": doc.metadata.get("page", 0) + 1,
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": i
        }

        points.append(models.PointStruct(
            id=point_id,
            vector={
                config.DENSE_VECTOR_NAME: dense_embeddings[i],
                config.SPARSE_VECTOR_NAME: sparse_vector
            },
            payload=payload
        ))

    # 7. Upload Points
    try:
        client.upload_points(
            collection_name=config.COLLECTION_NAME,
            points=points
        )
        st.balloons()
        st.success(f"Successfully ingested {len(points)} hybrid vectors into Qdrant.")
        
    except Exception as e:
        st.error(f"Error uploading points to Qdrant: {e}")