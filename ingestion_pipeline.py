# import streamlit as st
# import uuid
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from qdrant_client import models
# import config

# def ingest_documents_to_qdrant(pdf_path):
#     """
#     Processes PDF: Hybrid chunking, Sparse+Dense embedding, and manual Qdrant ingestion.
#     """
    
#     # 1. Initialize Models & Client
#     dense_model = config.get_dense_model()
#     sparse_model = config.get_sparse_model() 
#     client = config.get_qdrant_client() # Using cached client

#     if not dense_model or not sparse_model or not client:
#         st.error("Resources not loaded. Ingestion cannot proceed.")
#         return

#     st.info("Starting Hybrid ingestion...")

#     # 2. Load PDF
#     try:
#         loader = PyPDFLoader(pdf_path)
#         documents = loader.load()
#         st.success(f"Loaded {len(documents)} page(s).")
#     except Exception as e:
#         st.error(f"Error loading PDF: {e}")
#         return

#     # 3. Chunking
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len,
#         add_start_index=True # Helps preserve metadata order
#     )
#     chunks = splitter.split_documents(documents)
#     st.info(f"Created {len(chunks)} content chunks.")

#     # 4. Prepare Qdrant Collection
#     try:
#         # if client.collection_exists(collection_name=config.COLLECTION_NAME):
#         #     client.delete_collection(collection_name=config.COLLECTION_NAME)
#         #     st.info(f"Recreating collection '{config.COLLECTION_NAME}'...")
        
#         # client.create_collection(
#         #     collection_name=config.COLLECTION_NAME,
#         #     vectors_config={
#         #         config.DENSE_VECTOR_NAME: models.VectorParams(
#         #             size=config.VECTOR_SIZE,
#         #             distance=models.Distance.COSINE,
#         #         )
#         #     },
#         #     sparse_vectors_config={
#         #         config.SPARSE_VECTOR_NAME: models.SparseVectorParams(
#         #             index=models.SparseIndexParams(
#         #                 on_disk=False,
#         #             )
#         #         )
#         #     }
#         # )

#         # Indexing 'page_number' for fast filtering
#         client.create_payload_index(
#             collection_name=config.COLLECTION_NAME,
#             field_name="page_number",
#             field_schema=models.PayloadSchemaType.INTEGER
#         )
#         st.success("Collection created with Hybrid support and Indexing.")

#     except Exception as e:
#         st.error(f"Qdrant initialization error: {e}")
#         return

#     # 5. Vectorization
#     texts = [doc.page_content for doc in chunks]
    
#     with st.spinner("Generating Dense and Sparse embeddings..."):
#         dense_embeddings = dense_model.embed_documents(texts)
#         sparse_embeddings = list(sparse_model.embed(texts, batch_size=32))

#     # 6. Construct Points
#     points = []
#     for i, doc in enumerate(chunks):
#         point_id = str(uuid.uuid4())
        
#         sparse_vector = models.SparseVector(
#             indices=sparse_embeddings[i].indices.tolist(),
#             values=sparse_embeddings[i].values.tolist()
#         )

#         # PyPDFLoader uses 0-based indexing. 
#         original_page = doc.metadata.get("page")
#         if original_page is not None:
#             page_num = int(original_page) + 1 
#         else:
#             page_num = 0 # Fallback

#         payload = {
#             "page_content": doc.page_content,
#             "page_number": page_num, 
#             "source": doc.metadata.get("source", "unknown"),
#             "chunk_index": i
#         }

#         points.append(models.PointStruct(
#             id=point_id,
#             vector={
#                 config.DENSE_VECTOR_NAME: dense_embeddings[i],
#                 config.SPARSE_VECTOR_NAME: sparse_vector
#             },
#             payload=payload
#         ))

#     # 7. Upload Points
#     try:
#         client.upload_points(
#             collection_name=config.COLLECTION_NAME,
#             points=points
#         )
#         st.balloons()
#         st.success(f"Successfully ingested {len(points)} hybrid vectors into Qdrant.")
        
#     except Exception as e:
#         st.error(f"Error uploading points to Qdrant: {e}")

import streamlit as st
import uuid
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from qdrant_client import models
import config
import re

def extract_filename_from_markdown(md_content: str, fallback_name: str) -> str:
    """
    Extracts the first Markdown H1 (# ...) as filename.
    Falls back to original filename if not found.
    """
    for line in md_content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            title = line[2:].strip()
            # Sanitize filename
            title = re.sub(r'[\\/*?:"<>|]', "", title)
            title = re.sub(r"\s+", "_", title)
            return f"{title}.pdf"
    return fallback_name


def ingest_documents_to_qdrant(pdf_files, user_role="user"):
    # 1. Determine Collection Name based on Role
    if user_role == "admin":
        target_collection = config.ORGANIZATION_COLLECTION_NAME
    else:
        target_collection = config.COLLECTION_NAME

    # 2. Setup Resources
    if not pdf_files:
        st.warning("No files provided.")
        return
    if not isinstance(pdf_files, list):
        pdf_files = [pdf_files]

    dense_model = config.get_dense_model()  
    sparse_model = config.get_sparse_model()
    client = config.get_qdrant_client() 

    # 2. Initialization using your "Scroll" technique
    offset = 0          # This will be your file_chunk_id (Continuous)
    product_offset = 0  # This will be your global_chunk_id (Per PDF)

    try:
        if not client.collection_exists(collection_name=target_collection):
            client.create_collection(
                collection_name=target_collection,
                vectors_config={config.DENSE_VECTOR_NAME: models.VectorParams(size=config.VECTOR_SIZE, distance=models.Distance.COSINE)},
                sparse_vectors_config={config.SPARSE_VECTOR_NAME: models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))}
            )
            client.create_payload_index(target_collection, "file_chunk_id", models.PayloadSchemaType.INTEGER)
            client.create_payload_index(target_collection, "global_chunk_id", models.PayloadSchemaType.INTEGER)
            client.create_payload_index(target_collection, "page_number", models.PayloadSchemaType.INTEGER)
        
        info = client.get_collection(collection_name=target_collection)
        if info.points_count != 0:
            res, _ = client.scroll(
                collection_name=target_collection,
                limit=1,
                with_payload=True,
                # order_by={"key": "file_chunk_id", "direction": "desc"} # Following your snippet logic
            )
            if res:
                product_offset = res[0].payload.get("global_chunk_id", 0) + 1
                offset = res[0].payload.get("file_chunk_id", 0) + 1
    except Exception as e:
        st.error(f"Error initializing offsets for {target_collection}: {e}")
        return

    load_progress = st.progress(0)
    
    for i, pdffile_obj in enumerate(pdf_files):
        actual_filename =pdffile_obj.name if hasattr(pdffile_obj, 'name') else "document.pdf"
        try:
            md_content = pymupdf4llm.to_markdown(pdffile_obj)
            # actual_filename = extract_filename_from_markdown(
            # md_content=md_content,
            # fallback_name=pdffile_obj.name
            # )
            md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "legal_act_name"), ("##", "section_name")])
            md_header_splits = md_splitter.split_text(md_content)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
            chunks = text_splitter.split_documents(md_header_splits)

            # Inner loop: Each chunk (idx)
            for doc in chunks:
                content = doc.page_content
                original_page = doc.metadata.get("page")
                if original_page is not None:
                    page_num = int(original_page) + 1 
                else:
                    page_num = 0
                
                client.upsert(
                    collection_name=target_collection,
                    points=[
                        models.PointStruct(
                            id=str(uuid.uuid4()), 
                            vector={
                                config.DENSE_VECTOR_NAME: dense_model.embed_query(content),
                                config.SPARSE_VECTOR_NAME: list(sparse_model.embed([content]))[0].as_object()
                            },
                            payload={
                                "global_chunk_id": product_offset, # Document Index (Per PDF)
                                "file_chunk_id": offset,          # Sequence Index (Continuous)
                                "chunk": content,
                                "page_number": page_num,
                                "source_file": actual_filename,
                                "legal_act_name": doc.metadata.get("legal_act_name", "General Document"),
                            }
                        )
                    ]
                )
                offset += 1 
            
            product_offset += 1
            load_progress.progress((i + 1) / len(pdf_files))

        except Exception as e:
            st.error(f"Error on {actual_filename}: {e}")

    st.success(f"Ingested into **{target_collection}**. Final Global ID: {product_offset}, Final Chunk ID: {offset}")