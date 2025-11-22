import streamlit as st
import requests
import json
import time
from qdrant_client import QdrantClient, models
import config

def generate_refined_queries(user_query):
    """Uses Gemini to generate search query variations."""
    prompt = f"{config.QUERY_GEN_PROMPT}\nUser Question: {user_query}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(
            f"{config.LLM_API_URL}?key={config.GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            result = response.json()
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            queries = [q.strip() for q in text.split('\n') if q.strip()]
            queries.append(user_query)
            return list(set(queries))
    except Exception:
        pass
    return [user_query]

def query_qdrant_rag(user_query: str):
    """Refines query, performs Hybrid Search (RRF), generates RAG response."""
    
    dense_model = config.get_dense_model()
    sparse_model = config.get_sparse_model()

    if not dense_model or not sparse_model:
        return "LLM Error (Missing Embeddings Model)", []

    try:
        client = QdrantClient(url=config.QDRANT_URL)
        if not client.collection_exists(collection_name=config.COLLECTION_NAME):
            return "Collection not found. Please ingest a document.", []
    except Exception as e:
        return f"Connection Error: {e}", []

    with st.spinner("Generating query variations..."):
        queries_to_search = generate_refined_queries(user_query)
        st.caption(f"Internal search variations: {queries_to_search}")

    # Hybrid Search (Dense + Sparse)
    try:
        # Embed Query (Dense)
        dense_query = dense_model.embed_query(user_query)
        
        # Embed Query (Sparse)
        sparse_embedding_obj = list(sparse_model.embed([user_query]))[0]
        sparse_query = models.SparseVector(
            indices=sparse_embedding_obj.indices.tolist(),
            values=sparse_embedding_obj.values.tolist()
        )

        # Use Qdrant's Prefetch for Reciprocal Rank Fusion (RRF)
        prefetch = [
            models.Prefetch(
                query=dense_query,
                using=config.DENSE_VECTOR_NAME,
                limit=20, 
            ),
            models.Prefetch(
                query=sparse_query,
                using=config.SPARSE_VECTOR_NAME,
                limit=20,
            ),
        ]
        
        # CORRECTION: Used 'fusion' instead of 'method'
        hybrid_result = client.query_points(
            collection_name=config.COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF), 
            limit=5,
            with_payload=True
        )

        if not hybrid_result.points:
            return "No matching content found in documents.", []

        retrieved_docs = []
        context_parts = []
        
        for point in hybrid_result.points:
            payload = point.payload
            content = payload.get("page_content", "")
            page = payload.get("page_number", "?")
            
            context_str = f"[Page {page}]: {content}"
            context_parts.append(context_str)
            
            retrieved_docs.append({
                "content": content,
                "page": page,
                "score": point.score
            })

        full_context = "\n\n".join(context_parts)

    except Exception as e:
        st.error(f"Search Logic Error: {e}")
        return f"Search Error: {e}", []

    # LLM Generation
    final_prompt = f"CONTEXT:\n{full_context}\n\nUSER QUESTION: {user_query}"

    payload = {
        "contents": [{"parts": [{"text": final_prompt}]}],
        "systemInstruction": {"parts": [{"text": config.RAG_SYSTEM_PROMPT}]},
    }

    try:
        response = requests.post(
            f"{config.LLM_API_URL}?key={config.GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        
        result = response.json()
        answer = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        return answer, retrieved_docs

    except Exception as e:
        return f"LLM Error: {e}", retrieved_docs