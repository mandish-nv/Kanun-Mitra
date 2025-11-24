import streamlit as st
import requests
import json
from qdrant_client import models
import config

def generate_refined_query(user_query):
    """Uses Gemini to generate a single optimized search query."""
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
            return text.strip()
    except Exception:
        pass
    return user_query

def query_qdrant_rag(user_query: str, chat_history: list, page_filter: int = None):
    """
    Refines query, performs Hybrid Search with Filters, and generates response with history.
    """
    
    dense_model = config.get_dense_model()
    sparse_model = config.get_sparse_model()
    client = config.get_qdrant_client() # Cached client

    if not dense_model or not sparse_model or not client:
        return "LLM Error (Missing Resources)", []

    try:
        if not client.collection_exists(collection_name=config.COLLECTION_NAME):
            return "Collection not found. Please ingest a document.", []
    except Exception as e:
        return f"Connection Error: {e}", []

    with st.spinner("Optimizing search query..."):
        refined_query = generate_refined_query(user_query)
        st.caption(f"Generated Search Query: {refined_query}")

    query_filter = None
    if page_filter and page_filter > 0:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="page_number",
                    match=models.MatchValue(value=page_filter)
                )
            ]
        )

    # Hybrid Search
    try:
        # Embed the REFINED query
        dense_query = dense_model.embed_query(refined_query)
        
        sparse_embedding_obj = list(sparse_model.embed([refined_query]))[0]
        sparse_query = models.SparseVector(
            indices=sparse_embedding_obj.indices.tolist(),
            values=sparse_embedding_obj.values.tolist()
        )

        prefetch = [
            models.Prefetch(
                query=dense_query,
                using=config.DENSE_VECTOR_NAME,
                filter=query_filter, 
                limit=20, 
            ),
            models.Prefetch(
                query=sparse_query,
                using=config.SPARSE_VECTOR_NAME,
                filter=query_filter,
                limit=20,
            ),
        ]
        
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

    history_text = ""
    if chat_history:
        # Take last 3 messages excluding the current user prompt which is added via 'final_prompt' logic
        recent_history = chat_history[-3:] 
        history_text = "CHAT HISTORY:\n" + "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_history]) + "\n\n"

    final_prompt = f"{history_text}CONTEXT:\n{full_context}\n\nUSER QUESTION: {user_query}"

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