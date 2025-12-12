import streamlit as st
import re
import logging
import time
import random
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from qdrant_client import models
from google import genai
from google.genai import types

import config

# Initialize Google GenAI Client
try:
    if config.GEMINI_API_KEY:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
    else:
        logging.warning("GEMINI_API_KEY not found in environment variables.")
        client = None
except Exception as e:
    logging.error(f"Failed to initialize Google GenAI Client: {e}")
    client = None


def _execute_with_retry(func, retries=3, initial_delay=2):
    """
    Helper to retry API calls with exponential backoff on 429 errors.
    """
    delay = initial_delay
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            # Check for Rate Limit (429) or Service Unavailable (503)
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt == retries - 1:
                    logging.error(f"Max retries reached for API call. Error: {e}")
                    raise e
                
                # Jitter adds randomness to prevent thundering herd problem
                sleep_time = delay + random.uniform(0, 1)
                logging.warning(f"Rate limit hit (429). Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{retries})")
                time.sleep(sleep_time)
                delay *= 2  # Exponential backoff
            else:
                # If it's not a rate limit error, raise immediately
                raise e


def extract_page_number(query: str) -> int | None:
    """Extracts a page number from the query if explicitly mentioned."""
    match = re.search(r"page\s+(\d+)", query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def generate_refined_query(user_query: str) -> List[str]:
    """
    Generates multiple refined queries using Google GenAI SDK.
    Returns a list including the original query and generated variations.
    """
    if not client:
        return [user_query]

    prompt = f"{config.QUERY_GEN_PROMPT}\nUser Question: {user_query}"
    
    def _api_call():
        return client.models.generate_content(
            model=config.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7, 
                top_p=0.95,
                top_k=40
            )
        )

    try:
        # Wrap the API call with retry logic
        response = _execute_with_retry(_api_call)
        
        generated_text = response.text.strip()
        new_queries = [q.strip() for q in generated_text.split('\n') if q.strip()]
        
        all_queries = [user_query] + new_queries
        return list(dict.fromkeys(all_queries))

    except Exception as e:
        # If all retries fail, fall back to just the user query so the app doesn't crash
        logging.error(f"Error in generate_refined_query after retries: {e}")
        return [user_query]


def perform_hybrid_search(
    query: str, 
    client, 
    dense_model, 
    sparse_model, 
    page_filter: int = None
) -> List[Dict]:
    """
    Executes a single hybrid search (Dense + Sparse) for a given query.
    """
    try:
        # 1. Dense Embedding
        dense_query = dense_model.embed_query(query)

        # 2. Sparse Embedding
        sparse_embedding_obj = list(sparse_model.embed([query]))[0]
        sparse_query = models.SparseVector(
            indices=sparse_embedding_obj.indices.tolist(),
            values=sparse_embedding_obj.values.tolist()
        )

        # 3. Construct Filter
        qdrant_filter = None
        if page_filter is not None:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="page_number",
                        match=models.MatchValue(value=page_filter)
                    )
                ]
            )

        # 4. Prefetch objects
        prefetch = [
            models.Prefetch(
                query=dense_query,
                using=config.DENSE_VECTOR_NAME,
                limit=20, 
                filter=qdrant_filter
            ),
            models.Prefetch(
                query=sparse_query,
                using=config.SPARSE_VECTOR_NAME,
                limit=20,
                filter=qdrant_filter
            ),
        ]

        # 5. Execute Query
        results = client.query_points(
            collection_name=config.COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20, 
            with_payload=True
        )

        # 6. Format Results
        docs = []
        for point in results.points:
            docs.append({
                "content": point.payload.get("page_content", ""),
                "page": point.payload.get("page_number", "?"),
                "score": point.score, 
                "id": point.id
            })
        return docs

    except Exception as e:
        logging.error(f"Search failed for query '{query}': {e}")
        return []


def rrf_fusion(results_list: List[List[Dict]], k=60) -> List[Dict]:
    """
    Reciprocal Rank Fusion to merge results from multiple parallel queries.
    """
    fused_scores = {}
    doc_map = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            doc_content = doc["content"]
            if doc_content not in doc_map:
                doc_map[doc_content] = doc
            
            if doc_content not in fused_scores:
                fused_scores[doc_content] = 0.0
            
            fused_scores[doc_content] += 1 / (k + rank + 1)

    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for content, score in sorted_docs:
        d = doc_map[content]
        d["score"] = score 
        final_results.append(d)
        
    return final_results


def rerank_documents(query: str, docs: List[Dict], top_k: int) -> List[Dict]:
    """
    Re-ranks documents using a Cross-Encoder.
    """
    reranker = config.get_rerank_model()
    if not reranker or not docs:
        return docs[:top_k]

    pairs = [[query, d["content"]] for d in docs]
    
    try:
        scores = reranker.predict(pairs)
        
        for i, score in enumerate(scores):
            docs[i]["score"] = float(score) 
            
        ranked_docs = sorted(docs, key=lambda x: x["score"], reverse=True)
        return ranked_docs[:top_k]
    except Exception as e:
        logging.error(f"Re-ranking failed: {e}")
        return docs[:top_k]


def query_qdrant_rag(user_query: str, chat_history: list, refined_queries: List[str] = None):
    """
    Main Orchestrator:
    1. Extract Filters
    2. Parallel Search (Original + Refined Queries)
    3. RRF Fusion
    4. Re-ranking
    5. Final Generation
    """
    
    # Load Resources
    dense_model = config.get_dense_model()
    sparse_model = config.get_sparse_model()
    client_qdrant = config.get_qdrant_client()

    if not dense_model or not sparse_model or not client_qdrant:
        return "System Error: Missing Models or Database Connection.", []

    # 1. Pre-Filtering
    page_filter = extract_page_number(user_query)
    
    search_queries = refined_queries if refined_queries else [user_query]
    
    # 2. Parallel Retrieval
    all_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_query = {
            executor.submit(
                perform_hybrid_search, q, client_qdrant, dense_model, sparse_model, page_filter
            ): q for q in search_queries
        }
        
        for future in as_completed(future_to_query):
            res = future.result()
            if res:
                all_results.append(res)

    if not all_results:
        return "No matching content found in documents.", []

    # 3. RRF Fusion
    fused_docs = rrf_fusion(all_results)

    # 4. Re-Ranking
    final_docs = rerank_documents(user_query, fused_docs, top_k=config.TOP_K_RERANK)

    if not final_docs:
        return "No relevant context found after re-ranking.", []

    # 5. Construct Context
    context_parts = []
    for d in final_docs:
        context_parts.append(f"[Page {d['page']}]: {d['content']}")
    
    full_context = "\n\n".join(context_parts)
    print(f"\n\n\nFULL_CONTEXT:\n{full_context}\n\n\n")

    # 6. Final Generation with Retry Logic
    final_prompt = f"CONTEXT:\n{full_context}\n\nUSER QUESTION: {user_query}"
    def _final_gen_call():
        return client.models.generate_content(
            model=config.LLM_MODEL,
            contents=final_prompt,
            config=types.GenerateContentConfig(
                system_instruction=config.RAG_SYSTEM_PROMPT,
                temperature=config.GEN_CONFIG["temperature"],
                top_p=config.GEN_CONFIG["top_p"],
                top_k=config.GEN_CONFIG["top_k"],
                # max_output_tokens=config.GEN_CONFIG["max_output_tokens"],
            )
        )

    try:
        response = _execute_with_retry(_final_gen_call)
        answer = response.text
        return answer, final_docs

    except Exception as e:
        logging.error(f"LLM Generation Failed after retries: {e}")
        return "I encountered an error generating the answer due to high server load. Please try again in a moment.", final_docs