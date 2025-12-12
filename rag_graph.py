import time
from langgraph.graph import StateGraph, END
from dataclasses import dataclass, field
from typing import List, Dict, Any
import rag_query

@dataclass
class RAGState:
    user_query: str
    refined_queries: List[str] = field(default_factory=list)
    retrieved_docs: List[Dict[str, Any]] | None = None
    answer: str | None = None
    chat_history: list | None = None
    timings: Dict[str, float] = field(default_factory=dict)


# -------------------- NODES --------------------

def refine_query_node(state: dict) -> dict:
    """
    Generates N parallel queries for better coverage.
    """
    start = time.time()
    
    # Returns a List[str] containing original + generated queries
    refined_list = rag_query.generate_refined_query(state["user_query"])
    state["refined_queries"] = refined_list

    state["timings"]["refine_query_ms"] = (time.time() - start) * 1000
    return state


def retrieve_docs_node(state: dict) -> dict:
    """
    Orchestrates Parallel Search -> RRF -> Re-ranking.
    """
    start = time.time()

    # Pass the list of refined queries to the hybrid search function
    answer, docs = rag_query.query_qdrant_rag(
        user_query=state["user_query"],
        chat_history=state["chat_history"] or [],
        refined_queries=state["refined_queries"]
    )

    state["answer"] = answer
    state["retrieved_docs"] = docs

    state["timings"]["retrieve_and_gen_ms"] = (time.time() - start) * 1000
    return state


def final_answer_node(state: dict) -> dict:
    # Record total timing
    state["timings"]["total_ms"] = (
        state["timings"].get("refine_query_ms", 0)
        + state["timings"].get("retrieve_and_gen_ms", 0)
    )
    return state


# -------------------- BUILD GRAPH --------------------

def build_rag_graph():
    graph = StateGraph(dict) # Using dict as state container for flexibility

    graph.add_node("refine_query", refine_query_node)
    graph.add_node("retrieve_docs", retrieve_docs_node)
    graph.add_node("final_answer", final_answer_node)

    graph.set_entry_point("refine_query")
    graph.add_edge("refine_query", "retrieve_docs")
    graph.add_edge("retrieve_docs", "final_answer")
    graph.add_edge("final_answer", END)

    return graph.compile()


rag_graph = build_rag_graph()


# -------------------- EXECUTION WRAPPER --------------------

def run_rag_with_graph(user_query: str, chat_history: list):
    """
    Main entry point called by app.py.
    """
    # Initial state
    state = {
        "user_query": user_query,
        "refined_queries": [],
        "answer": None,
        "retrieved_docs": None,
        "chat_history": chat_history,
        "timings": {}
    }

    result_state = rag_graph.invoke(state)

    return (
        result_state.get("answer"),
        result_state.get("retrieved_docs"),
        result_state.get("timings"),
        result_state.get("refined_queries") 
    )
