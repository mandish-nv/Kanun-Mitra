import streamlit as st
import os
import tempfile

from ingestion_pipeline import ingest_documents_to_qdrant
from rag_graph import run_rag_with_graph 

st.set_page_config(page_title="Docx-Query-RAG", layout="wide")

def main():
    st.title("Docx-Query-RAG")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Ingesting (Hybrid Mode)..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    try:
                        ingest_documents_to_qdrant(tmp_path)
                    finally:
                        os.remove(tmp_path)

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask something from the PDF..."):

        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):

                answer, docs, timings, refined_queries = run_rag_with_graph(
                    user_query, 
                    st.session_state.messages[:-1]
                )

                st.markdown(answer)

                # --- Transparency Section ---
                with st.expander("Details: Query Refinement & Timing"):
                    st.subheader("Generated Queries")
                    if refined_queries:
                        for q in refined_queries:
                            st.text(f"- {q}")
                    
                    st.subheader("Timing Report (ms)")
                    st.json(timings)

                # Retrieved Context
                if docs:
                    with st.expander("Retrieved Context (Top Re-ranked)"):
                        for i, d in enumerate(docs):
                            st.markdown(f"**{i+1}. Page {d['page']} — Re-rank Score: {d['score']:.4f}**")
                            st.info(d["content"])

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
