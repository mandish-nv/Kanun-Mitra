import streamlit as st
import os
import tempfile

# Import functions
from ingestion_pipeline import ingest_documents_to_qdrant
from rag_query import query_qdrant_rag

st.set_page_config(page_title="Qdrant Hybrid RAG", page_icon="🧠", layout="wide")

def main():
    st.title("🧠 Advanced Hybrid RAG (Dense + Sparse)")
    st.markdown("""
    **Features:** 1. Hybrid Search (Vector + Keyword/SPLADE) 
    2. Query Expansion 
    3. Metadata Indexing
    """)

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Ingesting (Hybrid Mode)..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        ingest_documents_to_qdrant(tmp_path)
                    finally:
                        os.remove(tmp_path)

    # --- CHAT ---
    st.divider()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if user_query := st.chat_input("Ask a specific question about the document..."):
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing vectors..."):
                answer, docs = query_qdrant_rag(user_query)
                
                st.markdown(answer)
                
                # Expandable Source Viewer
                if docs:
                    with st.expander("🔍 View Retrieved Context (Hybrid Results)"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Rank {i+1} (Page {doc['page']}) [Score: {doc['score']:.3f}]**")
                            st.text(doc['content'])
                            st.divider()
                            
        # Add assistant message to chat
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()