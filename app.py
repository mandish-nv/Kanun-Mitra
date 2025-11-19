import streamlit as st
import os
import tempfile
import rag_backend  # Importing the logic file

# Set page configuration
st.set_page_config(
    page_title="Qdrant RAG Chat",
    page_icon="🤖",
    layout="wide"
)

def main():
    st.title("🤖 PDF RAG with Qdrant & Gemini")
    st.markdown("Upload a PDF, process it into the Vector DB, and ask questions based on its content.")

    # --- SIDEBAR: CONFIG & UPLOAD ---
    with st.sidebar:
        st.header("Configuration")
        
        # PDF Uploader
        st.subheader("1. Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        # Process Button
        if uploaded_file is not None:
            if st.button("Process / Ingest PDF"):
                with st.spinner("Processing PDF..."):
                    # Save uploaded file to a temporary file because PyPDFLoader needs a path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Call the backend ingestion function
                        rag_backend.ingest_documents_to_qdrant(tmp_path)
                    finally:
                        # Cleanup temp file
                        os.remove(tmp_path)

    # --- MAIN AREA: CHAT INTERFACE ---
    st.divider()
    st.subheader("2. Ask Questions")
    
    user_query = st.text_input("Enter your question here:", placeholder="What is this document about?")

    if st.button("Ask Gemini"):
        if user_query:
            with st.spinner("Thinking..."):
                # Call the backend query function
                answer, docs = rag_backend.query_qdrant_rag(user_query)
            
            # Display Answer
            st.markdown("### Answer:")
            st.write(answer)
            
            # Display Sources in an Expander
            if docs:
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(doc.page_content)
                        st.divider()
        else:
            st.warning("Please enter a question first.")

if __name__ == "__main__":
    main()