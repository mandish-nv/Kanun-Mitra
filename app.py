import streamlit as st
import os
import tempfile

from ingestion_pipeline import ingest_documents_to_qdrant
from rag_graph import run_rag_with_graph 
from rag_query import generate_compliant_rules 
import config

st.set_page_config(page_title="Docx-Query-RAG", layout="wide")

def main():
    st.title("Docx-Query-RAG")

    # --- Sidebar Navigation & Config ---
    with st.sidebar:
        st.header("Navigation")
        app_mode = st.radio("Choose Mode", ["Chat with PDF", "Rule Generation"])
        
        st.divider()
        st.header("Document Ingestion")
        uploaded_file = st.file_uploader("Upload PDF (Laws/Docs)", type=["pdf"])

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

    # --- MODE: CHAT WITH PDF ---
    if app_mode == "Chat with PDF":
        st.subheader("Interactive Knowledge Base")
        st.caption("Ask questions about the uploaded documents.")

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

                    # Transparency Section
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

# --- Update the Rule Generation section in app.py ---

    elif app_mode == "Rule Generation":
        st.subheader("Intelligent Rule Generator & Compliance Auditor")
        
        with st.sidebar:
            st.divider()
            st.markdown("### Rule Parameters")
            
            # 1. Dropdown for Industry Types
            industry_options = list(config.INDUSTRY_MANDATORY_RULES.keys())
            rule_context = st.selectbox(
                "Organization Type", 
                options=industry_options,
                help="Select the industry to apply mandatory legal templates."
            )
            
            custom_rules_input = st.text_area(
                "Custom Rules / Desires", 
                height=150, 
                placeholder="E.g., We want flexible working hours and remote work options."
            )
            generate_btn = st.button("Generate Rule Book")

        if generate_btn:
            if not custom_rules_input:
                st.warning("Please provide some custom rules or desires.")
            else:
                with st.spinner("Analyzing laws and drafting Rule Book..."):
                    generated_rules, compliance_report, source_docs = generate_compliant_rules(
                        rule_context, 
                        custom_rules_input
                    )

                # Layout
                col1, col2 = st.columns([1.2, 0.8])

                with col1:
                    st.success(f"📜 {rule_context} Rule Book")
                    st.markdown(generated_rules)
                    
                    # 5. Download Option
                    st.download_button(
                        label="Download Rule Book (Markdown)",
                        data=generated_rules,
                        file_name=f"{rule_context.replace(' ', '_')}_Rules.md",
                        mime="text/markdown"
                    )

                with col2:
                    st.error("Compliance Audit")
                    st.markdown(compliance_report)
                
                # Sources
                st.divider()
                with st.expander("View Referenced Legal Context"):
                    for i, d in enumerate(source_docs):
                        st.markdown(f"**Source {i+1} (Page {d['page']})**")
                        st.info(d["content"])
                        
if __name__ == "__main__":
    main()
