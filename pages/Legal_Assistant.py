import streamlit as st
from rag_graph import run_rag_with_graph
from utils.ui_components import init_page
import config

user_info = init_page("Legal Assistant")

# st.set_page_config(page_title="Legal Assistant", layout="wide")

st.title("💬 Legal Assistant Chat")
st.info("Ask questions based on the ingested legal documents.")

st.markdown("""
    <style>
        [data-testid="stSidebarNav"] ul li:first-child {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# with st.sidebar:
#     if st.button("⬅ Back to Dashboard"):
#         st.switch_page("main.py") # or your main dashboard file

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal context..."):
            answer, docs, timings, refined_queries = run_rag_with_graph(
                user_query, st.session_state.messages[:-1], collection_name=config.COLLECTION_NAME
            )
            st.markdown(answer)
            
            with st.expander("Search Transparency"):
                st.write("**Refined Queries:**", refined_queries)
                st.json(timings)
            
            if docs:
                with st.expander("Source Context"):
                    for d in docs:
                        st.markdown(f"**Page {d['page_number']}** (Score: {d['score']:.2f})")
                        st.caption(d["chunk"])

    st.session_state.messages.append({"role": "assistant", "content": answer})