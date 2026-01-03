import streamlit as st
import sys
import os
import tempfile
import time
import pandas as pd

# Import logic from old code
from ingestion_pipeline import ingest_documents_to_qdrant
from rag_graph import run_rag_with_graph 
from rag_query import generate_compliant_rules 
import config

# Import Auth from new code
from utils.auth import Authentication

# Page configuration
st.set_page_config(
    page_title="KanunMitra - Legal Assistance Portal",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS Styles from New Code ---
st.markdown("""
<style>
    .stApp { overflow: hidden !important; max-height: 100vh !important; }
    ::-webkit-scrollbar { display: none !important; }
    * { scrollbar-width: none !important; }
    .main { overflow: hidden !important; height: 100vh !important; padding: 0; margin: 0; }
    .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
    
    /* Login Button Styling */
    form[data-testid="stForm"] div[data-testid="column"] button[kind="formSubmit"] {
        background-color: #0d6efd !important;
        color: white !important;
        padding: 1.5rem 4rem !important;
        font-size: 1.4rem !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        width: 100% !important;
        height: 70px !important;
        margin-top: 30px !important;
    }
    
    .role-badge {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        color: white;
    }
    .admin { background-color: #ff4b4b; }
    .user { background-color: #0083B8; }
</style>
""", unsafe_allow_html=True)

# Initialize authentication
auth = Authentication()

def show_login_form():
    left_col, right_col = st.columns([2, 1])
    
    with right_col:
        st.markdown('<h1 style="text-align: center; color: #ffffff;">KanunMitra</h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #ffffff;">Legal Assistance Portal</h3>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("**Username**", placeholder="Enter your username", key="login_username")
            password = st.text_input("**Password**", type="password", placeholder="Enter your password", key="login_password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            btn_col = st.columns([1, 2, 1])[1]
            with btn_col:
                login_button = st.form_submit_button("**LOGIN**", use_container_width=True, type="primary")
            
            if login_button:
                if not username or not password:
                    st.error("Please fill in all fields!")
                else:
                    user = auth.authenticate(username, password)
                    if user:
                        st.session_state['user'] = user
                        st.success(f"Welcome, {user['username']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials!")

    with left_col:
        # Note: Ensure this image path is correct relative to your app
        try:
            st.image("images/RobotLaw.png", width=600)
        except:
            st.info("Legal AI System Ready")

def show_main_dashboard():
    """This integrates the 'Old Code' logic as the dashboard"""
    st.title("⚖️ KanunMitra Dashboard")

    # --- Sidebar Navigation (From Old Code) ---
    with st.sidebar:
        user = auth.get_current_user()
        st.markdown(f"**Logged in as:** {user['username']} ({user['role'].upper()})")
        
        if st.button("Logout"):
            auth.logout()
            st.rerun()
            
        st.divider()
        st.header("Navigation")
        app_mode = st.radio("Choose Mode", ["Chat with PDF", "Rule Generation"])
        
        st.divider()
        st.header("Document Ingestion")
        uploaded_file = st.file_uploader("Upload PDF (Laws/Docs)", type=["pdf"])

        if uploaded_file and st.button("Process Document"):
            with st.spinner("Ingesting (Hybrid Mode)..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    ingest_documents_to_qdrant(tmp_path)
                    st.sidebar.success("Document Ingested!")
                finally:
                    os.remove(tmp_path)

    # --- Application Modes ---
    if app_mode == "Chat with PDF":
        render_chat_mode()
    elif app_mode == "Rule Generation":
        render_rule_generation_mode()

def render_chat_mode():
    st.subheader("Interactive Knowledge Base")
    if "messages" not in st.session_state:
        st.session_state.messages = []

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
                
                with st.expander("Details: Query Refinement & Timing"):
                    st.json(timings)
                if docs:
                    with st.expander("Retrieved Context"):
                        for d in docs:
                            st.info(d["content"])

        st.session_state.messages.append({"role": "assistant", "content": answer})

def render_rule_generation_mode():
    st.subheader("Intelligent Rule Generator & Compliance Auditor")
    
    with st.sidebar:
        st.markdown("### Rule Parameters")
        industry_options = list(config.INDUSTRY_MANDATORY_RULES.keys())
        rule_context = st.selectbox("Organization Type", options=industry_options)
        custom_rules_input = st.text_area("Custom Rules / Desires", height=150)
        generate_btn = st.button("Generate Rule Book")

    if generate_btn:
        if not custom_rules_input:
            st.warning("Please provide some custom rules.")
        else:
            with st.spinner("Analyzing laws..."):
                generated_rules, compliance_report, source_docs = generate_compliant_rules(
                    rule_context, 
                    custom_rules_input
                )

            col1, col2 = st.columns([1.2, 0.8])
            with col1:
                st.success(f"📜 {rule_context} Rule Book")
                st.markdown(generated_rules)
                st.download_button("Download Rule Book", data=generated_rules, file_name="Rules.md")

            with col2:
                st.error("Compliance Audit")
                st.markdown(compliance_report)

def main():
    if auth.check_session():
        # User is logged in, show the RAG application
        show_main_dashboard()
    else:
        # User not logged in, show login form
        show_login_form()

if __name__ == "__main__":
    main()