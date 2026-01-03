import streamlit as st
from rag_query import generate_compliant_rules
import config

st.set_page_config(page_title="Rule Generator", layout="wide")

st.title("⚖️ Intelligent Rule Generator")

with st.sidebar:
    if st.button("⬅ Back to Dashboard"):
        st.switch_page("main.py") # or your main dashboard file

col_input, col_output = st.columns([1, 2])

with col_input:
    st.header("Parameters")
    industry = st.selectbox("Industry", options=list(config.INDUSTRY_MANDATORY_RULES.keys()))
    custom_input = st.text_area("Specific Requirements", placeholder="e.g. Remote work policy for devs", height=200)
    generate_btn = st.button("Generate Rule Book", type="primary", use_container_width=True)

if generate_btn:
    if not custom_input:
        st.error("Please enter requirements.")
    else:
        with st.spinner("Generating compliant rules..."):
            rules, audit, sources = generate_compliant_rules(industry, custom_input)
            
        with col_output:
            tab1, tab2, tab3 = st.tabs(["📜 Rule Book", "🔍 Compliance Audit", "📚 Sources"])
            with tab1:
                st.markdown(rules)
                st.download_button("Download Markdown", rules, file_name="rulebook.md")
            with tab2:
                st.markdown(audit)
            with tab3:
                for s in sources:
                    st.info(f"Page {s['page']}: {s['content'][:300]}...")