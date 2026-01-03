import streamlit as st
from rag_query import generate_compliant_rules
import config
from utils.ui_components import init_page

user_info = init_page("Rule Generator")

st.set_page_config(page_title="Rule Generator", layout="wide")

st.title("⚖️ Intelligent Rule Generator")

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

# Initialize session state for persistence
if "generated_rules" not in st.session_state:
    st.session_state.generated_rules = None

# Using a single column layout or a container to ensure tabs appear below the button
st.header("Parameters")
industry = st.selectbox("Industry", options=list(config.INDUSTRY_MANDATORY_RULES.keys()))
custom_input = st.text_area("Specific Requirements", placeholder="e.g. Remote work policy for devs", height=200)

generate_btn = st.button("Generate Rule Book", type="primary", use_container_width=True)

# 1. Logic for generation
if generate_btn:
    if not custom_input:
        st.error("Please enter requirements.")
    else:
        with st.spinner("Generating compliant rules..."):
            rules, audit, sources = generate_compliant_rules(industry, custom_input)
            # Store results in session state to prevent loss on rerun
            st.session_state.generated_rules = {
                "rules": rules,
                "audit": audit,
                "sources": sources
            }

st.divider()

# 2. Display logic (Placed below the button)
if st.session_state.generated_rules:
    res = st.session_state.generated_rules
    
    # Tabs now appear below the generation button in the main flow
    tab1, tab2, tab3 = st.tabs(["📜 Rule Book", "🔍 Compliance Audit", "📚 Sources"])
    
    with tab1:
        st.markdown(res["rules"])
        # Download button will trigger a rerun, but session_state persists the data
        st.download_button(
            label="Download Markdown",
            data=res["rules"],
            file_name="rulebook.md",
            mime="text/markdown"
        )
        
    with tab2:
        st.markdown(res["audit"])
        
    with tab3:
        for s in res["sources"]:
            st.info(f"Page {s['page']}: {s['content'][:300]}...")