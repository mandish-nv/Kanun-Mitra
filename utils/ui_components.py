import streamlit as st
from utils.auth import Authentication

def init_page(title):
    st.set_page_config(page_title=f"⚖️ Kanun Mitra - {title}", layout="wide")
    
    auth = Authentication()
    if not auth.check_session():
        st.warning("Please login to access this page.")
        st.stop() # Stops execution for unauthenticated users

    user = auth.get_current_user()
    
    with st.sidebar:
        st.title("⚖️ Kanun Mitra")
        st.markdown(f"**User:** {user['username']}")
        st.markdown(f"**Role:** `{user['role'].upper()}`")
        
        if st.button("Logout", type="secondary", use_container_width=True):
            auth.logout()
            st.rerun()
        st.divider()
    
    return user