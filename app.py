import streamlit as st
import time
from utils.auth import Authentication

# Global Page Config for the Login Screen
st.set_page_config(
    page_title="KanunMitra - Legal Assistance Portal",
    page_icon="⚖️",
    layout="wide"
)

auth = Authentication()
is_logged_in = auth.check_session()

# Logic to hide sidebar on login
if not is_logged_in:
    st.markdown("<style>[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
    


def main():
    if is_logged_in:
        # Redirect to the main functional page after login
        st.switch_page("pages/Legal_Assistant.py")
    else:
        # Login Form logic
        left_col, right_col = st.columns([2, 1])
        with right_col:
            st.markdown('<h1 style="text-align: center; color: #ffffff;">⚖️ KanunMitra</h1>', unsafe_allow_html=True)
            st.markdown('<h3 style="text-align: center; color: #ffffff;">Legal Assistance Portal</h3>', unsafe_allow_html=True)
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username", key="login_username")
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
                if st.form_submit_button("LOGIN", use_container_width=True, type="primary"):
                    user = auth.authenticate(username, password)
                    if user:
                        st.session_state['user'] = user
                        st.success(f"Welcome, {user['username']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials!")

    with left_col:
        try:
            st.image("images/RobotLaw.png", width=600)
        except:
            st.info("Legal AI System Ready")

if __name__ == "__main__":
    main()