import streamlit as st

def show_dashboard():
    """Display the main dashboard"""
    # ---------- CUSTOM CSS ----------
    st.markdown("""
    <style>
        .card {
            padding: 24px;
            border-radius: 14px;
            background: white;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
            height: 250px;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            margin-bottom: 10px;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.12);
            border-color: #0d6efd;
        }
        .card h3 { color: #2c3e50; margin-bottom: 12px; font-size: 1.2rem; }
        .card p { color: #6b7280; line-height: 1.4; font-size: 0.9rem; }
        .card-icon { font-size: 28px; margin-bottom: 10px; }
        .subtitle { color: #6b7280; font-size: 16px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("⚖️ KanunMitra")
    st.markdown("<p class='subtitle'>Your AI-powered platform for legal assistance and document analysis.</p>", unsafe_allow_html=True)
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    cards = [
        {"title": "Legal Assistant", "desc": "Chat with your PDF knowledge base.", "icon": "💬", "page": "pages/LegalAssist.py"},
        {"title": "Rule Generator", "desc": "Generate compliant industry rule books.", "icon": "⚖️", "page": "pages/RuleGenerator.py"},
        {"title": "Document Ingestion", "desc": "Upload and process new PDFs into the system.", "icon": "📄", "page": "pages/PDFAnalysis.py"}
    ]
    
    for i, col in enumerate([col1, col2, col3]):
        with col:
            card = cards[i]
            st.markdown(f"""
            <div class="card">
                <div class="card-icon">{card['icon']}</div>
                <h3>{card['title']}</h3>
                <p>{card['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {card['title']} →", key=f"btn_{i}", use_container_width=True):
                st.switch_page(card['page'])

    st.divider()
    st.caption("⚠️ Informational assistance only. Not a replacement for professional legal advice.")