import os
import time
import streamlit as st
import tempfile
import pandas as pd
from datetime import datetime
from core.system import ResumeRelevanceSystem
from core.models import JobDescription


# =========================
# Custom CSS for Premium Responsive UI
# =========================
def load_css():
    st.markdown("""
    <style>
    /* Global background */
    .main {
        background: #C0C0C0;
        background-attachment: fixed;
        min-height: 100vh;
        color: 07070D; 
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: CF472B !important;
    }

    h1 {
        font-size: 4rem !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }

    h2 {
        font-size: 3rem !important;
        font-weight: 600;
        margin-top: 1.5rem;
    }

    h3 {
        font-size: 2rem !important;
        font-weight: 500;
        margin-top: 1rem;
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        border: 2px solid #d1d5db;
        border-radius: 12px;
        padding: 0.75rem;
        font-size: 1rem;
        background: #ffffff;
        color: #000000;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }

    /* Sidebar Navigation */
    .css-1d391kg {
        background: 62B5A5;
        padding-top: 1rem;
    }

    .css-1d391kg h2 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff !important;
    }

    /* Modern Hamburger Icon */
    [data-testid="stSidebarNav"]::before {
        content: "☰ Navigation";
        font-size: 1.5rem;
        font-weight: 600;
        color: #fff;
        display: block;
        padding: 1rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        font-size: 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46a1 100%);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

    /* Responsive */
    @media (max-width: 768px) {
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.4rem !important; }
        .stButton > button { width: 100%; font-size: 0.95rem; }
    }
    </style>
    """, unsafe_allow_html=True)


# =========================
# Streamlit App
# =========================
def run_streamlit_app():
    st.set_page_config(
        page_title="Resume Relevance Check System",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css()

    # Header
    st.markdown("""
    <h1>🎯 Automated Resume Relevance Check System</h1>
    <p style="text-align: center; color: #4a5568; font-size: 1.2rem;">
        Innomatics Research Labs - Placement Team Dashboard
    </p>
    """, unsafe_allow_html=True)

    # Initialize system
    if "system" not in st.session_state:
        st.session_state.system = ResumeRelevanceSystem(use_llm=False)

    # Sidebar Navigation
    with st.sidebar:
        page = st.radio(
            "",
            ["📋 Upload JD", "📊 Evaluate Resumes", "🏆 View Shortlist", "📈 Analytics", "🔍 Advanced Search"],
            label_visibility="collapsed"
        )

    # Upload JD Page
    if page == "📋 Upload JD":
        st.subheader("📋 Upload Job Description")

        with st.form("jd_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                company = st.text_input("🏢 Company Name", placeholder="Enter company name")
                location = st.text_input("📍 Location", placeholder="City, State")
            with col2:
                role = st.text_input("💼 Role Title", placeholder="e.g., Software Engineer")
                experience = st.text_input("📊 Experience Required", placeholder="e.g., 2-4 years")

            jd_text = st.text_area(
                "📝 Job Description",
                height=300,
                placeholder="Paste the complete job description here..."
            )

            submit = st.form_submit_button("Process JD", type="primary")  # removed icon

        if submit:
            if jd_text:
                jd = st.session_state.system.process_job_description(
                    jd_text, company or "Unknown", location or "Unknown"
                )
                st.success(f"✅ JD processed successfully! Job ID: **{jd.job_id}**")
            else:
                st.error("⚠️ Please enter a job description")

    # Evaluate Resumes Page
    elif page == "📊 Evaluate Resumes":
        st.subheader("📊 Evaluate Resumes")

        jobs = st.session_state.system.db_manager.get_all_jobs()
        if not jobs:
            st.warning("⚠️ No jobs found. Please upload a JD first.")
        else:
            job_options = {f"{j['role_title']} - {j['company']}": j for j in jobs}
            selected_job_name = st.selectbox("🎯 Select Target Job", list(job_options.keys()))
            selected_job = job_options[selected_job_name]

            st.markdown("### 📂 Upload Resume Files")
            uploaded_files = st.file_uploader("", type=["pdf", "docx"], accept_multiple_files=True)

            if uploaded_files and st.button("🚀 Start Evaluation", type="primary"):
                all_results = []
                for file in uploaded_files:
                    temp_path = os.path.join(tempfile.gettempdir(), file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())

                    try:
                        resume = st.session_state.system.process_resume(temp_path)
                        jd = JobDescription(**selected_job)
                        evaluation = st.session_state.system.scoring_engine.calculate_relevance_score(resume, jd)
                        st.session_state.system.db_manager.save_evaluation(evaluation)

                        all_results.append({
                            "Name": resume.candidate_name,
                            "Email": resume.email,
                            "Score": evaluation.relevance_score,
                            "Verdict": evaluation.verdict.value,
                            "Match %": f"{evaluation.relevance_score:.1f}%",
                        })
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                if all_results:
                    st.success("✅ Evaluation complete!")
                    df = pd.DataFrame(all_results)
                    st.dataframe(df, use_container_width=True)


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    run_streamlit_app()
