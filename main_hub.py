import streamlit as st
from  dotenv import load_dotenv
import os


# Ensures OPENAI Key is loaded regardless of environment
if os.path.exists(".env"):
    # --- Load Environment variables (API Keys for RAG/Agents) ---
    load_dotenv()

# Use st.secrets if if on cloud, else fallback to environment variables
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN"))
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY"))


# --- Page Configuration ---
st.set_page_config(
    page_title = "AI Engineering Portfolio",
    page_icon = "🚀",
    layout = "wide"
)

# Ensure that switching from RAG tool to predictor doesnt cause memory bloat
if "last_selection" not in st.session_state:
    st.session_state.last_selection = "🏠 Home / About Me"


# --- Sidebar Navigation ---
st.sidebar.title("🚀 Project Navigator")
st.sidebar.markdown("---")

selection = st.sidebar.radio(
    "Select a Project:", [
         "🏠 Home / About Me",
        "💰 Loan Approval Prediction",
        "🏥 Medical Insurance Prediction",
        "🕵️‍♂️ PDF Investigator (RAG)",
        "📈 Agentic Market Analyst"
    ]
)

# Trigger 'clean state' if user switches projects
if selection != st.session_state.last_selection:
    st.session_state.last_selection = selection
    # Optional: st.cache_resource.clear() if memory becomes an issue
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("**Tech Stack:** Python, XGBoost, LangChain, OpenAI")


# --- Home Page Logic ---
if selection == "🏠 Home / About Me":
    st.title("Welcome to my AI Engineering Portfolio")
    st.markdown("""
    This unified hub showcases my expertise across the **AI development lifecycle**, 
    from classic Machine Learning to state-of-the-art Agentic Systems.
    
    ### 📂 Featured Projects:
    1. **Loan Approval Prediction**: A classification pipeline using XGBoost and MLOps principles.
    2. **Medical Insurance Prediction**: A regression-based cost estimator with clean data modularity.
    3. **PDF Investigator (RAG)**: A GenAI system that chats with your local documents using FAISS and OpenAI.
    4. **Agentic Market Analyst**: An autonomous AI agent capable of researching market trends.
    """)

# --- Project Routing ---
# Using 'Lazy Imports' inside the blocks to keep the initial app load fast
elif selection == "💰 Loan Approval Prediction":
    from src.loan.predict_ui import run_loan_ui
    run_loan_ui()

elif selection == "🏥 Medical Insurance Prediction":
    from src.insurance.predict_ui import run_insurance_ui
    run_insurance_ui()

elif selection == "🕵️‍♂️ PDF Investigator (RAG)":
    from src.rag.rag_ui import run_rag_ui
    run_rag_ui()

elif selection == "📈 Agentic Market Analyst":
    from src.agent.agent_ui import run_agent_ui
    run_agent_ui()