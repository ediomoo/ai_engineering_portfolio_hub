import streamlit as st
import os 
from src.rag.components.data_transformation import DataTransformation
from src.rag.components.data_ingestion import DataIngestion
from src.rag.components.vector_store import VectorStoreManager
from src.rag.pipeline.query_pipeline import QueryPipeline

def run_rag_ui():
    """
    Renders the UI for the PDF Investigator (RAG) project.
    Orchestrates file uploads, document processing, and the chat interface.
    """
    st.header("🕵️‍♂️ PDF Investigator (RAG)")
    st.markdown("Chat with your document using context-aware AI.")

    # --- Main Page: Document Management (Moved from Sidebar) ---
    # Using an expander makes it prominent but allows users to hide it after indexing
    with st.expander("📂 **Step 1: Upload and Index your Document**", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader('Select a PDF file', type="pdf", label_visibility="collapsed")
        
        with col2:
            process_btn = st.button('Process Document', use_container_width=True)

        if process_btn:
            if uploaded_file:
                with st.spinner("Processing PDF..."):
                    try:
                        # Ensure temporary data directory exists
                        os.makedirs("data", exist_ok=True)
                        temp_path = os.path.join('data', uploaded_file.name)

                        # Persist uploaded file
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Execute RAG Pipeline
                        docs = DataIngestion().load_pdf(temp_path)
                        chunks = DataTransformation().split_text(docs)
                        VectorStoreManager().create_store(chunks)
                    
                        st.session_state.messages = []
                        st.success("✅ Document Indexed! You can now ask questions below.")

                    except Exception as e:
                        st.error(f"❌ Failed to process: {str(e)}")
            else:
                st.warning("⚠️ Please upload a file first.")

    st.divider() # Visual break between upload and chat

    # --- Chat Interface Logic ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if prompt := st.chat_input("Ask something about the document"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pipeline = QueryPipeline()
                answer = pipeline.ask(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
