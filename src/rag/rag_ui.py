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

    # --- Sidebar: Document Management ---
    with st.sidebar:
        st.subheader("Document Control")
        uploaded_file = st.file_uploader('Upload_PDF', type = "pdf")
        
        if st.button('Process Document'):
            if uploaded_file:
                with st.spinner("Processing PDF.."):
                    try:

                        # Ensure temporary data directory exists
                        os.makedirs("data", exist_ok = True)
                        temp_path = os.path.join('data', uploaded_file.name)

                        # Persist uploaded file to local disk for processing
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Execute RAG Pipeline: Ingestion -> Transformation -> Vectorization
                        docs = DataIngestion().load_pdf(temp_path)
                        chunks = DataTransformation().split_text(docs)
                        VectorStoreManager().create_store(chunks)
                    
                        # Reset chat state on new document upload for consistency
                        st.session_state.messages = []
                        st.success("Document Indexed! Ready to chat!")

                    except Exception as e:
                        st.error(f"Failed to process document: {str(e)}")

            else:
                st.warning("Please upload a file before processing.")

    
    # --- Chat Interface Logic ---

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if prompt := st.chat_input("Ask something about the document"):
        # Append user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # Initialize pipeline and fetch response
                # Note: QueryPipeline handles its own FAISS loading logic
                pipeline = QueryPipeline()
                answer = pipeline.ask(prompt)

                st.markdown(answer)

                # Update chat history state
                st.session_state.messages.append({"role": "assistant",
                "content": answer})

