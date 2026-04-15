import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    """
    Manages the generation and local persistence of FAISS vector databases.
    This class abstracts the embedding process and ensures data is stored 
    correctly for efficient retrieval.
    """
    def __init__(self):
        """
        Initializes the manager and validates the required API environment.
        """
        self.db_path = os.path.join('artifacts', "rag", "faiss_index")
        
        # Immediate validation of API keys to prevent downstream failures
        # Validate GitHub Token 
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found in environment variables.")

        # Optimization: Use GitHub Marketplace for Embeddings
        # Crucial: This model name and base_url must match your QueryPipeline!
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.token,
            base_url="https://models.inference.ai.azure.com"
        )



    def create_store(self, chunks):
        """
        Generates embeddings for text chunks and persists them to a local FAISS index.
        """
        try:
            # Ensure the directory structure exists before attempting to save
            os.makedirs(os.path.dirname(self.db_path), exist_ok = True)
            
            # Create vector database using OpenAI's embedding model
            vector_db = FAISS.from_documents(chunks, self.embeddings)
            
            # Save the index locally to the artifacts directory
            vector_db.save_local(self.db_path)
            
            return self.db_path
        
        except Exception as e:
            # Re-raise with context to aid in higher-level UI calls
            raise RuntimeError(f"Failed to create vector store: {str(e)}")


    