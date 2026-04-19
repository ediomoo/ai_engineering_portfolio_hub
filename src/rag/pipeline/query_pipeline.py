from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chains import RetrievalQA
import os

class QueryPipeline:
    """
    Orchestrates the retrieval-augmented generation (RAG) process.
    This class connects the vector store to the LLM to provide 
    context-aware answers to user queries.
    """
    def __init__(self):
        """
        Initializes the pipeline with OpenAI embeddings and attempts 
        to load the existing local vector store.
        """

        # Configuration for GitHub Marketplace
        self.github_url = "https://models.inference.ai.azure.com"
        self.token = os.environ.get("GITHUB_TOKEN")

        # Use GitHub for Embeddings (free)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.token,
            base_url=self.github_url
        )
        
        self.db_path = os.path.join("artifacts", "rag", "faiss_index")

        # Ensure that FAISS Index file exists before attempting to load it
        if not os.path.exists(self.db_path):
            self.vector_db = None
        else:
            self.vector_db = FAISS.load_local(self.db_path, 
            self.embeddings, 
            allow_dangerous_deserialization = True)

    def ask(self, question):
        """
        Processes a user question through the RAG chain.
        """
        
        # Ensure a document is uploaded for processing
        if not self.vector_db:
            return "Please upload and process a document first!"

        try:
            # Initialize LLM with low temperature for factual consistency
            llm = ChatOpenAI(
                model="gpt-4o-mini", 
                api_key=self.token,
                base_url=self.github_url,
                temperature=0
            )
            
            # Define the RetrievalQA chain
            # 'stuff' chain type takes all retrieved documents and passes them to the LLM
            qa_chain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type = "stuff",
            retriever = self.vector_db.as_retriever(search_kwargs = {"k": 3})
            )

            # Invoke the chain and return the result
            response = qa_chain.invoke({"query": question})
            return response.get("result", "I'm sorry, I couldn't find an answer.")


        except Exception as e:
            # Return the error message so it can be deployed in the streamlit UI
            return f"An Error occured during processing: {str(e)}"
