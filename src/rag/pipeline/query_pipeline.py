from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os

class QueryPipeline:
    """
    Orchestrates the retrieval-augmented generation (RAG) process.
    This class connects the vector store to the LLM to provide 
    context-aware answers to user queries.
    """
    def __init__(self):
        # Configuration for GitHub Marketplace
        self.github_url = "https://models.inference.ai.azure.com"
        self.token = os.environ.get("GITHUB_TOKEN")

        # Use GitHub for Embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.token,
            base_url=self.github_url
        )
        
        self.db_path = os.path.join("artifacts", "rag", "faiss_index")

        if not os.path.exists(self.db_path):
            self.vector_db = None
        else:
            self.vector_db = FAISS.load_local(
                self.db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )

    def ask(self, question):
        """
        Processes a user question through the modern LCEL RAG chain.
        """
        if not self.vector_db:
            return "Please upload and process a document first!"

        try:
            # Initialize LLM
            llm = ChatOpenAI(
                model="gpt-4o-mini", 
                api_key=self.token,
                base_url=self.github_url,
                temperature=0
            )

            # 1. Define the System Prompt
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Keep the answer concise.\n\n"
                "{context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # 2. Create the internal chain that handles document "stuffing"
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)

            # 3. Create the final retrieval chain
            # This replaces RetrievalQA.from_chain_type
            rag_chain = create_retrieval_chain(
                self.vector_db.as_retriever(search_kwargs={"k": 3}), 
                combine_docs_chain
            )

            # 4. Invoke the chain using the new keys ('input' and 'answer')
            response = rag_chain.invoke({"input": question})
            
            return response.get("answer", "I'm sorry, I couldn't find an answer.")

        except Exception as e:
            return f"An Error occurred during processing: {str(e)}"
