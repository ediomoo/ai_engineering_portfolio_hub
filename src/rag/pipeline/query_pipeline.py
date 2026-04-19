from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

class QueryPipeline:
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

            # 1. Define the Prompt Template
            template = """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Keep the answer concise.

            Context: {context}

            Question: {question}
            
            Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)

            # 2. Build the RAG Chain using the "Pipe" (|) logic
            # This handles retrieval, prompting, and LLM call in one go
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}

                | prompt
                | llm
                | StrOutputParser()
            )

            # 3. Invoke the chain
            response = rag_chain.invoke(question)
            
            return response

        except Exception as e:
            return f"An Error occurred during processing: {str(e)}"
