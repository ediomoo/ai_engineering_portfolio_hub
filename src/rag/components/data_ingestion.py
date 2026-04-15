from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List
import os

class DataIngestion:
    """
    Handles the loading and validation of PDF documents for the RAG pipeline.
    """

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Loads a PDF file and converts it into LangChain Document objects.
        """

        # Ensure file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Ensure file is a PDF
        if not file_path.lower().endswith(".pdf"):
            raise ValueError(f"Unsupported file format. Please provide a PDF.")

        # Load and return document content
        loader = PyPDFLoader(file_path)
        return loader.load()

