from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document 


class DataTransformation:
    """
    Handles the transformation of raw documents into smaller, manageable chunks.
    This step is critical for ensuring the LLM receives relevant context within 
    its token limits.
    """
    def split_text(self, 
        docs: List[Document],
        chunk_size = 1000,
        chunk_overlap = 200):
        """
        Splits a list of Documents into smaller chunks using recursive character splitting.
        """
        # Instantiate splitter to keep semantically related text together 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            add_start_index = True,
            length_function = len
        )
        return text_splitter.split_documents(docs)