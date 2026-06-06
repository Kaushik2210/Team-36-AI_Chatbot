import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class EnterpriseDataPipeline:
    def __init__(self):
        # Local, open-source high-performance text embedding model
        # Downloads automatically on its first run to your cache folder
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.vector_store = None

    def build_vector_store(self, file_paths: list):
        """Accepts temporary disk storage paths, builds a FAISS index, and returns a retriever."""
        all_docs = []
        for path in file_paths:
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
        
        if not all_docs:
            return None
            
        # Standard chunk layout optimization for technical data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = text_splitter.split_documents(all_docs)
        
        # Build local FAISS Vector store matrix
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return self.vector_store.as_retriever(search_kwargs={"k": 3})