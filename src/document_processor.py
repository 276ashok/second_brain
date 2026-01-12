"""
Document Processing Module
Handles document loading, text splitting, embeddings, and vector storage.
"""

import os
from typing import List, Optional, Tuple
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
)


def load_document(file_path: str) -> List[Document]:
    """
    Load a document from file path.
    
    Supports PDF, TXT, and DOCX formats.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of Document objects with metadata
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = Path(file_path).suffix.lower()
    file_name = Path(file_path).name
    
    try:
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                "Supported formats: PDF, TXT, DOCX"
            )
        
        documents = loader.load()
        
        for doc in documents:
            if "source" not in doc.metadata:
                doc.metadata["source"] = file_name
            doc.metadata["file_path"] = file_path
        
        return documents
    
    except Exception as e:
        raise RuntimeError(f"Error loading document {file_path}: {str(e)}")


def split_text(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks with metadata preservation.
    
    Args:
        documents: List of Document objects to split
        
    Returns:
        List of Document chunks with metadata
    """
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    
    chunks = text_splitter.split_documents(documents)
    
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 0
    
    return chunks


def initialize_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize embeddings model using sentence-transformers (local, no API key needed).
    
    Returns:
        HuggingFaceEmbeddings instance
        
    Raises:
        RuntimeError: If embeddings cannot be initialized
    """
    try:
        model_name = EMBEDDING_MODEL if EMBEDDING_MODEL else "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Error initializing embeddings: {str(e)}")


def initialize_vectorstore(
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    collection_name: str = "documents",
) -> Chroma:
    """
    Initialize or load existing vector store.
    
    Args:
        embeddings: Embeddings model instance (optional, will create if not provided)
        collection_name: Name of the collection in the vector store
        
    Returns:
        Chroma vector store instance
    """
    if embeddings is None:
        embeddings = initialize_embeddings()
    
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    
    try:
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"Error initializing vector store: {str(e)}")


def process_and_store(
    file_path: str,
    vectorstore: Optional[Chroma] = None,
    collection_name: str = "documents",
) -> Tuple[Chroma, int]:
    """
    Process a document and store it in the vector database.
    
    Args:
        file_path: Path to the document file
        vectorstore: Existing vector store instance (optional)
        collection_name: Name of the collection
        
    Returns:
        Tuple of (vectorstore, number_of_chunks)
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not supported
    """
    documents = load_document(file_path)
    
    if not documents:
        raise ValueError(f"No content extracted from {file_path}")
    
    chunks = split_text(documents)
    
    if not chunks:
        raise ValueError(f"No chunks created from {file_path}")
    
    if vectorstore is None:
        embeddings = initialize_embeddings()
        vectorstore = initialize_vectorstore(embeddings, collection_name)
    
    try:
        vectorstore.add_documents(chunks)
    except Exception as e:
        raise RuntimeError(f"Error storing documents in vector store: {str(e)}")
    
    return vectorstore, len(chunks)
