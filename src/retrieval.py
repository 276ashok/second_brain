"""
Retrieval Module
Handles similarity search and context retrieval from vector store.
"""

from typing import List, Tuple, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import TOP_K_CHUNKS
from src.document_processor import initialize_embeddings, initialize_vectorstore


def initialize_vectorstore_for_retrieval(
    collection_name: str = "documents",
) -> Chroma:
    """
    Initialize vector store for retrieval operations.
    
    Args:
        collection_name: Name of the collection in the vector store
        
    Returns:
        Chroma vector store instance
        
    Raises:
        RuntimeError: If vector store cannot be initialized
    """
    try:
        embeddings = initialize_embeddings()
        vectorstore = initialize_vectorstore(embeddings, collection_name)
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"Error initializing vector store for retrieval: {str(e)}")


def retrieve_relevant_chunks(
    query: str,
    vectorstore: Chroma,
    k: int = TOP_K_CHUNKS,
) -> List[Tuple[Document, float]]:
    """
    Retrieve relevant document chunks for a query.
    
    Args:
        query: User query string
        vectorstore: Vector store instance
        k: Number of chunks to retrieve (default: TOP_K_CHUNKS)
        
    Returns:
        List of tuples containing (Document, similarity_score)
        
    Raises:
        ValueError: If query is empty or vector store is empty
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        if not results:
            raise ValueError("No documents found in vector store. Please upload documents first.")
        
        return results
    
    except Exception as e:
        if "not found" in str(e).lower() or "empty" in str(e).lower():
            raise ValueError("No documents found in vector store. Please upload documents first.")
        raise RuntimeError(f"Error retrieving chunks: {str(e)}")


def format_context(
    chunks_with_scores: List[Tuple[Document, float]],
    include_scores: bool = False,
) -> str:
    """
    Format retrieved chunks into a context string for LLM input.
    
    Args:
        chunks_with_scores: List of (Document, score) tuples
        include_scores: Whether to include relevance scores in output
        
    Returns:
        Formatted context string
    """
    if not chunks_with_scores:
        return ""
    
    context_parts = []
    
    for idx, (doc, score) in enumerate(chunks_with_scores, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        chunk_idx = doc.metadata.get("chunk_index", idx)
        
        context_entry = f"[Chunk {chunk_idx}]"
        if include_scores:
            context_entry += f" (Relevance: {score:.4f})"
        context_entry += f"\nSource: {source}"
        if page != "N/A":
            context_entry += f", Page: {page}"
        context_entry += f"\n{doc.page_content}\n"
        
        context_parts.append(context_entry)
    
    return "\n---\n".join(context_parts)


def get_relevant_context(
    query: str,
    vectorstore: Chroma,
    k: int = TOP_K_CHUNKS,
    include_scores: bool = False,
) -> Tuple[str, List[Tuple[Document, float]]]:
    """
    Get formatted context for a query.
    
    This is a convenience function that combines retrieval and formatting.
    
    Args:
        query: User query string
        vectorstore: Vector store instance
        k: Number of chunks to retrieve
        include_scores: Whether to include relevance scores
        
    Returns:
        Tuple of (formatted_context_string, chunks_with_scores)
    """
    chunks_with_scores = retrieve_relevant_chunks(query, vectorstore, k)
    context = format_context(chunks_with_scores, include_scores)
    return context, chunks_with_scores
