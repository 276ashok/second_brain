"""
Configuration Module
Manages application configuration and environment variables.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
TOP_K_CHUNKS: int = 5

GROQ_MODEL: str = "llama-3.3-70b-versatile"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
TEMPERATURE: float = 0.7

VECTORSTORE_PATH: str = "./vectorstore"
VECTORSTORE_TYPE: str = "chromadb"

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")


def validate_config() -> bool:
    """
    Validate that all required configuration is present.

    Note: GROQ_API_KEY is only required for LLM (ChatGroq),
    not for embeddings which use local sentence-transformers models.

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment. "
            "Note: This is only needed for the LLM (Groq), not for embeddings."
        )
    return True


def get_config_summary() -> dict:
    """
    Get a summary of current configuration.

    Returns:
        dict: Configuration summary
    """
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k_chunks": TOP_K_CHUNKS,
        "groq_model": GROQ_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "temperature": TEMPERATURE,
        "vectorstore_path": VECTORSTORE_PATH,
        "vectorstore_type": VECTORSTORE_TYPE,
        "api_key_set": bool(GROQ_API_KEY),
    }
