"""
Utility Functions
Helper functions for the RAG system.
"""

import os
from pathlib import Path
from typing import List, Optional


def ensure_directory_exists(directory: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def is_valid_file_format(file_path: str, allowed_formats: List[str] = None) -> bool:
    """
    Check if a file has a valid format.
    
    Args:
        file_path: Path to the file
        allowed_formats: List of allowed file extensions (with or without dot)
        
    Returns:
        True if format is valid, False otherwise
    """
    if allowed_formats is None:
        allowed_formats = [".pdf", ".txt", ".docx", ".doc"]
    
    file_ext = Path(file_path).suffix.lower()
    
    normalized_formats = [
        f".{fmt.lstrip('.')}" for fmt in allowed_formats
    ]
    
    return file_ext in normalized_formats


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    
    import re
    text = re.sub(r" +", " ", text)
    
    return text.strip()


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"
