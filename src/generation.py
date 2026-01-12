"""
Generation Module
Handles LLM setup and answer generation.
"""

from typing import List, Tuple, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    TEMPERATURE,
    validate_config,
)


def initialize_llm() -> ChatGroq:
    """
    Initialize the LLM (Groq).
    
    Returns:
        ChatGroq instance
        
    Raises:
        ValueError: If API key is not configured
    """
    validate_config()
    
    try:
        llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=TEMPERATURE,
        )
        return llm
    except Exception as e:
        raise RuntimeError(f"Error initializing LLM: {str(e)}")


def create_qa_prompt(include_history: bool = True) -> ChatPromptTemplate:
    """
    Create the prompt template for question answering.
    
    Args:
        include_history: Whether to include conversation history in the prompt
    
    Returns:
        ChatPromptTemplate instance
    """
    system_template = """You are a helpful AI assistant that answers questions based on the provided context from documents.

Instructions:
- Answer questions using ONLY the information provided in the context below
- If the context does not contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."
- Be concise and accurate
- Cite specific sources when possible (mention the document name and page number if available)
- If asked about something not in the context, politely decline and suggest the user check the uploaded documents
- When answering follow-up questions, you can reference previous questions and answers in the conversation history
- If the user asks about previous questions or conversation, use the conversation history to provide context

Context:
{context}"""

    human_template = """Previous conversation:
{chat_history}

Current question: {question}

Answer based on the context provided above:"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ]
    
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt


def format_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    """
    Format chat history into a readable string for the prompt.
    
    Args:
        chat_history: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Formatted string representation of chat history
    """
    if not chat_history:
        return "No previous conversation."
    
    formatted_history = []
    for msg in chat_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            formatted_history.append(f"User: {content}")
        elif role == "assistant":
            formatted_history.append(f"Assistant: {content}")
    
    return "\n".join(formatted_history)


def generate_answer(
    context: str,
    question: str,
    llm: Optional[ChatGroq] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Generate an answer to a question using the provided context and conversation history.
    
    Args:
        context: Formatted context string from retrieved chunks
        question: User's question
        llm: LLM instance (optional, will create if not provided)
        chat_history: Previous conversation history (list of dicts with 'role' and 'content')
        
    Returns:
        Generated answer string
        
    Raises:
        ValueError: If context or question is empty
        RuntimeError: If LLM generation fails
    """
    if not context or not context.strip():
        raise ValueError("Context cannot be empty")
    
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    if llm is None:
        llm = initialize_llm()
    
    try:
        history_text = "No previous conversation."
        if chat_history:
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            history_text = format_chat_history(recent_history)
        
        prompt = create_qa_prompt(include_history=True)
        chain = prompt | llm
        
        chain_input = {
            "context": context,
            "question": question,
            "chat_history": history_text,
        }
        
        response = chain.invoke(chain_input)
        
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)
        
        return answer.strip()
    
    except Exception as e:
        raise RuntimeError(f"Error generating answer: {str(e)}")


def format_response(
    answer: str,
    chunks_with_scores: List[Tuple[Document, float]],
    include_scores: bool = True,
) -> Dict[str, Any]:
    """
    Format the complete response with answer and sources.
    
    Args:
        answer: Generated answer string
        chunks_with_scores: List of (Document, score) tuples used for context
        include_scores: Whether to include relevance scores
        
    Returns:
        Dictionary with answer and source information
    """
    sources = []
    
    for doc, score in chunks_with_scores:
        source_info = {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
        }
        
        if include_scores:
            source_info["relevance_score"] = round(score, 4)
        
        sources.append(source_info)
    
    return {
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources),
    }


def answer_question(
    question: str,
    context: str,
    chunks_with_scores: List[Tuple[Document, float]],
    llm: Optional[ChatGroq] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    include_scores: bool = True,
) -> Dict[str, Any]:
    """
    Complete question answering pipeline with conversation memory.
    
    This is a convenience function that combines answer generation and formatting.
    
    Args:
        question: User's question
        context: Formatted context string
        chunks_with_scores: List of (Document, score) tuples
        llm: LLM instance (optional)
        chat_history: Previous conversation history (list of dicts with 'role' and 'content')
        include_scores: Whether to include relevance scores
        
    Returns:
        Dictionary with answer and source information
    """
    answer = generate_answer(context, question, llm, chat_history)
    response = format_response(answer, chunks_with_scores, include_scores)
    return response
