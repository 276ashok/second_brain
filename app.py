"""
Main Streamlit Application
second_brain frontend.
"""

import streamlit as st
import os
import time
from pathlib import Path
from typing import List, Dict, Any

try:
    from src.document_processor import process_and_store, initialize_vectorstore
    from src.retrieval import initialize_vectorstore_for_retrieval, get_relevant_context
    from src.generation import initialize_llm, answer_question
    from src.config import validate_config, get_config_summary
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)


st.set_page_config(
    page_title="second brain",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "llm" not in st.session_state:
    st.session_state.llm = None


@st.cache_resource
def _get_vectorstore():
    """Cached vectorstore initialization."""
    return initialize_vectorstore_for_retrieval()

@st.cache_resource
def _get_llm():
    """Cached LLM initialization."""
    return initialize_llm()

def initialize_components():
    """Initialize vector store and LLM if not already initialized (lazy loading with caching)."""
    try:
        if st.session_state.vectorstore is None and st.session_state.uploaded_documents:
            st.session_state.vectorstore = _get_vectorstore()
        
        if st.session_state.llm is None:
            st.session_state.llm = _get_llm()
        
        return True
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return False


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to data/uploads directory."""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


def sidebar():
    """Create sidebar with settings and information."""
    with st.sidebar:
        st.title("Settings")
        
        with st.expander("Configuration", expanded=False):
            try:
                if DEPENDENCIES_OK:
                    config = get_config_summary()
                    st.json(config)
            except Exception as e:
                st.error(f"Error loading config: {str(e)}")
        
        if DEPENDENCIES_OK:
            try:
                from src.config import GROQ_API_KEY
                if GROQ_API_KEY:
                    st.success("API Key configured")
                else:
                    st.warning("API Key not set")
            except:
                pass
        
        st.subheader("Uploaded Documents")
        if st.session_state.uploaded_documents:
            for doc_info in st.session_state.uploaded_documents:
                with st.container():
                    st.text(f"{doc_info['name']}")
                    st.caption(f"Chunks: {doc_info['chunks']} | {doc_info['date']}")
        else:
            st.info("No documents uploaded yet.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                time.sleep(0.5)
                st.rerun()
        
        with col2:
            if st.button("Clear All", type="secondary"):
                st.session_state.uploaded_documents = []
                st.session_state.chat_history = []
                st.session_state.vectorstore = None
                import shutil
                vectorstore_path = Path("vectorstore")
                if vectorstore_path.exists():
                    shutil.rmtree(vectorstore_path)
                st.success("All data cleared!")
                time.sleep(0.5)
                st.rerun()
        
        st.divider()
        
        st.subheader("Instructions")
        st.markdown("""
        1. **Upload Documents**: Go to the Upload tab and upload PDF, TXT, or DOCX files
        2. **Process Documents**: Click "Process Document" to add them to the knowledge base
        3. **Ask Questions**: Go to the Chat tab and ask questions about your documents
        4. **View Sources**: Each answer includes source citations
        """)


def upload_tab():
    """Upload tab for document processing."""
    st.header("Upload Documents")
    
    st.markdown("""
    Upload and process documents to add them to the knowledge base.
    Supported formats: PDF, TXT, DOCX
    """)
    
    if not DEPENDENCIES_OK:
        st.warning("Please install dependencies first (see main page)")
        return
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "docx"],
        help="Upload a PDF, TXT, or DOCX file to process"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**File:** {uploaded_file.name}\n**Size:** {uploaded_file.size / 1024:.2f} KB")
        
        if st.button("Process Document", type="primary"):
            try:
                validate_config()
                
                with st.spinner("Saving file..."):
                    file_path = save_uploaded_file(uploaded_file)
                
                with st.spinner("Processing document (this may take a moment)..."):
                    start_time = time.time()
                    
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = initialize_vectorstore_for_retrieval()
                    
                    vectorstore, num_chunks = process_and_store(
                        file_path,
                        st.session_state.vectorstore,
                    )
                    
                    st.session_state.vectorstore = vectorstore
                    
                    processing_time = time.time() - start_time
                
                doc_info = {
                    "name": uploaded_file.name,
                    "chunks": num_chunks,
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "file_path": file_path,
                }
                
                if doc_info not in st.session_state.uploaded_documents:
                    st.session_state.uploaded_documents.append(doc_info)
                
                st.success(f"Document processed successfully!")
                st.info(f"**Chunks created:** {num_chunks}\n**Processing time:** {processing_time:.2f} seconds")
                
                st.rerun()
            
            except ValueError as e:
                st.error(f"Configuration Error: {str(e)}")
                st.info("Please make sure you have set your GROQ_API_KEY in the .env file.")
            
            except FileNotFoundError as e:
                st.error(f"File Error: {str(e)}")
            
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                st.exception(e)


def chat_tab():
    """Chat tab for question answering."""
    st.header("Chat with Documents")
    
    if not DEPENDENCIES_OK:
        st.warning("Please install dependencies first (see main page)")
        return
    
    if not st.session_state.uploaded_documents:
        st.warning("No documents uploaded yet. Please upload and process documents in the Upload tab first.")
        st.info("Tip: Go to the Upload tab to add your first document.")
        return
    
    st.markdown("""
    <style>
    div[data-testid="stChatInputContainer"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 20rem !important;
        right: 0 !important;
        z-index: 999 !important;
        background-color: var(--background-color) !important;
        padding: 0.75rem 1rem !important;
        border-top: 1px solid rgba(250, 250, 250, 0.2) !important;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1) !important;
        max-width: calc(100% - 20rem) !important;
    }
    
    .stChatInput {
        position: fixed !important;
        bottom: 0 !important;
        left: 20rem !important;
        right: 0 !important;
        z-index: 999 !important;
        background-color: var(--background-color) !important;
        padding: 0.75rem 1rem !important;
        max-width: calc(100% - 20rem) !important;
    }
    
    div[data-testid="stChatInputContainer"] textarea {
        max-width: 100% !important;
        width: 100% !important;
    }
    
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    .stChatMessage {
        margin-bottom: 1rem !important;
    }
    
    section[data-testid="stMain"] {
        padding-bottom: 100px !important;
    }
    
    @media (max-width: 768px) {
        div[data-testid="stChatInputContainer"],
        .stChatInput {
            left: 0 !important;
            max-width: 100% !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("Sources", expanded=False):
                        for idx, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {idx}:** {source['source']}")
                            if source.get("page") != "N/A":
                                st.caption(f"Page: {source['page']}")
                            if "relevance_score" in source:
                                st.caption(f"Relevance: {source['relevance_score']}")
                            st.text(source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"])
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        if not initialize_components():
            st.error("Failed to initialize components. Please check your configuration.")
            return
        
        previous_history = st.session_state.chat_history.copy()
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question,
        })
        
        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    context, chunks_with_scores = get_relevant_context(
                        user_question,
                        st.session_state.vectorstore,
                    )
                    
                    if not context:
                        st.warning("No relevant context found for this question.")
                        response = {
                            "answer": "I couldn't find relevant information in the uploaded documents to answer this question.",
                            "sources": [],
                            "num_sources": 0,
                        }
                    else:
                        response = answer_question(
                            user_question,
                            context,
                            chunks_with_scores,
                            st.session_state.llm,
                            chat_history=previous_history,
                        )
                    
                    st.markdown(response["answer"])
                    
                    if response["sources"]:
                        with st.expander("Sources", expanded=False):
                            for idx, source in enumerate(response["sources"], 1):
                                st.markdown(f"**Source {idx}:** {source['source']}")
                                if source.get("page") != "N/A":
                                    st.caption(f"Page: {source['page']}")
                                if "relevance_score" in source:
                                    st.caption(f"Relevance: {source['relevance_score']}")
                                st.text(source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"])
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                    })
                
                except ValueError as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
                
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.exception(e)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


def main():
    """Main application function."""
    st.title("second brain")
    st.markdown("Ask questions about your documents using AI-powered retrieval and generation.")
    
    if not DEPENDENCIES_OK:
        st.error("Missing Dependencies")
        st.error(f"Import error: {IMPORT_ERROR}")
        st.warning("""
        **Please install the required dependencies first:**
        
        1. Open a terminal/command prompt
        2. Navigate to this project directory
        3. Run: `pip install -r requirements.txt`
        
        After installing, refresh this page.
        """)
        
        with st.expander("Installation Instructions", expanded=True):
            st.code("""
# For Windows PowerShell:
pip install -r requirements.txt

# Or if you're using a virtual environment:
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
            """, language="bash")
        st.stop()
    
    sidebar()
    
    tab1, tab2 = st.tabs(["Upload", "Chat"])
    
    with tab1:
        upload_tab()
    
    with tab2:
        chat_tab()


if __name__ == "__main__":
    main()
