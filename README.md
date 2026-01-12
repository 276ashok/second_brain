# Second Brain - Document Q&A RAG System

A Retrieval-Augmented Generation (RAG) system for question-answering over documents. This application allows you to upload documents (PDF, TXT, DOCX), process them into a vector database, and ask questions using Groq's LLM models.

## Features

- **Multi-format Document Support**: Upload and process PDF, TXT, and DOCX files
- **Semantic Search**: Find relevant document chunks using vector similarity search
- **AI-Powered Answers**: Generate accurate answers using Groq LLM
- **Source Citations**: View source documents and page numbers for each answer
- **Chat Interface**: Interactive chat interface built with Streamlit
- **Persistent Storage**: Vector database persists between sessions
- **Conversation Memory**: Maintains context across multiple questions

## Architecture

The system follows a RAG (Retrieval-Augmented Generation) architecture:

1. **Document Processing**: Documents are loaded, split into chunks, and embedded using local sentence-transformers models
2. **Vector Storage**: Chunks are stored in ChromaDB with their embeddings
3. **Retrieval**: User queries are embedded and matched against stored chunks
4. **Generation**: Retrieved context is used to generate answers via Groq LLM

## Installation

### Prerequisites

- Python 3.8 or higher
- Groq API Key (get one from [Groq Console](https://console.groq.com/))

### Setup Steps

1. **Clone or navigate to the project directory**:
   ```bash
   cd second_brain
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your Groq API key:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

## Usage

### Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Using the Application

1. **Upload Documents**:
   - Go to the "Upload" tab
   - Click "Choose a file" and select a PDF, TXT, or DOCX file
   - Click "Process Document" to add it to the knowledge base

2. **Ask Questions**:
   - Go to the "Chat" tab
   - Type your question in the chat input
   - View the answer with source citations

3. **View Sources**:
   - Click on "Sources" in any assistant response
   - See the document chunks used to generate the answer
   - View relevance scores and page numbers

## Configuration

Configuration can be modified in `src/config.py`:

- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K_CHUNKS`: Number of chunks to retrieve (default: 5)
- `GROQ_MODEL`: Groq model to use (default: "llama-3.3-70b-versatile")
- `EMBEDDING_MODEL`: Embedding model (default: "sentence-transformers/all-MiniLM-L6-v2")
- `TEMPERATURE`: LLM temperature (default: 0.7)

## Project Structure

```
second_brain/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── document_processor.py  # Document loading and processing
│   ├── retrieval.py      # Vector search and retrieval
│   ├── generation.py     # LLM setup and answer generation
│   └── utils.py          # Utility functions
├── data/
│   └── uploads/          # Uploaded document storage
└── vectorstore/          # ChromaDB vector database
```

## Dependencies

- **streamlit**: Web application framework
- **langchain**: LLM framework and utilities
- **langchain-groq**: Groq LLM integration
- **langchain-community**: Community integrations
- **langchain-chroma**: ChromaDB integration
- **langchain-huggingface**: HuggingFace embeddings integration
- **chromadb**: Vector database
- **pypdf**: PDF processing
- **python-docx**: DOCX processing
- **sentence-transformers**: Local embeddings models
- **python-dotenv**: Environment variable management

## Troubleshooting

### API Key Issues

If you see "GROQ_API_KEY not found" errors:
- Make sure you've created a `.env` file
- Verify the API key is correct
- Check that the `.env` file is in the project root

### Document Processing Errors

- **Unsupported format**: Only PDF, TXT, and DOCX are supported
- **Corrupted files**: Try re-saving the document
- **Large files**: Processing may take time for large documents

### Vector Store Issues

- If the vector store becomes corrupted, delete the `vectorstore/` directory
- The system will recreate it on the next run

### Memory Issues

- For very large documents, consider reducing `CHUNK_SIZE` in `src/config.py`
- Process documents one at a time if memory is limited

## Development

### Running Tests

Unit tests can be added in a `tests/` directory. Example structure:

```
tests/
├── test_document_processor.py
├── test_retrieval.py
└── test_generation.py
```

### Code Style

The project follows PEP 8 style guidelines. Consider using:
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking

## License

This project is provided as-is for educational and personal use.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Groq](https://groq.com/)
- UI built with [Streamlit](https://streamlit.io/)
- Embeddings provided by [sentence-transformers](https://www.sbert.net/)

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the error messages in the application
3. Check that all dependencies are installed correctly
4. Verify your API key is valid and has sufficient quota

---

**Note**: This application requires an active internet connection to use Groq's API. API usage may incur costs depending on your Groq account setup. Embeddings are processed locally using sentence-transformers, so no API key is needed for that component.
