# Recipe Chatbot - Technical Documentation

## Overview
A Retrieval-Augmented Generation (RAG) chatbot that answers recipe questions by searching through a collection of cookbooks and generating responses using a local LLM.

## Architecture

```
User Query
    ↓
[Streamlit UI] (app.py)
    ↓
[Sentence Transformer] → Query Embedding 
    ↓
[ChromaDB] → Vector Search → Top 5 relevant chunks(variable based on user)
    ↓
[Prompt Builder] → Context + Query + Guardrails
    ↓
[Ollama + Llama 3.2 3B] → Generate Response
    ↓
User receives answer
```

### Data Flow
1. **PDF Processing** (`data.py`): Extract text from 8 cookbooks → Clean with cleantext library → Split into 800-char chunks → Save to JSON
2. **Vector Database** (`vector_database.py`): Load chunks → Generate embeddings → Store in ChromaDB with persistent storage
3. **RAG Pipeline** (`rag_chatbot.py`): Query → Retrieve similar chunks → Build prompt with guardrails → Generate response
4. **Web Interface** (`app.py`): Streamlit UI with chat history and configuration

## Key Technical Decisions

### 1. LLM Selection: Llama 3.2 3B
**Why Llama 3.2 3B?**
- **Local execution**: No API costs, complete data privacy
- **Speed**: 3B model generates responses in ~1.5 seconds
- **Quality**: Good instruction following
- **Size**: 2GB download, runs on standard laptops

**Alternatives considered:**
- GPT-4 API: $0.03/request, requires internet, privacy concerns
- Llama 3.1 8B: Slower (4-5s response time), higher memory usage
- Gemma 2B: Lower quality responses in testing

### 2. Vector Database: ChromaDB
**Why ChromaDB?**
- **Simple setup**: Single Python library, no separate server needed
- **Persistent storage**: Database saved to disk
- **Fast search**: Very small time to retrieve 5 similar chunks from 1,851 documents

**Alternatives considered:**
- Pinecone: Requires cloud account
- FAISS: No metadata storage, manual persistence management
- Weaviate: Requires Docker

### 3. Embedding Model: all-MiniLM-L6-v2
**Why this model?**
- **Small**: 80MB, fast embedding generation
- **Quality**: SBERT model, 384 dimensions, good semantic understanding
- **Standard**: Widely used in production RAG systems


### 4. Text Processing: cleantext Library
**Why cleantext?**
- **Simple**: One-line cleanup instead of manually regexing
- **Robust**: Handles unicode, extra whitespace, special characters automatically
- **Maintained**: Active library with proper case handling

**Original approach:** Manual regex patterns → Failed on edge cases (recipes with unicode fractions, ingredient lists with special bullets)

### 5. Guardrails Implementation
**System Prompt Guardrails:**
```
1. Only use information from the provided context
2. If the context doesn't contain the answer, say "I don't have that information"
3. Always cite the source cookbook
4. Be concise and helpful
5. If asked for a recipe, provide clear steps and ingredients
```

**Why these specific rules?**
- Prevents hallucination (rule 1-2)
- Enables fact-checking (rule 3)
- Better user experience (rule 4-5)

### 6. Chunking Strategy: 800 characters, 150 overlap
**Why these parameters?**
- **800 chars**: Fits 1-2 recipe steps or ingredient lists, small enough for precise retrieval
- **150 overlap**: Prevents recipe steps from being split mid-instruction
- **Result**: 1,851 chunks from 8 cookbooks, good retrieval accuracy

**Testing results:**
- 500 chars: Too small, split single recipe steps
- 1500 chars: Too large, mixed unrelated recipes in same chunk

## System Components

### Files
- `data.py`: PDF extraction and preprocessing (PyPDF2 + cleantext)
- `vector_database.py`: ChromaDB setup and indexing
- `rag_chatbot.py`: Core RAG logic with CLI interface
- `app.py`: Streamlit web interface
- `requirements.txt`: Python dependencies

### Data
- 8 PDF cookbooks (1.8GB total)
- 1,851 text chunks in `processed_recipes.json`
- ChromaDB vector database in `chroma_db/` directory


## Setup Instructions

### 1. Install Dependencies
```bash
python -m venv NLPvenv
source NLPvenv/bin/activate
pip install -r requirements.txt
```

### 2. Install Ollama and Download Model
```bash
# Install Ollama from ollama.com
# Then download the model:
python -c "import ollama; ollama.pull('llama3.2:3b')"
```

### 3. Process PDFs 
```bash
python data.py
```

### 4. Create Vector Database 
```bash
python vector_database.py
```

### 5. Run the Chatbot
```bash
# Web interface (
streamlit run app.py

# Or CLI interface
python rag_chatbot.py
```

## Logging
The system logs all operations to the terminal:
- Query received with timestamp
- Number of contexts retrieved and their sources
- Response generation status
-

Example:
```
[10:30:45] Query: How do I make chocolate cake?
[10:30:45] Retrieved 5 contexts from: Betty Crockers International cookbook.pdf
[10:30:45] Generating response with llama3.2:3b...
[10:30:47] Response generated successfully
```

## Known Limitations
1. **Retrieval quality**: Embeddings sometimes mix unrelated recipes (e.g., eggs + rice)
2. **No recipe title extraction**: Metadata only includes source filename, not recipe names
3. **Single language**: Only processes English cookbooks

