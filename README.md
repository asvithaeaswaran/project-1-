# 🧠 DocuQuery AI — RAG Document Intelligence Platform

A high-performance **Retrieval-Augmented Generation (RAG)** system built with **Python**, **Django**, **LangChain**, **FAISS**, and **all-MiniLM-L6-v2** embeddings. Upload documents (PDF, DOCX, TXT, CSV, MD) and ask questions with precise answers and live source citations.

---

## ⚡ Core Architecture

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Vector Database**: `FAISS` (Facebook AI Similarity Search) with disk-persisted indexes per user
- **LLM Providers**:
  - **Google Gemini** (Gemini 2.0 Flash / Gemini 1.5 Pro)
  - **Ollama Local** (Llama 3, Mistral, Gemma 2, Phi-3, DeepSeek)
  - **Groq & OpenAI** (Ultra-fast cloud inference with 120B / GPT-4o models)
  - **Built-in Extractive Engine** (Zero-dependency fallback if offline)
- **RAG Framework**: `LangChain` (`RecursiveCharacterTextSplitter`, `create_retrieval_chain`, document loaders)
- **Backend**: Python 3.12 + Django 6.1
- **Database**: SQLite for document metadata, conversation history, and user authentication

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: `Django`, `langchain`, `langchain-community`, `langchain-huggingface`, `langchain-google-genai`, `langchain-ollama`, `langchain-openai`, `faiss-cpu`, `sentence-transformers`, `pypdf`, `python-docx`)*

### 2. Configure Environment (`.env`)
Create or edit your `.env` file:
```env
# Google Gemini
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-2.0-flash

# Ollama Local (Optional)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Groq / OpenAI (Optional)
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=openai/gpt-oss-120b

DEBUG=True
```

### 3. Run Migrations & Start Server
```bash
python manage.py migrate
python manage.py runserver
```

Open `http://localhost:8000` in your web browser.

---

## 📖 How to Use

1. **Log in / Demo Login**: Use your credentials or click **1-Click Instant Demo Login**.
2. **Upload Documents**:
   - Go to the **Documents (Knowledge Base)** tab in the sidebar or click **Upload Doc**.
   - Drag & drop your PDF, DOCX, TXT, CSV, or Markdown files.
   - The app extracts text, splits it into semantically coherent chunks with overlap, computes `all-MiniLM-L6-v2` embeddings, and saves them to the FAISS vector index.
3. **Inspect Chunks**: Click **Inspect** on any document card to view extracted chunks and character counts.
4. **Ask Questions**:
   - In the chat interface, ensure **⚡ RAG Mode** is toggled ON.
   - Select either **All Documents** or specific files.
   - Type your question (e.g., *"What is the main finding in section 3?"* or *"Summarize the metrics across all files"*).
5. **View Source Citations**:
   - Every RAG answer includes an interactive **Sources Cited** drawer showing exact document names, pages, and highlighted context excerpts.

---

## 🧪 Automated Testing

Run the automated test suite:
```bash
python manage.py test
```
