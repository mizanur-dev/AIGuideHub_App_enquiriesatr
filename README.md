# Advanced AI RAG Chatbot (Django REST Framework)

An enterprise-grade, fully functional Retrieval-Augmented Generation (RAG) AI Chatbot built with Django REST Framework, Google Gemini AI, and Pinecone Vector Database.

This application allows users to upload PDF documents, extract text securely, convert it into semantically meaningful chunks, and chat interactively with the document content using advanced AI architecture.

## 🚀 Key Features

*   **Advanced Semantic Chunking:** Moves beyond trivial fixed-size character chunking. Uses `langchain_experimental`'s `SemanticChunker` and Gemini embeddings to intelligently group sentences together mathematically by topic and meaning, improving search accuracy.
*   **Vector Database (Pinecone):** Efficiently indexes and retrieves document embeddings. Auto-batches vectors (100 at a time) to comfortably handle large PDF uploads without encountering payload limits.
*   **Google Gemini AI:** Fully utilizes `gemini-2.5-flash` for high-speed, high-accuracy conversational responses, and `models/gemini-embedding-001` for deep text representation.
*   **Dynamic Session Memory:** Keeps track of conversational history (up to 20 turns) to understand follow-up questions intelligently.
*   **Synchronous Production Flow:** No complex Celery/Redis dependencies required—processes everything efficiently right out of the box in standard HTTP threads.
*   **Dynamic Namespaces:** Organizes vector data securely. Every user gets a unique `session_id` acting as a secure "namespace" inside Pinecone. When you upload a new document, it automatically wipes your old namespace clean to prevent cross-document confusion.

## 🛠️ Technology Stack

*   **Backend:** Python 3.11, Django, Django REST Framework (DRF)
*   **AI/LLM Framework:** Langchain, Langchain Experimental, Google Generative AI
*   **Vector Store:** Pinecone
*   **File Parsing:** PyPDF

---

## ⚙️ Setup and Installation

### 1. Environment Setup

Create a virtual environment and install the dependencies:

```powershell
# Create Virtual Environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 2. Environment Variables (`.env`)

Create a `.env` file in the root of the directory (same level as `manage.py`) with your API keys:

```env
SECRET_KEY=your_django_secret_key_here
DEBUG=True
ALLOWED_HOSTS=*

# AI & Vector DB Credentials
GEMINI_API_KEY=your_google_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=advanced-ai-chatbot-using-cag
```

### 3. Database Migration & Running the Server

Initialize the local SQLite tracking database (handles session metadata) and boot up!

```powershell
python manage.py migrate
python manage.py runserver
```

---

## 📖 API Documentation & Workflow

The architecture uses a 3-step workflow: Authenticate ➔ Upload Document ➔ Chat.

### Step 1: Initialize User Session
**Endpoint:** `POST /api/set_email/`

Instead of a heavy authentication system, the API uses a lightweight email-based session initializer.

**Request:**
```json
{
    "email": "user@example.com"
}
```

**Response:**
```json
{
    "message": "Email set successfully: user@example.com. You can now use the chatbot.",
    "session_id": "user@example.com_a1b2c3d4"
}
```
*(Save this `session_id`. You will use it for all subsequent requests to isolate your documents and chat history.)*

### Step 2: Upload Document (PDF)
**Endpoint:** `POST /api/upload_pdf/`

Uploads the PDF, parses the text, calculates semantic chunks, generates embeddings, and batch-upserts them directly into Pinecone using your `session_id` as the secure internal namespace.

**Request:** 
*   **Format:** `multipart/form-data`
*   `file`: (Your PDF File)
*   `session_id`: `user@example.com_a1b2c3d4`

**Response:**
```json
{
    "message": "Document processed and indexed successfully."
}
```

### Step 3: Chat with the AI!
**Endpoint:** `POST /api/chat/`

Queries the AI. It will dynamically search Pinecone for the most relevant semantic chunks from your uploaded document, inject them into the system prompt alongside your recent chat history, and generate an intelligent response based *strictly* on the provided context.

**Request:**
```json
{
    "message": "What is the main topic of the document?",
    "session_id": "user@example.com_a1b2c3d4"
}
```

**Response:**
```json
{
    "response": "The main topic of the document is..."
}
```
