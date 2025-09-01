

# 🤖 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to query PDF documents related to a Customer Relationship Management (CRM) and application processing system. The project integrates a **Qdrant** vector database, a **FastAPI** backend, and a modern **React** frontend.

This system is designed not just for generic document querying but specifically to answer questions about customer application processes, CRM features, and candidate management workflows described in your PDF files.

<img width="1110" height="800" alt="Adsız" src="https://github.com/user-attachments/assets/e8ec323b-9910-493e-b42b-05e4e4803420" />


https://github.com/user-attachments/assets/252c6300-464a-4898-b824-42ad9c023736


---

## 🚀 Key Features

### Core RAG Features
*   **📄 Automated PDF Ingestion**: Automatically processes PDFs from the `data/` folder, cleans the content, and creates vector embeddings using Google Generative AI.
*   **⚡ High-Speed Vector Search**: Utilizes **Qdrant** to store and index document embeddings for incredibly fast and accurate similarity searches.
*   **💬 Conversational AI Queries**: Users can ask natural language questions and receive precise answers synthesized by an LLM.
*   **📚 Source Referencing**: Every answer is backed by references to the source PDF and page number, ensuring transparency and trust.

### CRM-Specific Functionality
*   **👤 Customer-Side Insights**: Quickly get information on features like dynamic application forms (for Single/Married applicants) or how customers can track their application status.
*   **🗂️ Admin/CRM Panel Insights**: Ask questions about the admin panel's capabilities, such as managing candidate profiles, adding notes, handling documents, or tracking process stages (`New Application`, `Document Collection`, `Approved`).
*   **✅ Task & Communication Management**: Query the system on how consultants can assign tasks, manage documents, and log communications within the CRM.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Backend** | Python 3.10+, FastAPI, LangChain, `langchain-google-genai` |
| **Frontend**| React 18 + Vite, CSS (Custom Styling) |
| **Database**| Qdrant (Vector Database) |
| **Deployment**| Docker (for Qdrant) |
| **Tooling** | `python-dotenv` for environment management |

---

## 📦 Project Structure

```
rag_project/
├─ rag_pipeline.py          # PDF processing and RAG logic
├─ qdrant_pipeline.py       # Qdrant integration
├─ main.py                  # FastAPI backend server
├─ data/                    # Place your PDFs here for ingestion
├─ frontend/                # React frontend application
│  ├─ src/
│  ├─ package.json
│  └─ vite.config.js
├─ .env                     # Environment variables (not tracked in Git)
├─ requirements.txt         # Python dependencies
└─ README.md                # You are here!
```

---

## ⚡ Prerequisites

Before you begin, ensure you have the following installed and configured:

*   ✅ Python 3.10 or higher
*   ✅ Node.js 18+ and npm
*   ✅ A running instance of **Qdrant** (e.g., via Docker).
*   🔑 A **Google API Key** with the Generative Language API enabled.

---

## ⚙️ Setup & Installation

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag_project.git
cd rag_project
```

### 2. Backend Setup

First, set up the Python virtual environment and install the required dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt```

Next, create a `.env` file in the project root and add your configuration:

```ini
# .env
GOOGLE_API_KEY="your_google_api_key_here"
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION="documents"
```

Finally, place the PDF files you want to query inside the `data/` directory and run the FastAPI server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The backend will now be running at `http://localhost:8000`.

### 3. Frontend Setup

In a new terminal, navigate to the `frontend` directory and install the Node.js dependencies.

```bash
cd frontend
npm install
```

Now, run the React development server:

```bash
npm run dev
```
The frontend will be accessible at `http://localhost:5173`. It is pre-configured to communicate with the backend at `http://localhost:8000`.

---

## 📝 How It Works

1.  **Ingestion**: When the backend starts, it scans the `data/` folder for PDF files.
2.  **Processing**: Each PDF is read, its text is split into smaller, meaningful chunks, and the content is cleaned.
3.  **Embedding**: Text chunks are converted into numerical vectors (embeddings) using Google's Generative AI models.
4.  **Storage**: These embeddings, along with their metadata (source file, page number), are stored in the Qdrant vector database.
5.  **Query**: A user types a question into the React chat interface.
6.  **Retrieval**: The backend creates an embedding for the user's question and queries Qdrant to find the most relevant document chunks (the "context").
7.  **Generation**: The original question and the retrieved context are sent to the Google Generative AI model.
8.  **Response**: The model generates a comprehensive answer based on the provided context, which is then sent back to the frontend and displayed to the user with source references.
