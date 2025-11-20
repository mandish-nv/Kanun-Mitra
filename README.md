### Docx-Query-RAG

This project is a modular Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents, index their content into a Qdrant vector database, and ask questions using the Google Gemini LLM.

The application is built with a focus on performance and modularity, separating configuration, data ingestion, and query logic.

# 🚀 Technologies Used

Streamlit: Interactive web interface for file uploading and chat.

Qdrant: High-performance vector database for storing and searching document embeddings.

Google Gemini API: Advanced Large Language Model (LLM) for generating answers based on retrieved context.

LangChain: Framework for document loading, text splitting, and managing the RAG pipeline.

HuggingFace Embeddings: Uses all-MiniLM-L6-v2 for creating local, efficient vector embeddings of the text.

# 📂 Project Structure

app.py: The main entry point for the Streamlit application. Handles UI and user interaction.

config.py: Central configuration file. Handles environment variables, API keys, and initializes the embedding model.

ingestion_pipeline.py: Logic for processing PDFs, chunking text, and storing vectors in Qdrant.

rag_query.py: Logic for converting user queries to vectors, searching Qdrant, and querying the Gemini API.

# 🛠️ Setup & Installation

Clone the repository (or download the files):

git clone <repository_url>
cd <project_directory>


# Install Dependencies:
Ensure you have Python=3.11 installed. Install the required libraries:

pip install streamlit qdrant-client langchain langchain-community langchain-huggingface python-dotenv requests pypdf sentence-transformers


# Set up Qdrant:
You need a running instance of Qdrant. The easiest way is via Docker:

docker run -p 6333:6333 qdrant/qdrant


# Note: The code assumes Qdrant is running on localhost:6333.

# Configure Environment Variables:
Create a .env file in the root directory and add your Gemini API key:

GEMINI_API_KEY=your_google_gemini_api_key_here


# Run the Application:

streamlit run app.py


# 📖 How to Use

Open the Streamlit app in your browser (usually http://localhost:8501).

Sidebar: Click "Browse files" to upload a PDF document.

Sidebar: Click "Process / Ingest PDF" to chunk the text and store it in Qdrant. Wait for the "Balloons" animation!

Main Area: Type your question into the text box and click "Ask Gemini".

The AI will answer based strictly on the content of the uploaded PDF. You can also view the source chunks used for the answer.
