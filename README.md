# Kanun Mitra

Kanun Mitra is an AI-driven legal assistant that helps user to understand Nepal’s laws easily, provide clear answers and helps businesses remain legally compliant.
It turns Nepal’s complex legal system into a smart and conversational experience with addition to compliance and clarity.

## Technologies Used

Streamlit: Interactive web interface for file uploading and chat.

Qdrant: High-performance vector database for storing and searching document embeddings.

Google Gemini API: Advanced Large Language Model (LLM) for generating answers based on retrieved context.

LangChain: Framework for document loading, text splitting, and managing the RAG pipeline.

HuggingFace Embeddings: Uses all-MiniLM-L6-v2 for creating local, efficient vector embeddings of the text.

## Project Structure

app.py: The main entry point for the Streamlit application. Handles UI and user interaction.

config.py: Central configuration file. Handles environment variables, API keys, and initializes the embedding model.

ingestion_pipeline.py: Logic for processing PDFs, chunking text, and storing vectors in Qdrant.

rag_query.py: Logic for converting user queries to vectors, searching Qdrant, and querying the Gemini API.

## Setup & Installation
### Install Dependencies:
Ensure you have Python=3.11 installed. Install the required libraries:

pip install -r requirements.txt

### Set up Qdrant:
You need a running instance of Qdrant. The easiest way is via Docker:
docker run -p 6333:6333 qdrant/qdrant

### Configure Environment Variables:
Create a .env file in the root directory and add your Gemini API key:
GEMINI_API_KEY=your_google_gemini_api_key_here

### Run the Application:
streamlit run app.py
