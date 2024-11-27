
# Dynamic Chatbot with Retrieval-Augmented Generation (RAG)

This project implements a **Dynamic Generative AI Chatbot** powered by **Retrieval-Augmented Generation (RAG)**. The system allows users to upload documents, processes them to extract and store embeddings in a vector database (Pinecone), and uses **OpenAI's GPT models** to provide contextually aware answers to user queries. 

## Features

- **Dynamic Document Upload**: Supports uploading documents in PDF, CSV, or JSON formats. The uploaded document's content is processed and stored as embeddings for efficient query retrieval.
- **Retrieval-Augmented Generation (RAG)**: Combines vector similarity search with OpenAI's GPT to enhance response accuracy.
- **Embeddings Management**: Utilizes OpenAI embeddings for document representation and supports efficient similarity searches via Pinecone.
- **Interactive User Interface**: Built using Gradio for an easy-to-use chatbot interface.

## Workflow

1. **Document Upload**:
   - Users upload a document via the chatbot interface.
   - The system detects the file type and extracts text using format-specific methods (e.g., PyPDF2 for PDFs, pandas for CSVs).
   - The extracted text is preprocessed (tokenization, lemmatization, stop-word removal) for efficient embedding generation.

2. **Embeddings Generation**:
   - Embeddings for the document text are generated using **OpenAI's `text-embedding-ada-002`** or Hugging Face models.
   - These embeddings are stored in the **Pinecone vector database**, replacing any previously stored embeddings in the session namespace.

3. **Query Processing**:
   - When a user asks a question, the chatbot generates a query embedding.
   - Relevant document embeddings are retrieved from Pinecone using similarity search.
   - The retrieved context is combined with the user's query to form a prompt for GPT-3.5-turbo.

4. **Response Generation**:
   - OpenAI's GPT model generates a response based on the prompt, including the retrieved context, and returns it to the user.

## Technologies Used

- **Language Models**:
  - OpenAI GPT (`gpt-3.5-turbo`) for response generation.
  - OpenAI and Hugging Face models for embedding generation.
- **Vector Database**:
  - Pinecone for storing and querying embeddings.
- **Natural Language Processing**:
  - SpaCy for text preprocessing (e.g., lemmatization, stop-word removal).
  - Hugging Face Transformers for alternate embedding generation.
- **Frontend**:
  - Gradio for creating an interactive chatbot interface.
- **Backend**:
  - Python for core logic.
  - PyPDF2, pandas, and JSON libraries for data processing.



## Future Improvements

- Improved handling of large files with streaming techniques.



