import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from processing import extract_text, preprocess_text_generalized, get_embeddings_from_huggingface

# Load environment variables from .env file
load_dotenv()

# Get Pinecone API key and environment from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "document-embeddings"
EMBEDDING_DIMENSION = 384
CLOUD = "aws"
REGION = "us-east-1"

def initialize_pinecone(api_key, index_name, dimension, cloud="aws", region="us-east-1"):
    """
    Initializes Pinecone and creates an index if it doesn't exist.
    """
    # Create a Pinecone client instance
    pc = Pinecone(api_key=api_key)

    # Check if the index exists; if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )

    # Wait for the index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        print("Waiting for index to be ready...")
        time.sleep(1)

    # Return the Pinecone Index object
    return pc.Index(index_name)

# Save embeddings to Pinecone vector DB
from pinecone.core.openapi.shared.exceptions import NotFoundException

def save_embeddings_to_pinecone(index, embeddings, metadata, namespace="default"):
    """
    Clears previous embeddings in the namespace (if they exist) and saves new embeddings.
    """
    # Try deleting all embeddings in the namespace, ignore errors if namespace does not exist
    try:
        index.delete(delete_all=True, namespace=namespace)
    except NotFoundException:
        print(f"Namespace '{namespace}' not found. Proceeding to save new embeddings.")

    # Save new embeddings
    vectors = [
        {
            "id": f"doc_{i}",
            "values": embedding.tolist(),
            "metadata": metadata
        }
        for i, embedding in enumerate(embeddings)
    ]
    index.upsert(vectors=vectors, namespace=namespace)


# Query Pinecone for relevant embeddings
def query_pinecone(index, query_embedding, namespace="default", top_k=3):
    """
    Retrieve relevant embeddings from Pinecone using similarity search.
    """
    results = index.query(
        vector=query_embedding.tolist(),
        namespace=namespace,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]  # Returns the top-k matches with metadata

# Pipeline for handling file uploads and updating Pinecone vector DB
def handle_file_upload(file_path, pinecone_index, model_name="sentence-transformers/all-MiniLM-L6-v2", namespace="default"):
    """
    Handles the complete pipeline: extract text, preprocess, generate embeddings, and save to Pinecone.
    """
    # Step 1: Extract text
    text = extract_text(file_path)

    # Step 2: Preprocess the text
    processed_text = preprocess_text_generalized(text)

    # Step 3: Generate embeddings
    embeddings = get_embeddings_from_huggingface(processed_text, model_name)

    # Step 4: Save embeddings to Pinecone
    metadata = {"file_name": file_path}
    save_embeddings_to_pinecone(pinecone_index, embeddings, metadata, namespace)

# Example usage
if __name__ == "__main__":
    # Initialize Pinecone with serverless specifications
    pinecone_index = initialize_pinecone(
        api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        cloud=CLOUD,
        region=REGION
    )

 