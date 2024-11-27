import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import openai
import hashlib
from processing import extract_text, preprocess_text_generalized

# Load environment variables from .env file
load_dotenv()

# Get Pinecone and OpenAI API keys from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "document-embeddings"
EMBEDDING_DIMENSION = 1536  # OpenAI's embeddings dimension for `text-embedding-ada-002`
CLOUD = "aws"
REGION = "us-east-1"

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY


# Initialize Pinecone
def initialize_pinecone(api_key, index_name, dimension, cloud="aws", region="us-east-1"):
    """
    Initializes Pinecone and creates an index if it doesn't exist.
    """
    # Create a Pinecone client instance
    pc = Pinecone(api_key=api_key)

    # Check if the index exists; if not, create it
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' does not exist. Creating a new index...")
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
    Save embeddings to Pinecone. Clears old embeddings if they exist.
    """
    try:
        # Check if the namespace exists before attempting deletion
        index_description = index.describe_index_stats()
        if namespace in index_description.get("namespaces", {}):
            index.delete(delete_all=True, namespace=namespace)
            print(f"Cleared all previous embeddings in namespace: {namespace}")
        else:
            print(f"Namespace '{namespace}' not found. Proceeding to save new embeddings.")
    except Exception as e:
        print(f"Error while checking/deleting embeddings in namespace {namespace}: {e}")

    if embeddings:
        vectors = [
            {"id": f"doc_{i}", "values": embedding, "metadata": metadata}
            for i, embedding in enumerate(embeddings)
        ]
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"Saved embeddings to namespace: {namespace}")
    else:
        print("No embeddings to save. Skipping upsert operation.")



# Generate embeddings using OpenAI API
def get_openai_embeddings(text, model="text-embedding-ada-002"):
    """
    Generate embeddings for a given text using OpenAI's embedding model.
    Handles splitting text into chunks if it exceeds the token limit.
    """
    max_tokens = 8192  # Adjust based on the model's maximum token limit
    try:
        # Split text into smaller chunks
        chunks = [text[i:i + max_tokens] for i in range(0, len(text), max_tokens)]
        embeddings = []
        for chunk in chunks:
            response = openai.Embedding.create(input=chunk, model=model)
            embeddings.extend([embedding["embedding"] for embedding in response["data"]])
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings with OpenAI API: {e}")
        return None

# Query Pinecone for relevant embeddings
def query_pinecone(index, query_embedding, namespace="default", top_k=3):
    """
    Retrieve relevant embeddings from Pinecone using similarity search.
    """
    results = index.query(
        vector=query_embedding,
        namespace=namespace,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]  # Returns the top-k matches with metadata


# Pipeline for handling file uploads and updating Pinecone vector DB
# Global variable to track the previous file hash
previous_file_hash = None

def calculate_file_hash(file_path):
    """
    Calculate a hash for the uploaded file to uniquely identify it.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def handle_file_upload(file_path, pinecone_index, namespace="default"):
    """
    Handle the process of uploading a file, clearing old embeddings,
    and saving new embeddings dynamically.
    """
    global previous_file_hash

    current_file_hash = calculate_file_hash(file_path)
    if current_file_hash == previous_file_hash:
        print(f"File '{file_path}' is identical to the previously uploaded file. Skipping processing.")
        return

    try:
        text = extract_text(file_path)
        processed_text = preprocess_text_generalized(text)

        # Generate embeddings
        embeddings = get_openai_embeddings(processed_text)
        if embeddings:
            metadata = {"file_name": os.path.basename(file_path), "text": processed_text}
            save_embeddings_to_pinecone(pinecone_index, embeddings, metadata, namespace)
            previous_file_hash = current_file_hash
        else:
            print("Failed to generate embeddings. Skipping save operation.")
    except Exception as e:
        print(f"Error processing file upload: {e}")


  

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

    
