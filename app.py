from processing import extract_text, preprocess_text_generalized
from pin import initialize_pinecone, handle_file_upload, query_pinecone,get_openai_embeddings
import gradio as gr
from dotenv import load_dotenv
import os
import openai

# Load environment variables
load_dotenv()

# OpenAI and Pinecone settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "document-embeddings"
EMBEDDING_DIMENSION = 1536  # OpenAI embeddings dimension for `text-embedding-ada-002`
CLOUD = "aws"
REGION = "us-east-1"

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

def generate_response(user_query, pinecone_index, namespace="default", model="gpt-3.5-turbo"):
    """
    Generate a response to the user's query using OpenAI GPT and Pinecone for context retrieval.
    """
    # Step 1: Generate query embedding
    query_embedding = get_openai_embeddings(user_query)

    if query_embedding is None:
        return "Error generating query embedding. Please try again."

    # Step 2: Retrieve context from Pinecone
    matches = query_pinecone(pinecone_index, query_embedding, namespace=namespace, top_k=5)
    context = " ".join([match["metadata"].get("text", "") for match in matches])

    # Step 3: Create prompt
    if context.strip():
        prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"
    else:
        # No relevant context found, use a general-purpose prompt
        prompt = f"Question: {user_query}\n\nAnswer:"

    # Step 4: Generate response using OpenAI GPT
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant capable of answering general questions and questions based on provided context."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating response: {e}"


# Gradio UI for chatbot
def handle_user_query(file, user_query):
    """
    Handles the entire pipeline: dynamically process new file uploads,
    update embeddings in Pinecone, and generate responses for user queries.
    """
    namespace = "user_session"
    pinecone_index = initialize_pinecone(
        api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        cloud=CLOUD,
        region=REGION,
    )

    # Process the uploaded file dynamically
    if file:
        handle_file_upload(file.name, pinecone_index, namespace=namespace)

    # Generate response for the user's query
    return generate_response(user_query, pinecone_index, namespace=namespace)

with gr.Blocks() as ui:
    gr.Markdown("# Dynamic Chatbot with Retrieval-Augmented Generation (RAG)")
    file_input = gr.File(label="Upload Document", file_types=[".pdf", ".csv", ".json"])
    user_query = gr.Textbox(label="Your Query", placeholder="Ask a question...")
    chatbot_response = gr.Textbox(label="Chatbot Response", interactive=False)
    submit_button = gr.Button("Submit")
    submit_button.click(handle_user_query, inputs=[file_input, user_query], outputs=chatbot_response)

if __name__ == "__main__":
    ui.launch()

