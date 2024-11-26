from processing import extract_text, preprocess_text_generalized, get_embeddings_from_huggingface
from pin import initialize_pinecone, handle_file_upload, query_pinecone
import gradio as gr
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables from .env file
load_dotenv()

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "document-embeddings"
EMBEDDING_DIMENSION = 384
CLOUD = "aws"
REGION = "us-east-1"

# Initialize LLM
def initialize_llm(model_name="tiiuae/falcon-7b"):
    """
    Load the tokenizer and model for the LLM.
    Replace 'gpt2' with a model suitable for RAG (e.g., GPT-4 or a Hugging Face model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Generate a response using RAG
def generate_response(llm_tokenizer, llm_model, user_query, pinecone_index, namespace="default"):
    """
    Generate a response to the user's query using RAG.
    """
    # Step 1: Generate query embedding
    query_embedding = get_embeddings_from_huggingface(user_query, model_name="sentence-transformers/all-MiniLM-L6-v2")[0]

    # Step 2: Retrieve context from Pinecone
    matches = query_pinecone(pinecone_index, query_embedding, namespace=namespace, top_k=5)
    context = " ".join([match["metadata"]["text"] for match in matches])  # Combine matched texts

    # Step 3: Create prompt
    prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"

    # Step 4: Generate response
    inputs = llm_tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_length=200, num_return_sequences=1)
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# Gradio UI functions
def handle_user_query(file, user_query):
    """
    Handles the file upload, saves embeddings to Pinecone, and processes user queries with RAG.
    """
    namespace = "user_session"  # Unique namespace for session
    pinecone_index = initialize_pinecone(
        api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        cloud=CLOUD,
        region=REGION,
    )

    # Process file upload (if a file is provided)
    if file:
        handle_file_upload(file, pinecone_index, namespace=namespace)

    # Initialize LLM
    llm_tokenizer, llm_model = initialize_llm()

    # Generate response to user query
    response = generate_response(llm_tokenizer, llm_model, user_query, pinecone_index, namespace=namespace)

    return response

# Gradio UI for chatbot
with gr.Blocks() as ui:
    gr.Markdown("# Chatbot with Retrieval-Augmented Generation (RAG)")
    gr.Markdown("Upload a document, and interact with the chatbot. The chatbot retrieves relevant context from your uploaded document to answer queries.")

    with gr.Row():
        file_input = gr.File(label="Upload Document (optional)", file_types=[".pdf", ".csv", ".json"])
        user_query = gr.Textbox(label="Your Query", placeholder="Type your question here...")
        chatbot_response = gr.Textbox(label="Chatbot Response", interactive=False)

    submit_button = gr.Button("Submit")
    submit_button.click(handle_user_query, inputs=[file_input, user_query], outputs=chatbot_response)

# Run the Gradio interface
if __name__ == "__main__":
    ui.launch()
