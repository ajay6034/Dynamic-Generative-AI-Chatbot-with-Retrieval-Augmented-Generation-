import mimetypes
import pandas as pd
import PyPDF2
import json
import re
import spacy
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Detect file type
def detect_file_type(file_path):
    file_type = mimetypes.guess_type(file_path)[0]
    if file_type in ["application/pdf"]:
        return "pdf"
    elif file_type in ["text/csv", "application/vnd.ms-excel"]:
        return "csv"
    elif file_type == "application/json":
        return "json"
    else:
        raise ValueError(f"Unsupported file format: {file_type}")

# Extract text from CSV
def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    text = " ".join(df.astype(str).stack())
    return text

# Extract text from PDF
def extract_text_from_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from JSON
def extract_text_from_json(file_path):
    def recursive_text_extraction(data):
        if isinstance(data, dict):
            return " ".join(recursive_text_extraction(value) for value in data.values())
        elif isinstance(data, list):
            return " ".join(recursive_text_extraction(item) for item in data)
        else:
            return str(data)
    with open(file_path, 'r') as f:
        data = json.load(f)
    return recursive_text_extraction(data)

# Generalized text extraction
def extract_text(file_path):
    file_type = detect_file_type(file_path)
    if file_type == "csv":
        return extract_text_from_csv(file_path)
    elif file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type == "json":
        return extract_text_from_json(file_path)
    else:
        raise ValueError("Unsupported file format")

# Preprocess text
def preprocess_text_generalized(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^\x20-\x7E]", "", text)
    text = re.sub(r"\s+", " ", text)
    chunk_size = 100000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    processed_chunks = []
    for chunk in chunks:
        doc = nlp(chunk.lower())
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and token.is_alpha
        ]
        processed_chunks.append(" ".join(tokens))
    processed_text = " ".join(processed_chunks)
    return processed_text

# Generate embeddings
def get_embeddings_from_huggingface(cleaned_text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    sentence_embeddings = embeddings.mean(dim=1).numpy()
    return sentence_embeddings