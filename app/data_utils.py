from sentence_transformers import SentenceTransformer
import re
from pathlib import Path
from typing import List, Dict
import nltk
import os
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

TEXT_PATH = os.getenv("TEXT_PATH", "./../data/france_population.txt")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TITLE = os.getenv("TITLE", "France Population")
SOURCE = os.getenv("SOURCE", "https://www.britannica.com/place/France/Population-structure")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "4"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "1"))

DATA_SAVE_PATH = "./../data/france_population_data.pkl"

# nltk.download('punkt')  # Run once

def load_text(file_path: str) -> str:
    raw = Path(file_path).read_text(encoding='utf-8')
    return re.sub(r'\s+', ' ', raw).strip()

def attach_metadata(text: str, title: str, source: str) -> Dict:
    return {"title": title, "source": source, "text": text}

def chunk_text(text: str, chunk_size: int = 4, overlap: int = 1) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def embed_chunks(chunks: List[str], model_name: str):
    model = SentenceTransformer(model_name)
    return model.encode(chunks, convert_to_tensor=False)

def save_data(file_path: str, data: dict):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_data(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Load and prepare data
    text = load_text(TEXT_PATH)
    doc = attach_metadata(text, TITLE, SOURCE)
    chunks = chunk_text(doc["text"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"Total chunks: {len(chunks)}")

    if Path(DATA_SAVE_PATH).exists():
        print("📂 Loading saved data from file...")
        saved = load_data(DATA_SAVE_PATH)
        chunks = saved["chunks"]
        embeddings = saved["embeddings"]
        metadatas = saved["metadatas"]
        ids = saved["ids"]
    else:
        print("🚀 Generating embeddings...")
        embeddings = embed_chunks(chunks, MODEL_NAME)
        metadatas = [{"title": TITLE, "source": SOURCE} for _ in chunks]
        ids = [f"doc-{i}" for i in range(len(chunks))]

        save_data(DATA_SAVE_PATH, {
            "chunks": chunks,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "ids": ids
        })
        print(f"💾 Data saved to {DATA_SAVE_PATH}")

    # Print sample
    for idx in range(min(3, len(chunks))):
        print(f"\n--- Chunk {idx + 1} ---\n{chunks[idx]}\nEmbedding (first 5): {embeddings[idx][:5]}")
