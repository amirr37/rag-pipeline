import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


questions = [
    "France lies near the western end of which great landmass?",
    "Between which latitudes does France lie?",
    "Which countries border France on the northeast, east, and south?",
    "Which bodies of water border France on the west and northwest?",
    "Which strait separates France from southeastern England?",
    "What is the name of the island in the Mediterranean that is treated as an integral part of France?",
    "What are the three main geologic regions distinguishable in France?",
    "What are the Hercynian massifs composed of?",
    "Which areas of France experienced direct sculpting by ice during the Pleistocene Epoch?",
    "How did periglacial lands modify the landscape during the Pleistocene?"
]


embedding_path = "../data/france_population_data.pkl"




# Load saved chunks and embeddings from the pickle file
with open(embedding_path, 'rb') as f:
    saved_data = pickle.load(f)

chunks = saved_data["chunks"]
chunk_embeddings = np.array(saved_data["embeddings"])

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_top_k_chunks(query: str, top_k: int = 2):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    # Sort indices by similarity descending
    sorted_indices = np.argsort(similarities)[::-1]

    seen = set()
    top_chunks = []

    for idx in sorted_indices:
        chunk = chunks[idx]
        if chunk not in seen :
            seen.add(chunk)
            top_chunks.append(chunk)
        if len(top_chunks) == top_k:
            break

    return top_chunks

# Example usage:
if __name__ == "__main__":
    query = "Which countries border France?"
    results = get_top_k_chunks(query, top_k=3)
    for i, chunk in enumerate(results, 1):
        print(f"Result {i}:\n{chunk}\n")
