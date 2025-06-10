import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
TOGETHER_API_URL = os.getenv("TOGETHER_API_URL", "https://api.together.xyz/inference")

headers = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}


def ask_together(query: str, contexts: list[str]) -> str:
    """Send a prompt with context to Together API and return the response."""
    # Combine all contexts into one string
    combined_context = "\n---\n".join(ctx.strip() for ctx in contexts)

    prompt = f"""### Instruction:
Answer the question based on the context provided.

### Context:
{combined_context}

### Question:
{query}

### Response:"""

    data = {
        "model": TOGETHER_MODEL,
        "prompt": prompt,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 2,
        "stop": ["</s>"]
    }

    response = requests.post(TOGETHER_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result["output"]["choices"][0]["text"].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"


# 🔹 نمونه تست
if __name__ == "__main__":
    query = "What is the name of the island in the Mediterranean that is treated as an integral part of France?"
    contexts = [
        "France lies near the western end of the great Eurasian landmass.",
        "The island of Corsica in the Mediterranean is treated as an integral part of the country."
    ]

    answer = ask_together(query, contexts)
    print("Answer:", answer)
