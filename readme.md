# 🧠 RAG Chatbot: Retrieval-Augmented Generation with FastAPI

This project implements a **Retrieval-Augmented Generation (RAG) pipeline chatbot** using semantic search and a powerful
hosted language model. It provides two API endpoints (`/retrieve` and `/generate`) to support natural language question
answering over a domain-specific dataset.

---

## 🚀 Features

- ✅ Chunk-based text preprocessing with metadata
- ✅ Sentence-transformer embeddings for semantic retrieval
- ✅ Cosine similarity scoring with Top-K control
- ✅ Prompted generation using LLaMA-3 via Together API
- ✅ FastAPI server with clean schema-based endpoints
- ✅ Modular code with support for `.env` configuration

---

## 🧱 What’s Included

### 🔹 Backend
- FastAPI server with `/retrieve` and `/generate` endpoints
- Sentence-transformer embeddings
- Cosine similarity search
- LLM-powered generation via Together API (LLaMA 3.0)

### 🔸 Frontend
- Vanilla HTML/CSS/JavaScript
- Simple UI to send user queries and display chatbot responses
- Interacts with the FastAPI endpoints



---

## 🧠 Models Used

### 🔹 Embedding Model: `all-MiniLM-L6-v2`

- Source: [sentence-transformers](https://www.sbert.net/)
- Description: Lightweight transformer fine-tuned for semantic similarity.
- Purpose: Converts text chunks and queries into vector embeddings for similarity comparison.
- Trade-off: Faster and smaller than larger transformer models with good accuracy for retrieval.

### 🔸 Generation Model: `meta-llama/Llama-3-70B-Instruct-Turbo-Free`

- Source: [Together API](https://www.together.xyz/)
- Description: Instruction-tuned version of LLaMA 3.0 with 70B parameters.
- Purpose: Takes top-k context chunks and a query, and generates a grounded, coherent response.
- Parameters:
    - `temperature`: 0.3 (less randomness, more factual)
    - `top_p`: 0.9 (nucleus sampling)
    - `top_k`: 2 (limits token diversity)

---

## 🧰 Libraries Used

| Library                 | Purpose                                    |
|-------------------------|--------------------------------------------|
| `sentence-transformers` | Semantic text embeddings                   |
| `nltk`                  | Sentence tokenization for chunking         |
| `sklearn`               | Cosine similarity calculation              |
| `requests`              | API call to Together for LLM generation    |
| `fastapi`               | RESTful API framework                      |
| `pydantic`              | Input/output validation models             |
| `python-dotenv`         | Loads environment variables from `.env`    |
| `pickle`                | Saving and loading preprocessed embeddings |
| `uvicorn`               | Local development server for FastAPI       |
| `CORS middleware`       | Enables frontend interaction (optional)    

---

## 🛠️ How to Run

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Add a .env file:**

```bash
TEXT_PATH=../data/france_population.txt
EMBEDDING_MODEL=all-MiniLM-L6-v2
TITLE=France Population
SOURCE=https://www.britannica.com/place/France/Population-structure
CHUNK_SIZE=4
CHUNK_OVERLAP=1

TOGETHER_API_KEY=your_api_key_here
TOGETHER_MODEL=meta-llama/Llama-3-70B-Instruct-Turbo-Free
TOGETHER_API_URL=https://api.together.xyz/inference
```

3. **Run preprocessing (optional if already done)**:

```bash
python data_utils.py
```

4. **Start the API server(in main directory) :**


```bash
uvicorn main:app --reload
```

5. **Open the Frontend **:

You can simply open the``` index.html``` file in your browser:


```bash
open index.html
# or manually double-click the file to launch it
```



