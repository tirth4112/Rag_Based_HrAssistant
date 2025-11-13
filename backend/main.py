
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import json

# ------------------ Load Environment Variables ------------------
load_dotenv()

# ------------------ App Setup ------------------
app = FastAPI(title="RAG HR Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Configuration ------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hr-policies")

if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in .env")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ------------------ Request Schema ------------------
class AskRequest(BaseModel):
    question: str


# ------------------ Helper: Generate Embedding ------------------
def get_embedding(text: str):
    """Generate text embedding from Ollama."""
    try:
        print("🔹 Generating embedding...")
        res = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=60
        )
        res.raise_for_status()
        data = res.json()

        if "embedding" not in data:
            print("❌ Ollama embedding response invalid:", data)
            raise HTTPException(status_code=500, detail="Invalid embedding response from Ollama")

        print("✅ Embedding generated.")
        return data["embedding"]

    except requests.exceptions.RequestException as e:
        print("❌ Ollama embedding error:", e)
        raise HTTPException(status_code=500, detail=f"Ollama embedding error: {str(e)}")


# ------------------ Helper: Parse Streaming Ollama Response ------------------
def parse_ollama_stream(raw_text):
    """Parse Ollama's streaming JSON response safely."""
    try:
        answer_chunks = []
        for line in raw_text.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "response" in obj:
                    answer_chunks.append(obj["response"])
            except json.JSONDecodeError:
                continue
        return "".join(answer_chunks).strip()
    except Exception as e:
        print("❌ Stream parse error:", e)
        return ""


# ------------------ Main Ask Endpoint ------------------
@app.post("/ask")
def ask_question(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    print(f"\n🧠 Received question: {question}")

    # Step 1: Generate embedding
    embedding = get_embedding(question)

    # Step 2: Query Pinecone
    try:
        print("🔹 Querying Pinecone...")
        search_results = index.query(vector=embedding, top_k=3, include_metadata=True)
        print("✅ Pinecone query complete.")
    except Exception as e:
        print("❌ Pinecone query failed:", e)
        raise HTTPException(status_code=500, detail=f"Pinecone query error: {str(e)}")

    matches = getattr(search_results, "matches", [])
    if not matches:
        print("⚠️ No matches found in Pinecone.")
        return {"answer": "Sorry, I couldn’t find anything related to that policy."}

    # Step 3: Prepare context
    context = "\n".join(
        [match.metadata.get("text", "") for match in matches if match.metadata]
    )
    print(f"📚 Context length: {len(context)}")

    # Step 4: Ask Ollama to generate final answer
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer clearly and politely."

    try:
        print("🔹 Sending prompt to Ollama model...")
        gen_res = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "llama3.2", "prompt": prompt},
            stream=True,
            timeout=120
        )
        gen_res.raise_for_status()

        # Parse streaming output
        raw_output = gen_res.text
        answer = parse_ollama_stream(raw_output)

        if not answer:
            print("⚠️ No valid text parsed from Ollama response.")
            answer = "Sorry, I couldn’t find an appropriate answer."

        print("✅ Final answer generated.")
        return {"answer": answer}

    except Exception as e:
        print("❌ Ollama generation failed:", str(e))
        raise HTTPException(status_code=500, detail=f"Ollama generation error: {str(e)}")


# ------------------ Root Endpoint ------------------
@app.get("/")
def home():
    return {"message": "✅ RAG HR Assistant Backend is running!"}
