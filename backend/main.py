# import os
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# from pydantic import BaseModel
# from pinecone import Pinecone, ServerlessSpec
# import requests

# # --- Load environment variables ---
# load_dotenv()

# # --- FastAPI App ---
# app = FastAPI(title="RAG HR Assistant", version="1.0")

# # --- Enable CORS (allow frontend requests) ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, use your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- Initialize Pinecone ---
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hr-policies")

# if not PINECONE_API_KEY:
#     raise ValueError("❌ Missing PINECONE_API_KEY in .env file")

# pc = Pinecone(api_key=PINECONE_API_KEY)

# # List existing indexes
# existing_indexes = [idx.name for idx in pc.list_indexes()]

# # Create index if it doesn't exist
# if PINECONE_INDEX_NAME not in existing_indexes:
#     print(f"📦 Creating Pinecone index: {PINECONE_INDEX_NAME}")
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=1536,  # typical for OpenAI/Ollama embeddings
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
# else:
#     print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' already exists")

# # Connect to existing index
# index = pc.Index(PINECONE_INDEX_NAME)
# print(f"🔗 Connected to Pinecone index: {PINECONE_INDEX_NAME}")

# # --- Ollama Setup (for local LLM embeddings) ---
# OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# # --- Pydantic model for input ---
# class AskRequest(BaseModel):
#     question: str


# # --- API Routes ---

# @app.get("/")
# def root():
#     return {"message": "🚀 RAG HR Assistant Backend Running Successfully!"}


# @app.get("/pinecone-info")
# def get_index_info():
#     stats = index.describe_index_stats()
#     return {"index_name": PINECONE_INDEX_NAME, "stats": stats}


# @app.post("/add-embedding")
# def add_embedding(item: dict):
#     """
#     Add a vector to Pinecone.
#     Example payload:
#     {
#         "id": "policy-1",
#         "values": [0.12, 0.45, ..., 0.78],
#         "metadata": {"title": "Leave Policy"}
#     }
#     """
#     try:
#         index.upsert(vectors=[item])
#         return {"status": "success", "message": f"Vector {item['id']} added successfully."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/search")
# def search(vector: str):
#     """
#     Search Pinecone for similar embeddings.
#     Example: /search?vector=0.11,0.22,0.33,...
#     """
#     try:
#         query_vector = [float(x) for x in vector.split(",")]
#         result = index.query(vector=query_vector, top_k=3, include_metadata=True)
#         return {"results": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # --- Main RAG endpoint ---
# # @app.post("/ask")
# # def ask_question(request: AskRequest):
# #     """
# #     Handle HR-related user questions.
# #     Frontend sends: { "question": "What is the leave policy?" }
# #     """
# #     question = request.question.strip()

# #     # Generate embedding using Ollama (if connected)
# #     try:
# #         emb_res = requests.post(
# #             f"{OLLAMA_URL}/api/embeddings",
# #             json={"model": "nomic-embed-text", "prompt": question},
# #             timeout=20,
# #         )
# #         emb_res.raise_for_status()
# #         embedding = emb_res.json()["embedding"]
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

# #     # Search for relevant HR documents in Pinecone
# #     try:
# #         search_results = index.query(vector=embedding, top_k=3, include_metadata=True)
# #         if not search_results.matches:
# #             return {"answer": "Sorry, I couldn’t find anything related to that policy."}

# #         # Combine metadata from results
# #         context = "\n".join(
# #             [match.metadata.get("text", "") for match in search_results.matches if match.metadata]
# #         )

# #         # Generate answer using Ollama
# #         prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer clearly and politely."
# #         gen_res = requests.post(
# #             f"{OLLAMA_URL}/api/generate",
# #             json={"model": "llama3.2", "prompt": prompt},
# #             stream=False,
# #             timeout=30,
# #         )
# #         gen_res.raise_for_status()
# #         answer = gen_res.json().get("response", "").strip()
# #         return {"answer": answer or "Sorry, I couldn’t find an appropriate answer."}

# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Error querying Pinecone: {str(e)}")

# @app.post("/ask")
# def ask_question(request: AskRequest):
#     """
#     Handle HR-related user questions.
#     Frontend sends: { "question": "What is the leave policy?" }
#     """
#     question = request.question.strip()
#     if not question:
#         raise HTTPException(status_code=400, detail="Question cannot be empty")

#     # --- Step 1: Generate embedding from Ollama ---
#     try:
#         emb_res = requests.post(
#             f"{OLLAMA_URL}/api/embeddings",
#             json={"model": "nomic-embed-text", "prompt": question},
#             timeout=30,
#         )
#         emb_res.raise_for_status()
#         embedding = emb_res.json().get("embedding")
#         if not embedding:
#             raise ValueError("No embedding returned from Ollama.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

#     # --- Step 2: Query Pinecone ---
#     try:
#         search_results = index.query(vector=embedding, top_k=3, include_metadata=True)
#         matches = search_results.get("matches", [])
#         if not matches:
#             return {"answer": "Sorry, I couldn’t find anything related to that policy."}

#         # Extract text context from Pinecone matches
#         context = "\n\n".join(
#             [m["metadata"].get("text", "") for m in matches if "metadata" in m]
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Pinecone query failed: {str(e)}")

#     # --- Step 3: Generate answer using Ollama ---
#     try:
#         prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer clearly and politely."
#         gen_res = requests.post(
#             f"{OLLAMA_URL}/api/generate",
#             json={"model": "llama3.2", "prompt": prompt},
#             timeout=60,
#         )
#         gen_res.raise_for_status()
#         response_json = gen_res.json()
#         answer = response_json.get("response") or response_json.get("text", "")
#         return {"answer": answer.strip() if answer else "Sorry, I couldn’t find an appropriate answer."}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")
# # --- Run with ---
# # uvicorn main:app --reload




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
