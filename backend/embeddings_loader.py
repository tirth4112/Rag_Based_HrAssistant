import os
import fitz  # PyMuPDF
import hashlib
import requests
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# === CONFIGURATION ===
DATA_FOLDER = os.path.abspath("data/hr_policies")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hr-policies")
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768  # nomic-embed-text produces 768-dimension embeddings


# === SETUP PINECONE ===
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, if not create or recreate with correct dimension
existing_indexes = [i.name for i in pc.list_indexes()]
if INDEX_NAME in existing_indexes:
    info = pc.describe_index(INDEX_NAME)
    if info.dimension != EMBEDDING_DIM:
        print(f"⚠️ Index '{INDEX_NAME}' has dimension {info.dimension}, expected {EMBEDDING_DIM}. Recreating...")
        pc.delete_index(INDEX_NAME)
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"✅ Recreated index '{INDEX_NAME}' with dimension {EMBEDDING_DIM}")
else:
    print(f"🆕 Creating new Pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"✅ Created index '{INDEX_NAME}' successfully")

# Connect to the index
index = pc.Index(INDEX_NAME)


# === FUNCTION: Generate embedding using Ollama ===
def get_embedding(text: str):
    """Fetch embedding from local Ollama using nomic-embed-text"""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=60
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None


# === FUNCTION: Process and upload documents ===
def load_and_embed_docs(folder_path: str):
    """Load all PDFs from a folder, extract text, embed, and push to Pinecone"""
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    print(f"\n📂 Loading documents from: {folder_path}")

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ No PDF files found.")
        return

    for file in pdf_files:
        pdf_path = os.path.join(folder_path, file)
        print(f"\n📘 Processing: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text("text")

            # Split text into 1000-character chunks
            chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]

            for chunk in chunks:
                if not chunk.strip():
                    continue

                emb = get_embedding(chunk)
                if emb is None:
                    print("⚠️ Skipping chunk due to embedding error.")
                    continue

                vector_id = hashlib.md5(chunk.encode()).hexdigest()
                index.upsert(
                    vectors=[{
                        "id": vector_id,
                        "values": emb,
                        "metadata": {"text": chunk, "source": file}
                    }]
                )

            print(f"✅ Uploaded all chunks from: {file}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    load_and_embed_docs(DATA_FOLDER)
    print("\n🎯 Embedding and upload completed successfully!")
