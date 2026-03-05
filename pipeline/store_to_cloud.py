import json
import os
import uuid
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

INPUT_FILE = os.path.join(PROJECT_ROOT, "output", "vector_database_ready.json")

COLLECTION_NAME = "research_papers"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def get_deterministic_id(chunk_id_str):
    """
    Generates a consistent UUID based on the chunk ID string.
    Ensures that re-running the script doesn't create duplicate vectors with different IDs.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id_str))

def initialize_collection(client):
    print("Initializing collection on Qdrant Cloud...")
    
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' exists. Deleting to recreate...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384, 
            distance=models.Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=200
        )
    )
    
    # --- ADD THIS BLOCK ---
    print("Creating payload index for 'type'...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="type",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

def main():
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("Error: Missing QDRANT_URL or QDRANT_API_KEY in .env file.")
        return

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device}...")
    encoder = SentenceTransformer(MODEL_NAME, device=device)

    initialize_collection(client)

    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Processing {len(chunks)} chunks for upload...")

    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i : i + BATCH_SIZE]
        
        batch_texts = [item['content'] for item in batch]
        embeddings = encoder.encode(batch_texts, convert_to_numpy=True)
        
        points = []
        for idx, item in enumerate(batch):
            # Use deterministic ID
            point_id = get_deterministic_id(item['id'])
            
            points.append(models.PointStruct(
                id=point_id,
                vector=embeddings[idx].tolist(),
                payload=item
            ))

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    print("Data storage complete. All records are hosted on Qdrant Cloud.")

if __name__ == "__main__":
    main()