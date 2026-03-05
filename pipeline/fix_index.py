import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

# Load Env Variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "research_papers"

def fix_index():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    print(f"Checking collection '{COLLECTION_NAME}'...")
    
    try:
        # 1. Index 'type' (text vs image)
        print(" -> Creating Keyword index for field: 'type'...")
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="type",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        # 2. Index 'page_no' (CRITICAL FIX FOR YOUR ERROR)
        print(" -> Creating Integer index for field: 'page_no'...")
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="page_no",
            field_schema=models.PayloadSchemaType.INTEGER
        )

        print("\nSUCCESS! Indexes created. You can now run 'main_api.py' and ask for specific pages.")
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    fix_index()