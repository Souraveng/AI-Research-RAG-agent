import os
import asyncio
import torch
import re
from typing import List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "research_papers"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

class ResearchRetriever(BaseRetriever):
    client: Any = None
    encoder: Any = None
    reranker: Any = None
    collection_name: str = COLLECTION_NAME
    k: int = 10 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   - Loading Retrieval Models on {device}...")
        self.encoder = SentenceTransformer(EMBED_MODEL_NAME, device=device)
        self.reranker = CrossEncoder(RERANK_MODEL_NAME, device=device)

    def extract_page_number(self, query: str) -> Optional[int]:
        """
        Extracts specific page requirements from the user query.
        Matches: "page 3", "page number 3", "p. 3", etc.
        """
        match = re.search(r"page\s*(?:number\s*|\.|no\.?\s*)?(\d+)", query.lower())
        if match:
            return int(match.group(1))
        return None

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        return asyncio.run(self._async_get_relevant_documents(query))

    async def _async_get_relevant_documents(self, query: str) -> List[Document]:
        """
        Logic:
        1. Embed Query.
        2. Detect if user wants a specific Page.
        3. If Page detected -> Hard Filter Qdrant (ONLY search that page).
        4. If No Page -> Standard Semantic Search.
        """
        
        # 1. Encode Query
        query_vector = self.encoder.encode(query).tolist()
        
        # 2. Check for "Page X" constraint
        target_page = self.extract_page_number(query)
        
        # Base filters: We always separate Text and Image search
        text_filter_conditions = [models.FieldCondition(key="type", match=models.MatchValue(value="text"))]
        image_filter_conditions = [models.FieldCondition(key="type", match=models.MatchValue(value="image"))]

        # 3. Apply Metadata Filter if Page detected
        if target_page:
            print(f"   [Smart Search] Detected constraint: Page {target_page}")
            page_filter = models.FieldCondition(key="page_no", match=models.MatchValue(value=target_page))
            text_filter_conditions.append(page_filter)
            image_filter_conditions.append(page_filter)

        # 4. Perform Search
        # Note: If a page is specified, we search strictly within that page.
        tasks = [
            self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=models.Filter(must=text_filter_conditions),
                limit=15, # Fetch 15 text candidates
                with_payload=True
            ),
            self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=models.Filter(must=image_filter_conditions),
                limit=15, # Fetch 15 image candidates
                with_payload=True
            )
        ]
        
        results = await asyncio.gather(*tasks)
        text_hits = results[0].points
        image_hits = results[1].points

        # 5. Fallback Logic
        # If we filtered by Page 3 but found nothing (maybe parsing error or blank page),
        # we should fallback to a global search so we don't return empty results.
        if target_page and not text_hits and not image_hits:
            print("   [Smart Search] strict page filter returned 0 results. Falling back to global search...")
            # Retry without the page filter (recursion or just copy-paste logic for safety)
            # For simplicity, we just strip the page filter and re-run here:
            tasks_fallback = [
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="text"))]),
                    limit=15, with_payload=True
                ),
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="image"))]),
                    limit=15, with_payload=True
                )
            ]
            results_fallback = await asyncio.gather(*tasks_fallback)
            text_hits = results_fallback[0].points
            image_hits = results_fallback[1].points

        combined_hits = text_hits + image_hits

        if not combined_hits:
            return []

        # 6. Reranking (Crucial for sorting the specific page results)
        sentence_pairs = []
        valid_hits = []
        
        for hit in combined_hits:
            content = hit.payload.get("content", "")
            if content:
                sentence_pairs.append([query, content])
                valid_hits.append(hit)

        if sentence_pairs:
            scores = self.reranker.predict(sentence_pairs)
            for i, hit in enumerate(valid_hits):
                hit.score = float(scores[i])
            valid_hits.sort(key=lambda x: x.score, reverse=True)

        # 7. Diversity Selection (Ensure Images exist)
        final_hits = []
        seen_ids = set()
        
        # Grab top text
        text_count = 0
        for hit in valid_hits:
            if hit.payload.get("type") == "text" and text_count < 7:
                final_hits.append(hit)
                seen_ids.add(hit.id)
                text_count += 1
        
        # Grab top images (forcing them in)
        img_count = 0
        for hit in valid_hits:
            if hit.payload.get("type") == "image" and hit.id not in seen_ids and img_count < 5:
                final_hits.append(hit)
                seen_ids.add(hit.id)
                img_count += 1

        # 8. Convert to LangChain Docs
        final_docs = []
        for hit in final_hits:
            final_docs.append(Document(
                page_content=hit.payload.get("content", ""),
                metadata={
                    "source": hit.payload.get("source", "unknown"),
                    "page_no": hit.payload.get("page_no", "unknown"),
                    "type": hit.payload.get("type", "text"),
                    "title": hit.payload.get("title", "unknown"),
                    "image_path": hit.payload.get("image_path", None)
                }
            ))
            
        return final_docs