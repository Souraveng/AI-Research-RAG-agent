from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.retrieval_pipeline import ResearchRetriever
from core.generator import LangChainGenerator
import uvicorn
import time
import os
import json

# Setup
app = FastAPI(title="LangChain Research API")
CACHE_FILE = "./output/api_cache.json"

# Initialize LangChain Components
retriever = ResearchRetriever(k=10) # Get top 10 chunks

generator = LangChainGenerator(model_name="gemini-2.5-pro") 

# Cache logic
api_cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r") as f: 
            api_cache = json.load(f)
    except: 
        pass

class QueryRequest(BaseModel):
    query: str

def get_attribute(obj, attr, default=''):
    """Safely get attribute from object or dictionary."""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    elif isinstance(obj, dict):
        return obj.get(attr, default)
    return default

@app.post("/ask")
async def ask_research_question(request: QueryRequest):
    print(f"\n--- NEW QUERY (LangChain): {request.query} ---")
    start_time = time.time()
    
    # 1. Check Cache
    if request.query.lower().strip() in api_cache:
        print("   -> Returning cached response.")
        return api_cache[request.query.lower().strip()]
    
    try:
        current_query = request.query
        final_response_obj = None
        retrieved_docs = []
        
        # 2. Retrieval Loop (Simulated Agent)
        # We try up to 2 times. If the first answer is "I don't know", we refine the query and try again.
        for attempt in range(2):
            print(f"   -> Retrieving for: {current_query}")
            
            # A. Invoke Retriever
            retrieved_docs = await retriever._async_get_relevant_documents(current_query)
            
            # B. Build Context String
            context_text = "\n".join([d.page_content for d in retrieved_docs])
            
            # C. Generate (Returns Pydantic Object)
            print("   -> Generating with Gemini...")
            final_response_obj = generator.generate_answer(
                request.query, 
                context_text, 
                retrieved_docs
            )
            
            # D. Check Failure
            answer = get_attribute(final_response_obj, 'answer')
            
            # If the answer is useful, stop the loop. 
            # If it says "documents do not contain", try refining the query once.
            if "does not contain" not in answer.lower():
                break
            
            if attempt == 0:
                print("   -> Answer not found. Refining query...")
                current_query = generator.refine_query(request.query, context_text)
                if not current_query: # If refinement fails, stop
                    break

        # 3. Format Response
        # Safely extract metadata from LangChain Documents
        sources = []
        for doc in retrieved_docs:
            sources.append(doc.metadata)
        
        response_data = {
            "answer": get_attribute(final_response_obj, 'answer'),
            "thought": get_attribute(final_response_obj, 'thought'),
            "sources": sources
        }

        # Save Cache
        api_cache[request.query.lower().strip()] = response_data
        with open(CACHE_FILE, "w") as f: 
            json.dump(api_cache, f, indent=4)
        
        print(f"--- Finished in {time.time() - start_time:.2f}s ---")
        return response_data

    except Exception as e:
        print(f"Error: {e}")
        # Return a clean error to the UI instead of crashing
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)