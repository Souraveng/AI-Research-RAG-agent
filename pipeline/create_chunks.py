import os
import json
import re
import uuid

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

INPUT_REGISTRY = os.path.join(PROJECT_ROOT, "output", "master_registry.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "output", "vector_database_ready.json")

TARGET_CHUNK_SIZE = 400 
OVERLAP = 50 

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_chunk_id(paper_id, index, type="text"):
    return f"{paper_id}_{index:04d}_{type}"

def estimate_tokens(text):
    return len(text.split()) * 1.3

# --- NEW: Importance Scoring Logic ---
def get_section_importance(section_name):
    """Assigns an importance score based on your Architecture Guide."""
    sec = section_name.lower()
    if "abstract" in sec: return 5
    if "conclusion" in sec or "summary" in sec: return 4
    # Figures are 3 (handled below)
    if "method" in sec or "experiment" in sec or "approach" in sec: return 2
    return 1 # Default body text

def recursive_chunking(text, page_num, base_metadata, current_section):
    """
    Splits text into chunks, attaching the current section and dynamic importance.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    
    current_chunk = []
    current_length = 0
    importance = get_section_importance(current_section)
    
    for sentence in sentences:
        sent_len = estimate_tokens(sentence)
        
        if current_length + sent_len > TARGET_CHUNK_SIZE and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "content": chunk_text,
                "page_no": page_num,
                "section": current_section, # NEW
                "importance": importance,   # NEW
                "token_count": int(current_length)
            })
            
            overlap_buffer = []
            overlap_len = 0
            for s in reversed(current_chunk):
                overlap_buffer.insert(0, s)
                overlap_len += estimate_tokens(s)
                if overlap_len >= OVERLAP:
                    break
            
            current_chunk = overlap_buffer
            current_length = overlap_len

        current_chunk.append(sentence)
        current_length += sent_len

    if current_chunk:
        chunks.append({
            "content": " ".join(current_chunk),
            "page_no": page_num,
            "section": current_section, # NEW
            "importance": importance,   # NEW
            "token_count": int(current_length)
        })
    
    return chunks

def process_papers():
    with open(INPUT_REGISTRY, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    all_chunks = []
    total_text_chunks = 0
    total_image_chunks = 0

    print(f" Processing {len(registry)} papers with Semantic Chunking...")

    for paper_id, data in registry.items():
        md_file = os.path.join(OUTPUT_DIR, data["text_file"])
        base_meta = data["metadata"]
        
        # --- 1. PROCESS TEXT ---
        if os.path.exists(md_file):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            page_splits = re.split(r'<!-- PAGE_(\d+) -->', content)
            
            current_section = "Introduction" # Default starting section
            
            for i in range(1, len(page_splits), 2):
                page_num = int(page_splits[i])
                page_text = page_splits[i+1]
                
                # --- NEW: Look for Markdown Headers (e.g., "## Abstract") ---
                # This simple regex finds lines starting with # and grabs the text
                headers = re.findall(r'^#+\s*(.+)$', page_text, flags=re.MULTILINE)
                if headers:
                    # If we found headers on this page, update our current section to the last one found
                    current_section = headers[-1].strip()
                
                text_sub_chunks = recursive_chunking(page_text, page_num, base_meta, current_section)
                
                for chunk in text_sub_chunks:
                    # Treat exact abstract matches slightly differently if needed, 
                    # but our logic already scored it a 5.
                    chunk_obj = {
                        "id": create_chunk_id(paper_id, total_text_chunks, "text"),
                        "paper_id": paper_id,
                        "title": base_meta["title"],
                        "year": base_meta["year"],
                        "source": base_meta["source_file"],
                        "type": "text",
                        "content": chunk["content"],
                        "page_no": chunk["page_no"],
                        "section": chunk["section"],       # Added
                        "importance": chunk["importance"]  # Added dynamic score
                    }
                    all_chunks.append(chunk_obj)
                    total_text_chunks += 1
        
        # --- 2. PROCESS IMAGES ---
        for img in data.get("images", []):
            chunk_obj = {
                "id": create_chunk_id(paper_id, total_image_chunks, "image"),
                "paper_id": paper_id,
                "title": base_meta["title"],
                "year": base_meta["year"],
                "source": base_meta["source_file"],
                "type": "image",
                "content": img["caption"], 
                "image_path": img["path"],
                "page_no": img["page_no"],
                "section": "Figures", # Explicitly label as Figures
                "importance": 3 # Architecture guide: figures > methods (2)
            }
            all_chunks.append(chunk_obj)
            total_image_chunks += 1

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4)

    print(f"\n Semantic Chunking Complete!")
    print(f" Text Chunks:  {total_text_chunks}")
    print(f" Image Chunks: {total_image_chunks}")
    print(f" Total Chunks: {len(all_chunks)}")
    print(f" Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_papers()