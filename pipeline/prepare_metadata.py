import os
import json
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

md_dir = os.path.join(PROJECT_ROOT, "output")
caption_file = os.path.join(PROJECT_ROOT, "output", "image_captions.json")
output_registry = os.path.join(PROJECT_ROOT, "output", "master_registry.json")

def extract_page_number(filename):
    """
    Extracts page number from typical LlamaParse image filenames.
    Pattern examples: '...-page_12.jpg' or 'page_12.jpg'
    """
    match = re.search(r"page_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None

def generate_registry():
    registry = {}

    # 1. Load image captions
    captions = {}
    if os.path.exists(caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            captions = json.load(f)
    else:
        print(" No caption file found. Run image_caption.py first.")

    # 2. Scan all Markdown files
    files = [f for f in os.listdir(md_dir) if f.endswith(".md")]
    
    for filename in files:
        # Basic metadata from filename
        parts = filename.replace(".md", "").split("_")
        
        # Robust parsing for 2023_Title_Author format
        year = parts[0] if parts[0].isdigit() else "Unknown"
        # Join middle parts as title if title contains underscores
        title = "_".join(parts[1:-1]) if len(parts) > 2 else (parts[1] if len(parts) > 1 else "Unknown")
        author = parts[-1] if len(parts) > 2 else "Unknown"

        paper_id = filename.replace(".md", "")
        
        # 3. Find images belonging to THIS paper
        paper_images = []
        
        for img_path, desc in captions.items():
            # Check if this image belongs to the current paper ID
            # Normalize paths to handle windows backslashes
            normalized_path = img_path.replace("\\", "/")
            
            if paper_id in normalized_path:
                page_num = extract_page_number(normalized_path)
                
                image_entry = {
                    "path": normalized_path,
                    "caption": desc,
                    "modality": "image",
                    "page_no": page_num if page_num else "unknown",
                    # Placeholder: Advanced logic needed to determine section
                    "section": "figures" 
                }
                paper_images.append(image_entry)

        # Sort images by page number for cleaner data
        paper_images.sort(key=lambda x: x['page_no'] if isinstance(x['page_no'], int) else 9999)

        # 4. Create the Registry Entry
        registry[paper_id] = {
            "metadata": {
                "paper_id": paper_id,
                "year": year,
                "title": title.replace("_", " "),
                "author": author,
                "source_file": f"{paper_id}.pdf",
                "total_images": len(paper_images)
            },
            "text_file": filename,
            "images": paper_images
        }

    # 5. Save
    with open(output_registry, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=4)
    
    print(f" Master Registry created for {len(registry)} papers.")

if __name__ == "__main__":
    generate_registry()