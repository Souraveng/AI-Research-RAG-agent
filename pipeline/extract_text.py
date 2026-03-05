import os
import nest_asyncio
from llama_parse import LlamaParse, ResultType
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

api_key = os.getenv("LLAMA_CLOUD_API_KEY")
if not api_key:
    raise ValueError(" API Key not found in .env file")

# Setup Parser
parser = LlamaParse(
    api_key=api_key,
    result_type=ResultType.MD,
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="openai-gpt4o",
    verbose=True,
)

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FOLDER)

input_dir = os.path.join(PROJECT_ROOT, "data")
output_dir = os.path.join(PROJECT_ROOT, "output")
os.makedirs(output_dir, exist_ok=True)

def extract_markdown():
    files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    
    print(f"Found {len(files)} PDFs. Checking status...")

    for filename in files:
        md_filename = filename.replace(".pdf", ".md")
        md_path = os.path.join(output_dir, md_filename)
        
        # Resume Logic
        if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
            print(f"Skipping {filename} - Text already exists.")
            continue

        file_path = os.path.join(input_dir, filename)
        print(f"\n Extracting Text from: {filename}...")

        try:
            documents = parser.load_data(file_path)
            
            if documents:
                final_text = ""
                # Iterate through documents (usually 1 doc = 1 page in LlamaParse)
                for i, doc in enumerate(documents):
                    # Try to get page label from metadata, else use index
                    page_num = doc.metadata.get('page_label', i + 1)
                    
                    # Insert a clear Page Marker that we can regex split later
                    final_text += f"\n\n<!-- PAGE_{page_num} -->\n\n"
                    final_text += doc.text

                if len(final_text.strip()) > 0:
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(final_text)
                    print(f" Saved Markdown with Page Markers: {md_filename}")
                else:
                    print(f"Warning: Parsed text was empty for {filename}")
            else:
                print(f" No content returned for {filename}")

        except Exception as e:
            print(f" Error extracting {filename}: {e}")

if __name__ == "__main__":
    extract_markdown()