import os
import nest_asyncio
from llama_parse import LlamaParse, ResultType
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

api_key = os.getenv("LLAMA_CLOUD_API_KEY")
if not api_key:
    raise ValueError(" API Key not found in .env file")

# Setup Parser (Optimized for Image Extraction)
parser = LlamaParse(
    api_key=api_key,
    result_type=ResultType.MD,
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="openai-gpt4o",
    verbose=True,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

input_dir = os.path.join(PROJECT_ROOT, "data")
output_img_dir = os.path.join(PROJECT_ROOT, "output", "images")
os.makedirs(output_img_dir, exist_ok=True)

def extract_images_only():
    files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    
    print(f" Found {len(files)} PDFs. Checking image status...")

    for filename in files:
        paper_name = filename.replace(".pdf", "")
        specific_img_dir = os.path.join(output_img_dir, paper_name)

        # --- RESUME LOGIC (IMAGES) ---
        # 1. Check if the specific folder exists
        # 2. Check if the folder is not empty (contains images)
        if os.path.exists(specific_img_dir) and len(os.listdir(specific_img_dir)) > 0:
            print(f" Skipping {filename} - Images already extracted.")
            continue

        file_path = os.path.join(input_dir, filename)
        print(f"\n Extracting Images from: {filename}...")

        try:
            # Create the folder if it doesn't exist yet
            os.makedirs(specific_img_dir, exist_ok=True)
            
            # get_json_result is required to access the image objects
            json_objs = parser.get_json_result(file_path)
            
            if json_objs:
                # Download images specifically to this paper's folder
                images = parser.get_images(json_objs, download_path=specific_img_dir)
                
                if images:
                    print(f" Saved {len(images)} images to /{paper_name}/")
                else:
                    print(f"ℹ No images found in {filename} (Folder created but empty)")
            else:
                print(f" No data returned for {filename}")

        except Exception as e:
            print(f" Error extracting images from {filename}: {e}")

if __name__ == "__main__":
    extract_images_only()