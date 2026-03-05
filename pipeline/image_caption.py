import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv

load_dotenv()

# 1. SETUP
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f" Using GPU: {torch.cuda.get_device_name(0)}")
print(" Loading BLIP safely (using safetensors)...")

model_id = "Salesforce/blip-image-captioning-base"

# 2. LOAD MODEL SAFELY
processor = BlipProcessor.from_pretrained(model_id)

# use_safetensors=True bypasses the torch.load security error
model = BlipForConditionalGeneration.from_pretrained(
    model_id, 
    use_safetensors=True
).to(device) # type: ignore

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

data_dir = os.path.join(PROJECT_ROOT, "data")
image_base_dir = os.path.join(PROJECT_ROOT, "output", "images")
output_file = os.path.join(PROJECT_ROOT, "output", "image_captions.json")

def get_description(image_path):
    try:
        raw_image = Image.open(image_path).convert('RGB')
        
        inputs = processor(
            images=raw_image, 
            text="a technical diagram showing", 
            return_tensors="pt"
        ).to(device)

        out = model.generate(**inputs, max_new_tokens=100)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        # Clean up the output prefix
        return caption.replace("a technical diagram showing", "").strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    if not os.listdir("./data"):
        print(" No PDFs found in data folder.")
        return

    # Load existing to resume
    data = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # Get paper folders
    folders = [d for d in os.listdir(image_base_dir) if os.path.isdir(os.path.join(image_base_dir, d))]

    for paper in folders:
        path = os.path.join(image_base_dir, paper)
        images = [i for i in os.listdir(path) if i.lower().endswith(('.png', '.jpg'))]

        for img_name in images:
            img_path = os.path.join(path, img_name).replace("\\", "/")
            
            if img_path in data:
                continue

            print(f" Analyzing: {paper}/{img_name}")
            caption = get_description(img_path)
            
            data[img_path] = caption
            
            # Save progress
            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(data, f_out, indent=4)

    print(f"\n All images described! Metadata saved in {output_file}")

if __name__ == "__main__":
    main()