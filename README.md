# 🧭 AI Research Paper Navigator

An advanced **Multimodal Retrieval-Augmented Generation (RAG)** system designed to help you interact with, query, and extract insights from academic research papers. 

Unlike standard RAG systems, this project processes and retrieves **both text and images/diagrams**, allowing you to ask questions like: *"Show me the Transformer architecture diagram from page 3 and explain how the encoder works."*

---

## ✨ Key Features
- **🖼️ Multimodal Retrieval:** Extracts and indexes text chunks, images, and image captions separately.
- **🎯 Smart Metadata Filtering:** Automatically detects page number requests (e.g., "page 3") using regex and filters the Vector DB to guarantee precise results.
- **🧠 Advanced Reranking:** Uses Cross-Encoder models (`BAAI/bge-reranker-base`) to score and prioritize the most relevant retrieved text and images.
- **⚡ Fast & Structured LLM:** Powered by **Google Gemini 1.5 Flash** via LangChain, enforcing strict JSON outputs for perfectly formatted citations and reasoning steps.
- **💬 Interactive Chat UI:** A sleek, conversational interface built with Streamlit that renders text, markdown, expandable "AI Thoughts", and retrieved images.

---

## 🛠️ Technology Stack
- **Frontend:** Streamlit
- **Backend API:** FastAPI & Uvicorn
- **LLM Engine:** Google Gemini 1.5 Flash (via LangChain)
- **Vector Database:** Qdrant Cloud
- **Embeddings:** `BAAI/bge-small-en-v1.5`
- **Reranker:** `BAAI/bge-reranker-base`
- **PDF Extraction:** LlamaParse (LlamaIndex)
- **Image Captioning:** Salesforce BLIP (`Salesforce/blip-image-captioning-base`)

---

## 📁 Project Structure

```text
AI-Research-Navigator/
├── .env                     # API Keys (Not tracked in Git)
├── .gitignore               # Git ignore file
├── Dockerfile               # Production Docker configuration
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
├── app_ui.py                # Streamlit Frontend
├── main_api.py              # FastAPI Backend Entry Point
│
├── core/                    # Core Engine
│   ├── __init__.py
│   ├── generator.py         # LangChain/Gemini prompt & generation logic
│   └── retrieval_pipeline.py# Qdrant hybrid search & reranking logic
│
├── pipeline/                # Data Ingestion Pipeline (Run once per new PDF)
│   ├── __init__.py
│   ├── extract_text.py      # Parses PDF to Markdown
│   ├── extract_images.py    # Extracts images directly from PDF
│   ├── image_caption.py     # Uses BLIP to generate text descriptions of images
│   ├── prepare_metadata.py  # Links text, images, and pages into a master registry
│   ├── create_chunks.py     # Semantic chunking of text and images
│   ├── store_to_cloud.py    # Embeds and uploads chunks to Qdrant Cloud
│   └── fix_index.py         # Creates Qdrant Payload Indexes for strict filtering
│
├── data/                    # Put raw .pdf files here
└── output/                  # Generated .md files, .json databases, and /images/
