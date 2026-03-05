import os
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Define the Strict Output Structure using Pydantic
class ResearchAnswer(BaseModel):
    thought: str = Field(description="Step-by-step reasoning about the answer and checks for images.")
    answer: str = Field(description="The final answer with citations. If an image is relevant, describe it here.")

class LangChainGenerator:
    def __init__(self, model_name="gemini-2.5-pro"): 
        api_key = os.getenv("GEMINIAPIKEY")
    
        # Initialize Gemini via LangChain
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key, # Use Vertex AI for Gemini access
        )
        
        # Enforce JSON output automatically
        self.structured_llm = self.llm.with_structured_output(ResearchAnswer)
        print(f"   - LangChain Generator initialized: {model_name}")

    def _encode_image(self, image_path):
        """Helper to convert local image to Base64 for Gemini"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def generate_answer(self, query, context_text, source_documents):
        """
        Constructs a Multimodal Prompt (Text + Base64 Images)
        """
        
        # 1. Prepare Text Prompt
        system_prompt = f"""
        You are a highly capable AI Research Assistant integrated into a Multimodal RAG system.
        You have access to both text chunks and actual images retrieved from research papers.

        # INSTRUCTIONS:
        1. You MUST respond in valid JSON format. 
        2. Your JSON response must have exactly two keys: "thought" and "answer".
        3. "thought": Write your step-by-step reasoning here. Check if the provided context contains the answer or the requested image.
        4. "answer": Your final response to the user.
           - Citations: Always cite facts like [Paper Title, Page X].
           - Images: If the user asks for an image/diagram, and you see it in the provided context, ACT AS IF YOU ARE SHOWING IT DIRECTLY. Say "Here is the diagram showing..." and describe its visual details. Do NOT ever say "I cannot show images", "The UI will show it", or "I cannot embed images." The system automatically attaches the image below your text.
           - If the context completely lacks the info, say "The provided documents do not contain this information."

        # EXPECTED JSON OUTPUT FORMAT:
        {{
          "thought": "I need to look for the transformer diagram. I see an image from Attention Is All You Need, page 3. I will describe it.",
          "answer": "Here is the architecture diagram for the Transformer. As you can see, the left side contains the encoder..."
        }}
        """

        # 2. Prepare Message Content (Multimodal)
        message_content: list = [
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": f"User Query: {query}"}
        ]

        # 3. Inject Images
        # We loop through docs, find images, and add them to the message
        added_images = set()
        for doc in source_documents:
            if doc.metadata.get("type") == "image":
                path = doc.metadata.get("image_path")
                if path and path not in added_images and os.path.exists(path):
                    b64_img = self._encode_image(path)
                    if b64_img:
                        # Add image to payload
                        message_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        })
                        added_images.add(path)

        # 4. Invoke LLM
        # We pass a single HumanMessage containing text + images
        msg = HumanMessage(content=message_content)
        
        try:
            # LangChain returns a ResearchAnswer object (thought, answer)
            response_obj = self.structured_llm.invoke([msg])
            return response_obj
        except Exception as e:
            return ResearchAnswer(
                thought="Error during generation.", 
                answer=f"I encountered an error: {str(e)}"
            )

    def refine_query(self, query, context_summary):
        """Simple text-only call for query refinement"""
        prompt = f"Original: {query}\nContext: {context_summary[:500]}\nSuggest a better search query:"
        response = self.llm.invoke(prompt)
        if isinstance(response.content, str):
            return response.content
        elif isinstance(response.content, list) and len(response.content) > 0:
            content = response.content[0]
            return content if isinstance(content, str) else content.get("text", "").strip()
        return ""