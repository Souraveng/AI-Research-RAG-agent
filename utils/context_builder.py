import json 

class ContextBuilder:
    def __init__(self, max_tokens=2000):
        self.max_tokens = max_tokens

    def build(self, search_results):
        """
        Stage 6: Context Builder
        Formats search results into a structured JSON string for the LLM.
        """
        context_list = []
        
        # Guide Rule: Ensure diversity and remove redundancy
        seen_content = set()

        for res in search_results:
            p = res.payload
            content = p.get('content', '')

            # Basic de-duplication
            if content[:50] in seen_content:
                continue
            seen_content.add(content[:50])

            # --- MODIFIED: Create a dictionary instead of an XML string ---
            struct = {
                "title": p.get('title', 'Unknown'),
                "year": p.get('year', 'Unknown'),
                "page": p.get('page_no', 'Unknown'),
                "modality": p.get('type', 'text'),
                "content": content
            }
            context_list.append(struct)

        # --- MODIFIED: Convert the list of dictionaries to a JSON string ---
        # Using indent=2 makes it nicely formatted and easier for the LLM to read
        final_context = json.dumps(context_list, indent=2)
        
        # Stage 6 Rule: Relevant context > more context
        # (This check should ideally be token-based, but for now, it's handled by the reranker)
        return final_context

# Example Usage:
# builder = ContextBuilder()
# prompt_ready_context = builder.build(final_hits)
# print(prompt_ready_context)