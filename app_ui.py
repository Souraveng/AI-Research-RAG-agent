import streamlit as st
import requests
import os

st.set_page_config(page_title="Research Navigator", layout="wide")

st.title("Research Paper Navigator (Gemini Multimodal)")
st.markdown("Query your papers. Ask directly for diagrams and explanations!")

API_URL = "http://127.0.0.1:8000/ask"

if "messages" not in st.session_state:
    st.session_state.messages =[]

# Display Chat History (Bug Fixed: Cleanly renders thoughts vs answers)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("thought"):
            with st.expander("View Reasoning"):
                st.write(message["thought"])
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Enter your question..."):
    # Add User Message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("AI is thinking & analyzing visuals..."):
        try:
            response_res = requests.post(API_URL, json={"query": prompt}, timeout=1000)
            
            if response_res.status_code == 200:
                response = response_res.json()
                answer_content = response.get("answer", "No answer provided.")
                thought_content = response.get("thought", "")
                sources = response.get("sources",[])

                with st.chat_message("assistant"):
                    # 1. Render Thought
                    if thought_content:
                        with st.expander("View Reasoning"):
                            st.write(thought_content)
                    
                    # 2. Render Text Answer
                    st.markdown(answer_content)

                    # 3. Render Sources & Images
                    if sources:
                        st.markdown("---")
                        st.markdown("### Evidence Found:")
                        cols = st.columns(3)
                        for i, src in enumerate(sources[:3]):
                            with cols[i]:
                                if src.get("type") == "image":
                                    img_path = src.get("image_path")
                                    if img_path and os.path.exists(img_path):
                                        st.image(img_path, caption=f"Figure P.{src.get('page_no')}")
                                    else:
                                        st.warning(f"Image not found: {img_path}")
                                else:
                                    st.info(f"Text: {src.get('title')} (Page {src.get('page_no')})")

                # Save everything neatly into session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer_content,
                    "thought": thought_content
                })
            else:
                st.error(f"API Error: {response_res.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to API. Is 'main_api.py' running?")
        except Exception as e:
            st.error(f"Error: {e}")