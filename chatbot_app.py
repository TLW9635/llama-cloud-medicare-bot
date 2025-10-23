import streamlit as st
import os
from llama_index.core import Settings
# CHANGE 1: Import the Gemini LLM component
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

# --- 1. Set Page Configuration and Title ---
st.set_page_config(page_title="TBL Medicare Chatbot", layout="centered")
st.title("Ask the TBL Medicare Knowledge Base 💬 (Powered by Gemini)")

# --- 2. Load API Keys from Streamlit Secrets ---
LLAMA_CLOUD_API_KEY = st.secrets.get("LLAMA_CLOUD_API_KEY")
# CHANGE 2: Reference the Google/Gemini API key name
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not (LLAMA_CLOUD_API_KEY and GEMINI_API_KEY):
    st.error("API Keys not found. Please configure LLAMA_CLOUD_API_KEY and GEMINI_API_KEY in Streamlit Secrets.")
else:
    # Set the environment variables so LlamaIndex/Gemini can use them
    os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_CLOUD_API_KEY
    # CHANGE 3: Set the correct environment variable name for Google's API
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    
    # --- 3. Configure and Load the LlamaCloud Index ---
    @st.cache_resource(show_spinner="Connecting to LlamaCloud Index...")
    def get_chat_engine():
        try:
            # CHANGE 4: Initialize the Gemini LLM
            # NOTE: Change the model name if you use something other than gemini-2.5-flash
            Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.1)

            # Connect to your EXISTING LlamaCloud Index
            index_name = "TBL Medicare Knowledge Base" 
            index = LlamaCloudIndex(name=index_name)

            # Create the chat engine (a conversational interface)
            chat_engine = index.as_chat_engine(
                chat_mode="condense_question",
                verbose=True,
            )
            return chat_engine
        except Exception as e:
            st.error(f"Error loading LlamaCloud Index or Gemini model. Details: {e}")
            return None

    chat_engine = get_chat_engine()
    
    if chat_engine:
        # --- 4. Initialize Chat History ---
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I am your TBL Medicare expert, powered by Gemini. How can I help you today?"}
            ]

        # --- 5. User Input and Response Loop ---
        if prompt := st.chat_input("Ask a question about Medicare policies..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

        # Display all chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Generate a new response if the last message was from the user
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Searching the Medicare Knowledge Base..."):
                    try:
                        # Get response from Llama Cloud Index
                        response = chat_engine.chat(prompt)
                        st.markdown(response.response)
                        st.session_state.messages.append({"role": "assistant", "content": response.response})
                    except Exception as e:
                        st.error(f"A chat error occurred while retrieving data: {e}")