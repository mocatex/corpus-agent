import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

# setup API
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

model: str = "deepseek-chat"

st.title("CorpusAgent")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# initialize model
if "model" not in st.session_state:
    st.session_state.model = model

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here"):
    # Add a user-message to chat-history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in a chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in a chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state.model,
            messages=[
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})