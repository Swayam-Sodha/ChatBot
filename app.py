import streamlit as st
from chatbot import chatbot_semantic, chatbot_hybrid

st.set_page_config(
    page_title="NLP Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 NLP Chatbot")
st.caption("Hybrid NLP Chatbot using Transformers")

st.sidebar.header("⚙️ Chatbot Mode")

mode = st.sidebar.radio(
    "Select Version",
    ("V1 – Dialogue Dataset Based", "V2 – Advanced Conversational")
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if mode == "V1 – Dialogue Dataset Based":
                response = chatbot_semantic(user_input)
            else:
                response = chatbot_hybrid(user_input)

            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
