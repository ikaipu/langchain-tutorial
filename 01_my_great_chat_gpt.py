import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

def main():
    llm = ChatOpenAI(temperature=0)

    st.set_page_config(page_title="My Great ChatGPT", page_icon="🤖", layout="wide")

    st.header("My Great ChatGPT 🤖")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="ユーザーが最初に入力した言語で回答してください")
        ]

    if user_input := st.chat_input("聞きたいことを入力してね"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Chat GPT is typing..."):
            ai_response = llm(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=ai_response.content))


    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:
            st.write(f"System message: {message.content}")

if __name__ == "__main__":
    main()

