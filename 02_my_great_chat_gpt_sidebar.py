import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,  # システムメッセージ
    HumanMessage,  # 人間の質問
    AIMessage  # ChatGPTの返答
)

from langchain.callbacks import get_openai_callback

def init_page():
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤖", layout="wide"
    )
    st.header("My Great ChatGPT 🤖")
    st.sidebar.title("Options")

def init_messages():
    clear_button = st.sidebar.button("Clear messages", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="ユーザーが最初に入力した言語で回答してください")
        ]
        st.session_state.costs = []

def select_model():
    model = st.sidebar.radio("Select model", ["GPT-3.5", "GPT-4"])
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"
    
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    
    return ChatOpenAI(model_name=model_name, temperature=temperature)

def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost

def main():
    init_page()

    llm = select_model()
    init_messages()

    # ユーサーの入力を監視
    if user_input := st.chat_input("聞きたいことを入力してね"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Chat GPT is typing..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    # メッセージを表示
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

    # コストを表示
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"-${cost:.5f}")

if __name__ == "__main__":
    main()


