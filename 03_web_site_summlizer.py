import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,  # システムメッセージ
    HumanMessage,  # 人間の質問
    AIMessage  # ChatGPTの返答
)

from langchain.callbacks import get_openai_callback

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def init_page():
    st.set_page_config(
        page_title="Web Site Summarizer",
        page_icon="🤗"
    )
    st.header("Web Site Summarizer 🤗")
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
    
    return ChatOpenAI(model_name=model_name, temperature=0)

def get_url_input():
    url = st.text_input("URL:", key="input")
    return url

def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_content(url):
    try:
        with st.spinner("Fetching Content ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # fetch text from main (change the below code to filter page)
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write("something wrong")
        return None

def bulid_prompt(content, n_chars=300):
    return f"""以下はとある。Webページのコンテンツである。内容を{n_chars}程度でわかりやすく要約してください。

========

{content[:1000]}

========

日本語で書いてね！
"""

def get_answer(llm, messeges):
    with get_openai_callback() as cb:
        answer = llm(messeges)
    return answer.content, cb.total_cost

def main():
    init_page()

    llm = select_model()
    init_messages()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write("Please enter a valid URL")
            answer = None
        else:
            content = get_content(url)
            if content:
                prompt = bulid_prompt(content)
                st.session_state.messages.append(HumanMessage(content=prompt))
                with st.spinner("Chat GPT is typing..."):
                    answer, cost = get_answer(llm, st.session_state.messages)
                st.session_state.costs.append(cost)
            else:
                answer = None

    if answer:
        with response_container:
            st.markdown("## Summary")
            st.write(answer)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(content)
    
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == "__main__":
    main()