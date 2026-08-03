"""
NewsScan Agent — chat UI, deployed as a Databricks App.

Talks to the serving endpoint created in notebook 06 (news_qa_agent_endpoint).
Databricks Apps automatically gives this app its own identity (a service
principal) to call other Databricks resources with — no API keys to manage —
but you MUST manually grant that identity permission to query the serving
endpoint. See the "Databricks App" section of README.md, step App-4.
"""

import os
import streamlit as st
from databricks.sdk import WorkspaceClient

SERVING_ENDPOINT_NAME = os.environ.get("SERVING_ENDPOINT_NAME", "news_qa_agent_endpoint")

st.set_page_config(page_title="NewsScan Agent", page_icon="📰")
st.title("📰 NewsScan Agent")
st.caption("Ask questions about the documents you've ingested. Answers are grounded in your documents, with sources.")

if "history" not in st.session_state:
    st.session_state.history = []

w = WorkspaceClient()


def ask_agent(question: str):
    response = w.serving_endpoints.query(
        name=SERVING_ENDPOINT_NAME,
        dataframe_records=[{"question": question}],
    )
    # predictions is a list with one dict per input row: {"answer": ..., "sources": [...]}
    prediction = response.predictions[0]
    return prediction["answer"], prediction.get("sources", [])


for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Ask something about your documents...")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                answer, sources = ask_agent(question)
                reply = answer
                if sources:
                    reply += "\n\n**Sources:** " + ", ".join(sources)
            except Exception as e:
                reply = f"Something went wrong calling the agent: {e}"
        st.markdown(reply)
    st.session_state.history.append(("assistant", reply))
