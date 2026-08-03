"""
NewsScan Agent — chat UI, deployed as a Databricks App.

Talks to the serving endpoint created in notebook 06 (news_qa_agent_endpoint).
Databricks Apps automatically gives this app its own identity (a service
principal) to call other Databricks resources with — no API keys to manage —
but you MUST manually grant that identity permission to query the serving
endpoint. See the "Databricks App" section of README.md, step App-4.

ACCESS CONTROL: a user can use this app if EITHER they're individually listed in
ALLOWED_USERS, OR they belong to the ALLOWED_GROUP_NAME group. See the
"Access control" section of README.md for the one-time manual setup the group
check requires (creating the group, adding members, and granting the app's
service principal permission to read group membership). The individual list
needs no special permission at all — useful for letting yourself in while
that permission grant is still pending, or for one-off access without
touching group membership.
"""

import os
import streamlit as st
from databricks.sdk import WorkspaceClient

SERVING_ENDPOINT_NAME = os.environ.get("SERVING_ENDPOINT_NAME", "news_qa_agent_endpoint")
ALLOWED_GROUP_NAME = os.environ.get("ALLOWED_GROUP_NAME", "newsscan-agent-users")
# Comma-separated list of individual emails, e.g. "me@company.com, other@company.com"
ALLOWED_USERS = {
    u.strip().lower()
    for u in os.environ.get("ALLOWED_USERS", "").split(",")
    if u.strip()
}

st.set_page_config(page_title="NewsScan Agent", page_icon="📰")


@st.cache_resource
def get_workspace_client():
    return WorkspaceClient()


def get_current_user_email():
    """Databricks Apps always forwards the logged-in user's email in this
    header — no special auth mode needs to be enabled for this part."""
    return st.context.headers.get("x-forwarded-email")


def is_user_in_allowed_group(email: str, group_name: str) -> bool | None:
    """Returns True/False, or None if the check itself failed (e.g. the app's
    service principal doesn't have permission to read group membership --
    see README for the permission grant this requires)."""
    if not email:
        return False
    w = get_workspace_client()
    try:
        matches = list(w.users.list(filter=f"userName eq '{email}'", attributes="id,userName,groups"))
        if not matches:
            return False
        user_groups = [g.display for g in (matches[0].groups or [])]
        return group_name in user_groups
    except Exception:
        return None


def check_access(email: str) -> bool | None:
    """Returns True (allowed), False (denied), or None (couldn't determine --
    the group check failed and the user isn't individually allow-listed
    either, so we genuinely don't know)."""
    if email.lower() in ALLOWED_USERS:
        return True
    return is_user_in_allowed_group(email, ALLOWED_GROUP_NAME)


# ---- Access control ----
user_email = get_current_user_email()

if not user_email:
    st.error("🚫 Could not identify who you are. Access denied.")
    st.stop()

access = check_access(user_email)

if access is None:
    st.error(
        "⚠️ Couldn't verify access — you're not on the individual allow-list, "
        "and the app doesn't have permission to check group membership yet. "
        "See the README's access-control setup step (granting the app's "
        "service principal a role with SCIM read access), or ask an admin to "
        "add your email to ALLOWED_USERS in app.yaml as a quick workaround."
    )
    st.stop()

if access is False:
    st.error(
        f"🚫 Access denied. You must either be individually allow-listed or a "
        f"member of the **{ALLOWED_GROUP_NAME}** group to use this agent."
        f"\n\nSigned in as: {user_email}"
    )
    st.stop()
# ---- End access control ----

st.title("📰 NewsScan Agent")
st.caption("Ask questions about the documents you've ingested. Answers are grounded in your documents, with sources.")
st.caption(f"Signed in as {user_email}")

if "history" not in st.session_state:
    st.session_state.history = []

w = get_workspace_client()


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
