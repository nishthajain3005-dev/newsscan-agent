"""
NewsScan Agent — chat UI, deployed as a Databricks App.

Talks to the serving endpoint created in notebook 06 (news_qa_agent_endpoint).
Databricks Apps automatically gives this app its own identity (a service
principal) to call other Databricks resources with — no API keys to manage —
but you MUST manually grant that identity permission to query the serving
endpoint. See the "Databricks App" section of README.md, step App-4.

ACCESS CONTROL: a user can use this app if EITHER they're individually listed
in ALLOWED_USERS, OR they belong to ANY of the groups listed in
ALLOWED_GROUPS (a comma-separated list, e.g. "data-engineers,data-scientists,
business-analysts,project-managers" -- the demo set is exactly this: four
role-based groups, standing in for whatever real grouping the business
settles on later. Adding/removing a role later is just editing this one list
in app.yaml, no code change). See the "Access control" section of README.md
for the one-time manual setup the group check requires (creating the groups,
adding members, and granting the app's service principal permission to read
group membership). The individual list needs no special permission at all --
useful for letting yourself in while that permission grant is still pending,
or for one-off access without touching group membership.
"""

import os
import streamlit as st
from databricks.sdk import WorkspaceClient

SERVING_ENDPOINT_NAME = os.environ.get("SERVING_ENDPOINT_NAME", "news_qa_agent_endpoint")

# Comma-separated list of group names -- membership in ANY ONE of these grants
# access. Demo default: four role-based groups. Change freely later -- this is
# the one place that needs editing to add/remove/rename which roles are allowed.
ALLOWED_GROUPS = {
    g.strip()
    for g in os.environ.get(
        "ALLOWED_GROUPS",
        "data-engineers,data-scientists,business-analysts,project-managers",
    ).split(",")
    if g.strip()
}

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


def get_user_groups(email: str) -> set[str] | None:
    """Returns the set of group names this user belongs to, or None if the
    lookup itself failed (e.g. the app's service principal doesn't have
    permission to read group membership yet -- see README)."""
    if not email:
        return set()
    w = get_workspace_client()
    try:
        matches = list(w.users.list(filter=f"userName eq '{email}'", attributes="id,userName,groups"))
        if not matches:
            return set()
        return {g.display for g in (matches[0].groups or [])}
    except Exception:
        return None


def check_access(email: str) -> tuple[bool | None, set[str]]:
    """Returns (True/False/None, matched_groups). None means the check
    couldn't be completed (see get_user_groups) and the user also isn't
    individually allow-listed, so we genuinely don't know."""
    if email.lower() in ALLOWED_USERS:
        return True, set()

    user_groups = get_user_groups(email)
    if user_groups is None:
        return None, set()

    matched = user_groups & ALLOWED_GROUPS
    return (bool(matched), matched)


# ---- Access control ----
user_email = get_current_user_email()

if not user_email:
    st.error("🚫 Could not identify who you are. Access denied.")
    st.stop()

access, matched_groups = check_access(user_email)

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
        f"member of one of these groups: {', '.join(sorted(ALLOWED_GROUPS))}."
        f"\n\nSigned in as: {user_email}"
    )
    st.stop()
# ---- End access control ----

st.title("📰 NewsScan Agent")
st.caption("Ask questions about the documents you've ingested. Answers are grounded in your documents, with sources.")
role_note = f" ({', '.join(sorted(matched_groups))})" if matched_groups else ""
st.caption(f"Signed in as {user_email}{role_note}")

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