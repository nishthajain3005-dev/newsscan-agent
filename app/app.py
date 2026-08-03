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
business-analysts,data-analysts,project-managers" -- the demo set is exactly
this: five role-based groups, standing in for whatever real grouping the
business settles on later. Adding/removing a role later is just editing this
one list in app.yaml, no code change). See the "Access control" section of
README.md for the one-time manual setup the group check requires (creating
the groups, adding members, and granting the app's service principal
permission to read group membership). The individual list needs no special
permission at all -- useful for letting yourself in while that permission
grant is still pending, or for one-off access without touching group
membership.

DOCUMENT-LEVEL ACCESS POLICIES: on top of app access above, the "Upload" tab
lets you set/apply policies about WHAT can be uploaded and WHO can see it once
ingested -- e.g. "only Project Managers may upload here, content must actually
be a resume, and only Data Engineers + Data Analysts can ever query it back".
Policies live in the `upload_policies` Delta table (see
notebooks/00b_setup_access_policies.py) and are read live on every visit -- no
redeploy needed to add/edit a policy. See the README's "Document-level upload
policies" section for the one-time SQL Warehouse permission grant this tab
needs.
"""

import io
import os
import re
import uuid

import streamlit as st
from databricks.sdk import WorkspaceClient

SERVING_ENDPOINT_NAME = os.environ.get("SERVING_ENDPOINT_NAME", "news_qa_agent_endpoint")
CHAT_MODEL_ENDPOINT = os.environ.get("CHAT_MODEL_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
SQL_WAREHOUSE_ID = os.environ.get("SQL_WAREHOUSE_ID", "")

CATALOG = os.environ.get("CATALOG", "news_agent")
SCHEMA = os.environ.get("SCHEMA", "docs")
VOLUME = os.environ.get("VOLUME", "raw_files")
POLICIES_TABLE = f"{CATALOG}.{SCHEMA}.upload_policies"
MANIFEST_TABLE = f"{CATALOG}.{SCHEMA}.upload_manifest"

# Comma-separated list of group names -- membership in ANY ONE of these grants
# access. Demo default: five role-based groups. Change freely later -- this is
# the one place that needs editing to add/remove/rename which roles are allowed.
ALLOWED_GROUPS = {
    g.strip()
    for g in os.environ.get(
        "ALLOWED_GROUPS",
        "data-engineers,data-scientists,business-analysts,data-analysts,project-managers",
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


# ---- SQL Warehouse helpers (used by the Upload tab to read policies / write the manifest) ----

def run_query(statement: str):
    """Runs a SQL statement against SQL_WAREHOUSE_ID and returns rows as a list
    of dicts. Requires the app's service principal to have "Can Use" on that
    warehouse -- see README "Document-level upload policies" section."""
    if not SQL_WAREHOUSE_ID:
        raise RuntimeError("SQL_WAREHOUSE_ID is not set in app.yaml -- the Upload tab needs a SQL Warehouse to read policies / write the upload log.")
    w = get_workspace_client()
    resp = w.statement_execution.execute_statement(
        statement=statement, warehouse_id=SQL_WAREHOUSE_ID, catalog=CATALOG, schema=SCHEMA, wait_timeout="30s",
    )
    if not resp.result or not resp.result.data_array:
        return []
    columns = [c.name for c in resp.manifest.schema.columns]
    return [dict(zip(columns, row)) for row in resp.result.data_array]


def run_statement(statement: str):
    """Fire-and-forget SQL (INSERT/UPDATE) against SQL_WAREHOUSE_ID."""
    if not SQL_WAREHOUSE_ID:
        raise RuntimeError("SQL_WAREHOUSE_ID is not set in app.yaml -- the Upload tab needs a SQL Warehouse to read policies / write the upload log.")
    w = get_workspace_client()
    w.statement_execution.execute_statement(
        statement=statement, warehouse_id=SQL_WAREHOUSE_ID, catalog=CATALOG, schema=SCHEMA, wait_timeout="30s",
    )


def sql_string_literal(value: str) -> str:
    """Escapes a Python string for safe embedding as a SQL string literal."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def sql_array_literal(values) -> str:
    return "array(" + ", ".join(sql_string_literal(v) for v in values) + ")"


def parse_sql_array(value):
    """Databricks SQL array columns come back from statement_execution as
    Python lists already in most SDK versions, but guard against a
    string-encoded fallback (e.g. '[\"a\", \"b\"]' or 'a,b')."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip("[]")
        return [v.strip().strip('"').strip("'") for v in stripped.split(",") if v.strip()]
    return []


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


def get_eligible_policies(user_groups: set[str], is_individually_allowed: bool):
    """Returns active policies this user is allowed to upload under: their
    groups overlap the policy's uploader_groups. Individually-allow-listed
    users (no group info to check) can see every active policy -- a
    deliberate simplification worth tightening if you need per-user policy
    restriction beyond groups."""
    rows = run_query(f"SELECT * FROM {POLICIES_TABLE} WHERE active = true")
    eligible = []
    for r in rows:
        uploader_groups = set(parse_sql_array(r.get("uploader_groups")))
        if is_individually_allowed or (user_groups & uploader_groups):
            eligible.append(r)
    return eligible


def extract_text_for_check(filename: str, file_bytes: bytes) -> str:
    """Best-effort text extraction for the content-classification step.
    Returns '' on any failure -- callers should treat that as
    "couldn't verify content" and fail closed (reject)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        if ext == "pdf":
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)[:8000]
        elif ext == "docx":
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs)[:8000]
        elif ext in ("txt", "html", "htm"):
            return file_bytes.decode("utf-8", errors="ignore")[:8000]
    except Exception:
        return ""
    return ""


def classify_document(text: str, allowed_categories: list[str]) -> tuple[bool, str]:
    """Asks the chat model endpoint whether the document's content matches ANY
    of allowed_categories. Returns (matches, raw_model_reply). Fails closed
    (False) if the endpoint call itself errors, rather than silently letting
    an unverifiable document through."""
    if not text.strip():
        return False, "no extractable text"

    categories_str = " / ".join(allowed_categories)
    prompt = (
        f"You are a strict document classifier. Categories: {categories_str}.\n"
        f"Does the document below clearly belong to ANY of those categories? "
        f"Answer with exactly one word: yes or no.\n\n"
        f"Document:\n{text}"
    )
    w = get_workspace_client()
    try:
        resp = w.serving_endpoints.query(
            name=CHAT_MODEL_ENDPOINT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        reply = resp.choices[0].message.content.strip().lower()
        return reply.startswith("yes"), reply
    except Exception as e:
        return False, f"classification call failed: {e}"


def upload_document(file_name: str, file_bytes: bytes, policy: dict, uploaded_by: str):
    """Writes the file to the Volume under a policy-specific subfolder and
    records an 'accepted' manifest row. Raises on failure -- callers should
    catch and show st.error."""
    policy_id = policy["policy_id"]
    viewer_groups = parse_sql_array(policy.get("viewer_groups"))
    allowed_categories = parse_sql_array(policy.get("allowed_categories"))
    category = allowed_categories[0] if allowed_categories else "general"

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", file_name)
    volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{policy_id}/{safe_name}"

    w = get_workspace_client()
    w.files.upload(volume_path, io.BytesIO(file_bytes), overwrite=True)

    manifest_id = str(uuid.uuid4())
    run_statement(f"""
        INSERT INTO {MANIFEST_TABLE}
        (manifest_id, file_path, original_filename, policy_id, category, viewer_groups, uploaded_by, uploaded_at, status, rejection_reason)
        VALUES (
            {sql_string_literal(manifest_id)}, {sql_string_literal(volume_path)}, {sql_string_literal(file_name)},
            {sql_string_literal(policy_id)}, {sql_string_literal(category)}, {sql_array_literal(viewer_groups)},
            {sql_string_literal(uploaded_by)}, current_timestamp(), 'accepted', NULL
        )
    """)
    return volume_path


def log_rejected_upload(file_name: str, policy: dict, uploaded_by: str, reason: str):
    """Best-effort audit log of a rejected upload attempt -- swallow errors so
    a manifest-table hiccup never blocks showing the rejection message itself."""
    try:
        manifest_id = str(uuid.uuid4())
        policy_id = policy["policy_id"] if policy else None
        run_statement(f"""
            INSERT INTO {MANIFEST_TABLE}
            (manifest_id, file_path, original_filename, policy_id, category, viewer_groups, uploaded_by, uploaded_at, status, rejection_reason)
            VALUES (
                {sql_string_literal(manifest_id)}, NULL, {sql_string_literal(file_name)},
                {sql_string_literal(policy_id) if policy_id else 'NULL'}, NULL, array(),
                {sql_string_literal(uploaded_by)}, current_timestamp(), 'rejected', {sql_string_literal(reason)}
            )
        """)
    except Exception:
        pass


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
role_note = f" ({', '.join(sorted(matched_groups))})" if matched_groups else ""
st.caption(f"Signed in as {user_email}{role_note}")

# Full group membership (may be a superset of matched_groups, which is only
# the intersection with ALLOWED_GROUPS) -- this is what's sent to the agent
# for role-based document filtering, and what's checked against each upload
# policy's uploader_groups.
_full_groups = get_user_groups(user_email)
user_full_groups = _full_groups if _full_groups is not None else matched_groups
is_individually_allowed = user_email.lower() in ALLOWED_USERS

if "history" not in st.session_state:
    st.session_state.history = []

w = get_workspace_client()

ask_tab, upload_tab = st.tabs(["💬 Ask", "📤 Upload"])


def ask_agent(question: str):
    response = w.serving_endpoints.query(
        name=SERVING_ENDPOINT_NAME,
        dataframe_records=[{"question": question, "user_groups": ",".join(sorted(user_full_groups))}],
    )
    # predictions is a list with one dict per input row: {"answer": ..., "sources": [...]}
    prediction = response.predictions[0]
    return prediction["answer"], prediction.get("sources", [])


with ask_tab:
    st.caption("Ask questions about the documents you've ingested. Answers are grounded in your documents, with sources — and only ever pull from documents your role is allowed to see.")

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


with upload_tab:
    st.caption(
        "Upload documents directly into the pipeline's Volume. Which policies you see, what "
        "you're allowed to upload, and who can later see/query that content are all driven by "
        "the `upload_policies` table — ask an admin to add a policy for your team if you don't "
        "see one here."
    )

    if not SQL_WAREHOUSE_ID:
        st.warning(
            "⚠️ Upload isn't configured yet — `SQL_WAREHOUSE_ID` is missing from app.yaml. "
            "See the README's \"Document-level upload policies\" section: create/pick a SQL "
            "Warehouse, add its ID to app.yaml, and grant this app's service principal "
            "**Can Use** on it."
        )
    else:
        try:
            policies = get_eligible_policies(user_full_groups, is_individually_allowed)
        except Exception as e:
            policies = []
            st.error(f"Couldn't load upload policies: {e}")

        if not policies:
            st.info(
                "No upload policy currently allows your role to upload here. "
                f"Signed in as {user_email}{role_note}. Ask an admin to add your group to a "
                "policy's `uploader_groups` in the `upload_policies` table."
            )
        else:
            policy_labels = {p["policy_id"]: f"{p['display_name']}" for p in policies}
            selected_id = st.selectbox(
                "Upload policy",
                options=list(policy_labels.keys()),
                format_func=lambda pid: policy_labels[pid],
            )
            selected_policy = next(p for p in policies if p["policy_id"] == selected_id)

            allowed_categories = parse_sql_array(selected_policy.get("allowed_categories"))
            allowed_extensions = parse_sql_array(selected_policy.get("allowed_extensions"))
            viewer_groups = parse_sql_array(selected_policy.get("viewer_groups"))
            require_content_check = bool(selected_policy.get("require_content_check"))

            st.markdown(
                f"**{selected_policy.get('description', '')}**\n\n"
                f"- Accepted file types: `{', '.join(allowed_extensions)}`\n"
                f"- Content must match: `{', '.join(allowed_categories)}`" +
                (" (verified automatically)" if require_content_check else " (not content-checked)") + "\n"
                f"- Visible afterward only to: `{', '.join(viewer_groups)}`"
            )

            uploaded_file = st.file_uploader(
                "Choose a file",
                type=allowed_extensions or None,
                key=f"uploader_{selected_id}",
            )

            if uploaded_file is not None and st.button("Validate & upload"):
                file_bytes = uploaded_file.getvalue()
                file_name = uploaded_file.name
                ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

                if allowed_extensions and ext not in allowed_extensions:
                    reason = f"file type '.{ext}' not allowed by policy '{selected_policy['display_name']}' (allowed: {', '.join(allowed_extensions)})"
                    st.error(f"❌ Can't upload this file — {reason}.")
                    log_rejected_upload(file_name, selected_policy, user_email, reason)
                elif require_content_check:
                    with st.spinner("Checking that the document matches this policy's allowed content..."):
                        text = extract_text_for_check(file_name, file_bytes)
                        matches, model_reply = classify_document(text, allowed_categories)
                    if not matches:
                        reason = f"content doesn't appear to be {'/'.join(allowed_categories)} (classifier said: {model_reply})"
                        st.error(
                            f"❌ Can't upload this file under policy '{selected_policy['display_name']}' — "
                            f"only {'/'.join(allowed_categories)} documents are accepted here."
                        )
                        log_rejected_upload(file_name, selected_policy, user_email, reason)
                    else:
                        try:
                            path = upload_document(file_name, file_bytes, selected_policy, user_email)
                            st.success(
                                f"✅ Uploaded to `{path}`. It'll become visible in Ask answers to "
                                f"`{', '.join(viewer_groups)}` after the next ingestion run "
                                f"(steps 01-04, or the scheduled job)."
                            )
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
                else:
                    try:
                        path = upload_document(file_name, file_bytes, selected_policy, user_email)
                        st.success(
                            f"✅ Uploaded to `{path}`. It'll become visible in Ask answers to "
                            f"`{', '.join(viewer_groups)}` after the next ingestion run "
                            f"(steps 01-04, or the scheduled job)."
                        )
                    except Exception as e:
                        st.error(f"Upload failed: {e}")