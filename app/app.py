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

ADMIN "MANAGE" TAB: visible only to accounts listed in ADMIN_USERS (separate
from ALLOWED_USERS -- everyone in ALLOWED_USERS can use the app, only
ADMIN_USERS get admin capabilities). Lets an admin edit/create policies
without writing SQL, and re-tag documents that were already ingested BEFORE a
policy's viewer_groups changed (editing the policy table alone only affects
FUTURE uploads -- see README "Re-tagging already-ingested documents"), then
triggers a vector index resync so the change is queryable immediately.
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
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_documents"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_documents"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.gold_chunks"

VS_ENDPOINT = os.environ.get("VS_ENDPOINT", "news_agent_vs_endpoint")
VS_INDEX_NAME = os.environ.get("VS_INDEX_NAME", f"{CATALOG}.{SCHEMA}.gold_chunks_index")

# Comma-separated list of role-group names offered as checkboxes when
# editing/creating a policy's uploader_groups / viewer_groups in the Manage tab.
ALL_ROLE_GROUPS = [
    g.strip()
    for g in os.environ.get(
        "ALL_ROLE_GROUPS",
        "data-engineers,data-scientists,business-analysts,data-analysts,project-managers",
    ).split(",")
    if g.strip()
]

# Comma-separated list of individual emails allowed to see/use the "Manage"
# tab (edit policies, re-tag already-ingested documents, trigger reindex).
# Deliberately separate from ALLOWED_USERS -- everyone in ALLOWED_USERS can
# use the app, but only these accounts get admin capabilities.
ADMIN_USERS = {
    u.strip().lower()
    for u in os.environ.get("ADMIN_USERS", "").split(",")
    if u.strip()
}

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


def get_user_groups(email: str) -> tuple[set[str] | None, dict]:
    """Returns (group_names_or_None, debug_info). group_names is None only if
    the lookup itself errored (e.g. missing SCIM permission -- see README
    step 23). If the lookup succeeds but finds no matching user, that's
    reported in debug_info rather than silently treated the same as
    "verified to have zero groups" -- a userName/email mismatch (common with
    some SSO setups) looks identical to "not in any group" unless you can see
    the raw match count, which is why this now surfaces it."""
    if not email:
        return set(), {"matches_found": 0, "note": "no email"}

    w = get_workspace_client()
    debug_info = {"queries_tried": []}

    for filt in (
        f"userName eq '{email}'",
        f"userName eq '{email.lower()}'",
    ):
        try:
            matches = list(w.users.list(filter=filt, attributes="id,userName,groups"))
            debug_info["queries_tried"].append({"filter": filt, "matches": len(matches)})
            if matches:
                found_username = matches[0].user_name
                groups = {g.display for g in (matches[0].groups or [])}
                debug_info["matched_username"] = found_username
                debug_info["groups_found"] = sorted(groups)
                return groups, debug_info
        except Exception as e:
            debug_info["error"] = str(e)
            return None, debug_info

    # No exact match on either casing -- last resort: narrow by the local
    # part of the email (SCIM "contains") and compare client-side, in case
    # userName differs from email in some other way (whitespace, etc.).
    local_part = email.split("@")[0]
    try:
        candidates = list(w.users.list(filter=f"userName co '{local_part}'", attributes="id,userName,groups"))
        debug_info["queries_tried"].append({"filter": f"userName co '{local_part}'", "matches": len(candidates)})
        debug_info["candidate_usernames"] = [c.user_name for c in candidates]
        for c in candidates:
            if (c.user_name or "").strip().lower() == email.strip().lower():
                groups = {g.display for g in (c.groups or [])}
                debug_info["matched_username"] = c.user_name
                debug_info["groups_found"] = sorted(groups)
                return groups, debug_info
    except Exception as e:
        debug_info["error"] = str(e)
        return None, debug_info

    debug_info["note"] = "No Databricks user found whose userName matches this email in any form tried."
    return set(), debug_info


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


# ---- Admin ("Manage" tab) helpers ----

def get_all_policies():
    """All policies (active or not) -- for the admin editor, unlike
    get_eligible_policies() which is what uploaders see."""
    return run_query(f"SELECT * FROM {POLICIES_TABLE} ORDER BY policy_id")


def update_policy(policy_id: str, fields: dict):
    """fields: display_name, description (str); allowed_categories,
    allowed_extensions, uploader_groups, viewer_groups (list[str]);
    require_content_check, active (bool)."""
    set_clauses = []
    for key in ("display_name", "description"):
        if key in fields:
            set_clauses.append(f"{key} = {sql_string_literal(fields[key])}")
    for key in ("allowed_categories", "allowed_extensions", "uploader_groups", "viewer_groups"):
        if key in fields:
            set_clauses.append(f"{key} = {sql_array_literal(fields[key])}")
    for key in ("require_content_check", "active"):
        if key in fields:
            set_clauses.append(f"{key} = {'true' if fields[key] else 'false'}")
    if not set_clauses:
        return
    run_statement(f"UPDATE {POLICIES_TABLE} SET {', '.join(set_clauses)} WHERE policy_id = {sql_string_literal(policy_id)}")


def insert_policy(policy_id: str, fields: dict, created_by: str):
    run_statement(f"""
        INSERT INTO {POLICIES_TABLE}
        (policy_id, display_name, description, allowed_categories, allowed_extensions, uploader_groups, viewer_groups, require_content_check, active, created_by, created_at)
        VALUES (
            {sql_string_literal(policy_id)}, {sql_string_literal(fields.get('display_name', ''))}, {sql_string_literal(fields.get('description', ''))},
            {sql_array_literal(fields.get('allowed_categories', []))}, {sql_array_literal(fields.get('allowed_extensions', []))},
            {sql_array_literal(fields.get('uploader_groups', []))}, {sql_array_literal(fields.get('viewer_groups', []))},
            {'true' if fields.get('require_content_check') else 'false'}, {'true' if fields.get('active', True) else 'false'},
            {sql_string_literal(created_by)}, current_timestamp()
        )
    """)


def retag_documents(policy_id: str, new_viewer_groups: list[str]):
    """Updates viewer_groups on every already-ingested row (bronze, silver,
    gold) tagged with this policy_id -- needed because editing the policy
    table alone only changes what happens to FUTURE uploads. Returns the
    number of gold_chunks rows affected (used to warn if nothing matched)."""
    new_array = sql_array_literal(new_viewer_groups)
    pid = sql_string_literal(policy_id)

    run_statement(f"UPDATE {BRONZE_TABLE} SET viewer_groups = {new_array} WHERE policy_id = {pid}")
    run_statement(f"UPDATE {SILVER_TABLE} SET viewer_groups = {new_array} WHERE policy_id = {pid}")
    run_statement(f"UPDATE {GOLD_TABLE} SET viewer_groups = {new_array} WHERE policy_id = {pid}")

    affected = run_query(f"SELECT count(*) as n FROM {GOLD_TABLE} WHERE policy_id = {pid}")
    return int(affected[0]["n"]) if affected else 0


def trigger_vector_index_resync():
    """Vector Search uses TRIGGERED sync -- edits to gold_chunks (including
    the retag above) are invisible to queries until a sync is explicitly
    triggered. Requires the app's service principal to have query/manage
    permission on the vector search endpoint -- same class of grant as step
    17 in the README, but for the vector search endpoint instead of the
    serving endpoint."""
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient(disable_notice=True)
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX_NAME)
    index.sync()


def check_access(email: str) -> tuple[bool | None, set[str], dict]:
    """Returns (True/False/None, matched_groups, debug_info). None means the
    check couldn't be completed (see get_user_groups) and the user also isn't
    individually allow-listed, so we genuinely don't know."""
    if email.lower() in ALLOWED_USERS:
        return True, set(), {"note": "individually allow-listed, group lookup skipped"}

    user_groups, debug_info = get_user_groups(email)
    if user_groups is None:
        return None, set(), debug_info

    matched = user_groups & ALLOWED_GROUPS
    return (bool(matched), matched, debug_info)


# ---- Access control ----
user_email = get_current_user_email()

if not user_email:
    st.error("🚫 Could not identify who you are. Access denied.")
    st.stop()

access, matched_groups, access_debug = check_access(user_email)

if access is None:
    st.error(
        "⚠️ Couldn't verify access — you're not on the individual allow-list, "
        "and the app doesn't have permission to check group membership yet. "
        "See the README's access-control setup step (granting the app's "
        "service principal a role with SCIM read access), or ask an admin to "
        "add your email to ALLOWED_USERS in app.yaml as a quick workaround."
    )
    with st.expander("Diagnostic details (for whoever's troubleshooting this)"):
        st.json(access_debug)
    st.stop()

if access is False:
    st.error(
        f"🚫 Access denied. You must either be individually allow-listed or a "
        f"member of one of these groups: {', '.join(sorted(ALLOWED_GROUPS))}."
        f"\n\nSigned in as: {user_email}"
    )
    with st.expander("Diagnostic details (for whoever's troubleshooting this)"):
        st.caption(
            "If you're confident you ARE a member of one of the groups above, this usually "
            "means the SCIM lookup couldn't match your Databricks `userName` to your login "
            "email — check `matched_username` / `candidate_usernames` below against your "
            "actual Databricks username (Admin Settings > Users)."
        )
        st.json(access_debug)
    st.stop()
# ---- End access control ----

st.title("📰 NewsScan Agent")
role_note = f" ({', '.join(sorted(matched_groups))})" if matched_groups else ""

# Full group membership (may be a superset of matched_groups, which is only
# the intersection with ALLOWED_GROUPS) -- this is what's sent to the agent
# for role-based document filtering, and what's checked against each upload
# policy's uploader_groups. Shown to the user directly below so it's obvious
# WHY they can/can't see a given document, instead of them having to guess.
# NOTE: this is looked up fresh even for individually-allow-listed users --
# ALLOWED_USERS only bypasses the APP-ACCESS check, never document-visibility
# filtering, which always uses real group membership.
_full_groups, _full_groups_debug = get_user_groups(user_email)
user_full_groups = _full_groups if _full_groups is not None else matched_groups
is_individually_allowed = user_email.lower() in ALLOWED_USERS
is_admin = user_email.lower() in ADMIN_USERS

st.caption(f"Signed in as {user_email}{role_note}")
if _full_groups is None:
    st.caption(
        f"⚠️ Couldn't verify your full group membership (SCIM read permission not "
        f"granted to the app yet — see README step 23) — falling back to "
        f"`{', '.join(sorted(user_full_groups)) or '(none)'}` for document access filtering. "
        f"This may be narrower than your real groups."
    )
elif not _full_groups:
    st.caption(
        "⚠️ No groups detected for document access — you'll only see documents tagged "
        "open to everyone. If you expect to be in a specific group, expand diagnostics below."
    )
    with st.expander("Diagnostic details"):
        st.json(_full_groups_debug)
else:
    st.caption(f"Groups used for document access: `{', '.join(sorted(user_full_groups))}`")

if "history" not in st.session_state:
    st.session_state.history = []

w = get_workspace_client()

if is_admin:
    ask_tab, upload_tab, manage_tab = st.tabs(["💬 Ask", "📤 Upload", "⚙️ Manage"])
else:
    ask_tab, upload_tab = st.tabs(["💬 Ask", "📤 Upload"])
    manage_tab = None


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


if manage_tab is not None:
    with manage_tab:
        st.caption(
            "Admin-only. Edit existing policies, add new ones, and re-tag documents that were "
            "already ingested before a policy changed (editing a policy alone only affects "
            "future uploads)."
        )

        if not SQL_WAREHOUSE_ID:
            st.warning("⚠️ `SQL_WAREHOUSE_ID` isn't set in app.yaml — see README \"Document-level upload policies\".")
        else:
            edit_tab, new_tab, retag_tab = st.tabs(["Edit policies", "New policy", "Re-tag existing documents"])

            try:
                all_policies = get_all_policies()
            except Exception as e:
                all_policies = []
                st.error(f"Couldn't load policies: {e}")

            # ---- Edit existing policy ----
            with edit_tab:
                if not all_policies:
                    st.info("No policies yet — create one in the 'New policy' tab.")
                else:
                    edit_labels = {p["policy_id"]: p["display_name"] for p in all_policies}
                    edit_id = st.selectbox("Policy", options=list(edit_labels.keys()), format_func=lambda pid: edit_labels[pid], key="edit_policy_select")
                    p = next(x for x in all_policies if x["policy_id"] == edit_id)

                    with st.form(f"edit_form_{edit_id}"):
                        display_name = st.text_input("Display name", value=p.get("display_name") or "")
                        description = st.text_area("Description", value=p.get("description") or "")
                        allowed_categories = st.text_input("Allowed categories (comma-separated)", value=", ".join(parse_sql_array(p.get("allowed_categories"))))
                        allowed_extensions = st.text_input("Allowed file extensions (comma-separated, no dots)", value=", ".join(parse_sql_array(p.get("allowed_extensions"))))
                        uploader_groups = st.multiselect("Who can upload (uploader_groups)", options=ALL_ROLE_GROUPS, default=[g for g in parse_sql_array(p.get("uploader_groups")) if g in ALL_ROLE_GROUPS])
                        viewer_groups = st.multiselect("Who can view/query (viewer_groups)", options=ALL_ROLE_GROUPS, default=[g for g in parse_sql_array(p.get("viewer_groups")) if g in ALL_ROLE_GROUPS])
                        require_content_check = st.checkbox("Require content check (LLM verifies uploaded content matches allowed categories)", value=bool(p.get("require_content_check")))
                        active = st.checkbox("Active (shown to uploaders)", value=bool(p.get("active")))

                        if st.form_submit_button("Save policy"):
                            update_policy(edit_id, {
                                "display_name": display_name,
                                "description": description,
                                "allowed_categories": [c.strip() for c in allowed_categories.split(",") if c.strip()],
                                "allowed_extensions": [e.strip().lstrip(".") for e in allowed_extensions.split(",") if e.strip()],
                                "uploader_groups": uploader_groups,
                                "viewer_groups": viewer_groups,
                                "require_content_check": require_content_check,
                                "active": active,
                            })
                            st.success(
                                f"Saved '{edit_id}'. This only affects FUTURE uploads under this policy — "
                                f"use the 'Re-tag existing documents' tab to also update already-ingested files."
                            )
                            st.rerun()

            # ---- Create new policy ----
            with new_tab:
                with st.form("new_policy_form"):
                    new_id = st.text_input("Policy ID (unique, e.g. 'contracts_legal_only')")
                    new_display_name = st.text_input("Display name")
                    new_description = st.text_area("Description")
                    new_allowed_categories = st.text_input("Allowed categories (comma-separated)", value="general")
                    new_allowed_extensions = st.text_input("Allowed file extensions (comma-separated, no dots)", value="pdf")
                    new_uploader_groups = st.multiselect("Who can upload (uploader_groups)", options=ALL_ROLE_GROUPS)
                    new_viewer_groups = st.multiselect("Who can view/query (viewer_groups)", options=ALL_ROLE_GROUPS)
                    new_require_content_check = st.checkbox("Require content check", value=False)

                    if st.form_submit_button("Create policy"):
                        existing_ids = {p["policy_id"] for p in all_policies}
                        if not new_id.strip():
                            st.error("Policy ID is required.")
                        elif new_id.strip() in existing_ids:
                            st.error(f"Policy ID '{new_id}' already exists — edit it instead, or pick a different ID.")
                        else:
                            insert_policy(new_id.strip(), {
                                "display_name": new_display_name,
                                "description": new_description,
                                "allowed_categories": [c.strip() for c in new_allowed_categories.split(",") if c.strip()],
                                "allowed_extensions": [e.strip().lstrip(".") for e in new_allowed_extensions.split(",") if e.strip()],
                                "uploader_groups": new_uploader_groups,
                                "viewer_groups": new_viewer_groups,
                                "require_content_check": new_require_content_check,
                                "active": True,
                            }, user_email)
                            st.success(f"Created policy '{new_id}'. It'll show up for eligible uploaders immediately.")
                            st.rerun()

            # ---- Re-tag already-ingested documents + trigger reindex ----
            with retag_tab:
                st.markdown(
                    "Updates `viewer_groups` on every already-ingested document (bronze, silver, "
                    "and gold tables) tagged with a given policy, then triggers a vector index "
                    "resync — so the change is queryable within a few minutes, without waiting for "
                    "the next scheduled ingestion run."
                )
                if not all_policies:
                    st.info("No policies yet.")
                else:
                    retag_labels = {p["policy_id"]: p["display_name"] for p in all_policies}
                    retag_id = st.selectbox("Policy", options=list(retag_labels.keys()), format_func=lambda pid: retag_labels[pid], key="retag_policy_select")
                    rp = next(x for x in all_policies if x["policy_id"] == retag_id)
                    current_viewer_groups = parse_sql_array(rp.get("viewer_groups"))
                    st.caption(f"Policy's current `viewer_groups`: `{', '.join(current_viewer_groups) or '(none)'}`")

                    retag_groups = st.multiselect(
                        "New viewer_groups to apply to already-ingested documents under this policy",
                        options=ALL_ROLE_GROUPS,
                        default=current_viewer_groups,
                        key=f"retag_groups_{retag_id}",
                    )

                    if st.button("Re-tag documents + trigger reindex", key=f"retag_btn_{retag_id}"):
                        try:
                            with st.spinner("Updating bronze/silver/gold tables..."):
                                affected = retag_documents(retag_id, retag_groups)
                            if affected == 0:
                                st.warning(f"No ingested documents currently have policy_id = '{retag_id}' — nothing to re-tag yet.")
                            else:
                                with st.spinner("Triggering vector index resync..."):
                                    trigger_vector_index_resync()
                                st.success(
                                    f"✅ Re-tagged {affected} document(s) and triggered a reindex. "
                                    f"Changes are typically queryable within a few minutes — check "
                                    f"Serving > vector search index status if it takes longer."
                                )
                        except Exception as e:
                            st.error(f"Re-tag/reindex failed: {e}")