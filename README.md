# NewsScan Agent — Automated Document/Newspaper Q&A Agent on Databricks

An end-to-end pipeline + AI agent that:
1. Ingests documents (PDF / HTML / TXT — newspaper articles, reports, etc.) dropped into a folder
2. Cleans and parses them into plain text
3. Splits text into searchable chunks and indexes them for semantic search
4. Lets you ask questions in plain English and get answers grounded in your documents (with sources)
5. Runs automatically on a schedule so new documents get picked up without manual work

## Architecture (medallion + RAG agent)

```
Raw files (PDF/HTML/TXT)
   -> Volume (Unity Catalog storage)
      -> [01] Bronze table (raw bytes + metadata, via Auto Loader)
         -> [02] Silver table (cleaned plain text per document)
            -> [03] Gold table (chunked text, ready for embedding)
               -> [04] Vector Search Index (Databricks-managed embeddings)
                  -> [05] Agent (retrieve chunks -> ask LLM -> human-language answer)
                     -> [06] Logged to MLflow, registered in Unity Catalog, deployed as a Model Serving endpoint
                        -> [07] Ask it questions from a notebook or the REST API
                           -> [App] Databricks App — a chat webpage anyone in your workspace can open and use
```

### Document-level upload policies (who can upload what, who can see it)

On top of the app-access groups above, there's a second, finer-grained layer:
you (or an admin) define **policies** in the `upload_policies` table that say,
per role, what content is allowed in and who can query it back out once
ingested. The demo ships with an example matching a common ask: *"Project
Managers can upload here, but the content must actually be a resume, and only
Data Engineers + Data Analysts can ever see it in answers."*

```
Project Manager opens the app's "Upload" tab
   -> picks a policy (only policies their role can upload under are shown)
      -> uploads a file
         -> extension checked against the policy's allowed_extensions
            -> if require_content_check: the LLM verifies the file's actual
               content matches the policy's allowed_categories (e.g. "resume")
               -> mismatch -> upload rejected with a clear error, nothing written
               -> match -> file written to the Volume, tagged with the policy's
                  viewer_groups, logged in upload_manifest
                     -> next ingestion run carries policy_id/category/viewer_groups
                        through bronze -> silver -> gold -> the vector index
                           -> the agent filters retrieved chunks by the ASKING
                              user's groups vs. each chunk's viewer_groups, so a
                              Business Analyst's question never surfaces that
                              resume, even if it's the best semantic match
```

See "Document-level upload policies" further down for the one-time setup this
needs (running `00b_setup_access_policies.py`, granting the app a SQL
Warehouse, and granting it query access to the chat model endpoint used for
content verification).

## Folder structure

```
newsscan-agent/
├── README.md                          <- you are here
├── requirements.txt                   <- Python libraries to install on the cluster
├── config.yaml                        <- all the names/settings used across notebooks
├── notebooks/
│   ├── 00_setup_catalog_and_volume.py
│   ├── 00b_setup_access_policies.py     <- upload_policies + upload_manifest tables (document-level access control)
│   ├── 01_ingest_bronze.py
│   ├── 02_parse_clean_silver.py
│   ├── 03_chunk_gold.py
│   ├── 04_create_vector_index.py
│   ├── 05_build_and_test_agent.py
│   ├── 06_log_register_deploy.py
│   ├── 06b_deploy_serving_endpoint.py
│   └── 07_ask_the_agent.py
├── jobs/
│   └── ingestion_workflow.json        <- Databricks Job definition to automate steps 01-04
└── app/                               <- the Databricks App (chat webpage)
    ├── app.py                         <- Streamlit chat UI
    ├── app.yaml                       <- tells Databricks Apps how to start it
    └── requirements.txt               <- Python libraries the app needs
```

## Prerequisites (things to set up in Databricks before you start)

- A Databricks workspace with **Unity Catalog** enabled (ask your admin if unsure — most workspaces created after 2023 have it).
- **Serverless compute enabled** for notebooks and jobs. This is on by default in most workspaces — you'll know it's available if "Serverless" appears in the compute dropdown at the top-right of a notebook. **You do not create or manage a cluster anywhere in this project** — every notebook, job, and the app itself all run on serverless. If you don't see "Serverless" as an option, ask your workspace admin to enable it.
- **Vector Search** enabled for the workspace (admin console > Previews, or ask your admin — it's on by default in most workspaces now; the Vector Search endpoint itself is always serverless infrastructure, so no compute choice needed there either).
- Access to **Foundation Model APIs** (the pay-per-token Databricks-hosted models like `databricks-bge-large-en` for embeddings and `databricks-meta-llama-3-3-70b-instruct` for the chat model). These are enabled by default in most workspaces, and are serverless/pay-per-token by nature.
- Permission to create a catalog/schema in Unity Catalog (or ask an admin to create one for you and give you `USE CATALOG` / `CREATE TABLE` rights).

**Note on serverless:** the base serverless environment doesn't come with every library preinstalled (e.g. `pymupdf`, `langchain`, `databricks-vectorsearch`), which is exactly why notebooks 02–06 each start with a `%pip install ... ` cell — that installs what that specific notebook needs, scoped to that session, every time it runs. That's normal and by design; nothing extra for you to do.

## Datasets

You have two options:

**Option A — use your own documents.** This is the point of the project. Just have some PDFs, HTML pages, or .txt files of newspaper articles ready to upload.

**Option B — use a public sample dataset to test the pipeline first.**
- Kaggle "All the News 2.0" or "News Category Dataset" (CSV of article text — you'd convert rows to .txt files)
- Any local newspaper's public RSS/HTML archive (save a few pages as .html)
- A handful of PDF news articles or reports you already have on your laptop

Either way, all files go into a Unity Catalog **Volume** — think of it as a managed folder in cloud storage that Databricks tracks for you. Step 00 creates it; step by step instructions below tell you exactly where to drop files.

## How to run this, step by step

**Legend** — every step below is tagged so you know what kind of action it is:
- 🖱️ **MANUAL** — something you do yourself by clicking around in the Databricks UI, or typing a command in a terminal. No code to write.
- ▶️ **RUN** — open a notebook that's already written for you and click "Run All". No coding needed.

### Part 1 — the data pipeline and agent

1. 🖱️ **MANUAL — Import this project.** Workspace → your user folder → Import → upload each file, keeping the same folder structure. (Better option: push this folder to a Git repo, then Repos → Add Repo in Databricks — everything lands in the right place automatically, and it's also required for the App deployment in Part 2.)
2. ▶️ **RUN `notebooks/00_setup_catalog_and_volume.py`** — open it, and at the top-right compute dropdown select **Serverless** (there's nothing to create or configure — it just attaches). Click Run All. Creates the catalog/schema/Volume.
2b. ▶️ **RUN `notebooks/00b_setup_access_policies.py`** (Serverless) — creates the `upload_policies` and `upload_manifest` tables and seeds two demo policies (`resumes_pm_only`, `general_docs`). Skip this if you don't need document-level upload restrictions/visibility — everything else still works without it (files just fall back to open visibility, same as before this feature existed).
3. 🖱️ **MANUAL — Upload documents.** Two ways now:
   - **Recommended: the app's Upload tab** (Part 2 below) — applies the policies from step 2b (file-type + content checks, role-based visibility).
   - **Or, same as before:** Catalog (left sidebar) → `news_agent` → `docs` → Volumes → `raw_files` → "Upload to this volume" → drag in files directly. These bypass policy checks entirely and default to open visibility (see notebook 01's fallback behavior) — fine for the original "everyone sees everything" use case, not for restricted content like resumes.
4. ▶️ **RUN `01_ingest_bronze.py`** (Serverless) — copies raw files into a Delta table.
5. ▶️ **RUN `02_parse_clean_silver.py`** (Serverless) — extracts clean text from each file.
6. ▶️ **RUN `03_chunk_gold.py`** (Serverless) — splits each document into search-ready chunks.
7. ▶️ **RUN `04_create_vector_index.py`** (Serverless) — creates the search index (first run takes a few minutes to spin up; be patient).
8. ▶️ **RUN `05_build_and_test_agent.py`** (Serverless) — the last cell asks a test question; check the answer makes sense and cites real documents.
9. ▶️ **RUN `06_log_register_deploy.py`** (Serverless) — registers the agent in Unity Catalog and tags it with a `@champion` alias. This step only logs/registers — it does not deploy anything yet.
10. ▶️ **RUN `06b_deploy_serving_endpoint.py`** (Serverless) — deploys the `@champion` version as a live REST endpoint (several minutes for the first deploy). Model Serving is always serverless infrastructure, so nothing to configure there either.
11. 🖱️ **MANUAL — Check it's live.** Left sidebar → Serving → `news_qa_agent_endpoint` → **Deployments** tab → wait for status **Ready** before moving on. If it shows **Failed**, check the **Logs** tab for the actual error.
12. ▶️ **RUN `07_ask_the_agent.py`** (Serverless) to sanity-check real questions against the live endpoint.
13. 🖱️ **MANUAL — Automate ingestion.** Open `jobs/ingestion_workflow.json`, replace `<you>` with your actual workspace username/path. In Databricks: Workflows → Create Job → "Import from JSON" (or recreate the 4 tasks by hand in the UI, leaving compute set to **Serverless** for each task) → set the schedule → un-pause it. New files dropped in the Volume now get ingested and indexed automatically — no job cluster to size or manage.

### Part 2 — the Databricks App (chat webpage)

This gives you (and anyone else in your workspace you grant access to) a simple chat webpage — no notebook required — to ask the agent questions. It calls the serving endpoint from Part 1, so **do Part 1 first**. Databricks Apps always run on serverless compute — there's no compute choice to make here at all.

14. 🖱️ **MANUAL — Install the Databricks CLI** on your own laptop, if you don't already have it: `curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh` (Mac/Linux) or see [docs.databricks.com CLI install](https://docs.databricks.com/en/dev-tools/cli/install.html) for Windows.
15. 🖱️ **MANUAL — Authenticate the CLI**: run `databricks auth login --host https://<your-workspace-url>` and follow the browser prompt.
16. 🖱️ **MANUAL — Create the app** (one-time): `databricks apps create newsscan-agent`
17. 🖱️ **MANUAL — Grant the app permission to call your agent.** This is the step people miss. Databricks Apps run as their own identity (a service principal), separate from you — it can't query your serving endpoint until you explicitly allow it:
    - Left sidebar → **Serving** → `news_qa_agent_endpoint` → **Permissions** tab
    - Add the app's service principal (find its name/ID under **Apps** → `newsscan-agent` → **Authorization**, it looks like `newsscan-agent` or a generated app ID) → grant **Can Query**
18. 🖱️ **MANUAL — Upload the app code to your workspace** (from your laptop, in this project's folder): `databricks sync ./app /Workspace/Users/<you>/newsscan-agent/app`
19. 🖱️ **MANUAL — Deploy it**: `databricks apps deploy newsscan-agent --source-code-path /Workspace/Users/<you>/newsscan-agent/app`
20. 🖱️ **MANUAL — Open it.** Left sidebar → Apps → `newsscan-agent` → click the app URL. You (and anyone you grant "Can Use" on the app) now have a chat page for the agent.

To push a code change later: edit `app/app.py`, then repeat steps 18–19 (`sync` then `deploy`) — no need to recreate the app.

### Access control (who's allowed to use the app)

The app checks, on every visit, whether the logged-in user is allowed in — everyone else sees "Access denied." A user is let in if **either** of these is true:
- their email is individually listed in `ALLOWED_USERS` (in `app/app.yaml`), **or**
- they belong to **any one** of the groups listed in `ALLOWED_GROUPS`

**Demo setup: 5 role-based groups.** For now this is standing in for whatever the business eventually settles on — `ALLOWED_GROUPS` defaults to:
```
data-engineers,data-scientists,business-analysts,data-analysts,project-managers
```
(`data-analysts` was added alongside the original four specifically so "Data Analyst" is its own role, distinct from `business-analysts` — merge or rename these in `app.yaml` if your org treats them as the same thing.)

Membership in any single one of these five is enough to get in — someone doesn't need to be in all five. This is deliberately just a comma-separated list in `app.yaml`, so changing which roles are allowed later — adding a sixth, removing one, renaming one — is a one-line edit and a redeploy, no code change.

**The individual list needs no setup at all** — just edit `ALLOWED_USERS` in `app/app.yaml` and redeploy. Good for letting yourself in immediately, or one-off access, without touching group membership.

**The group check needs one-time manual setup, once per group:**

21. 🖱️ **MANUAL — Create the five groups.** Databricks UI (or account console) → **Admin Settings** → **Identity and access** → **Groups** → **Add group**. Create each of: `data-engineers`, `data-scientists`, `business-analysts`, `data-analysts`, `project-managers` (exact names — if you use different names, update `ALLOWED_GROUPS` in `app/app.yaml`, and the `uploader_groups`/`viewer_groups` values in `upload_policies`, to match).
22. 🖱️ **MANUAL — Add members to each.** Open each group → **Members** tab → add the people who belong to that role.
23. 🖱️ **MANUAL — Let the app read group membership.** The app's service principal needs permission to look up users' group membership via the SCIM API — by default it doesn't have this. Ask your workspace/account admin to grant the app's service principal a role that includes "read" access to Users/Groups (in many workspaces this means adding it as a member of a group with delegated user-management rights, or granting it limited SCIM read scope — the exact option depends on your workspace's admin settings, so this is one to check with your admin on if you don't see it yourself). This one grant covers all five groups — it's a single permission, not per-group.

If step 23 isn't set up yet, anyone not on the individual list will see a distinct warning ("Couldn't verify access...") rather than being silently let in or wrongly blocked — and that same warning tells them to ask about being added to `ALLOWED_USERS` as an immediate workaround while the group permission is sorted out.

**To change who's allowed** later:
- Add/remove members from a group in the UI (no redeploy needed), or
- Edit `ALLOWED_GROUPS` or `ALLOWED_USERS` in `app/app.yaml` and redeploy (needed when adding/removing/renaming a *group itself*, not for changing who's in an existing group)

### Document-level upload policies

This is the layer on top of app access: not just "can this person open the
app", but "what is this person allowed to upload, and who's allowed to see it
afterward". It's driven entirely by rows in the `upload_policies` table — no
code change needed to add a new policy or tweak an existing one.

**One-time setup:**

24. ▶️ **RUN `notebooks/00b_setup_access_policies.py`** if you haven't already (step 2b in Part 1) — creates `upload_policies` (the rules) and `upload_manifest` (the audit log of every upload attempt, accepted or rejected).
25. 🖱️ **MANUAL — Create/pick a SQL Warehouse.** Left sidebar → SQL Warehouses → use an existing one or create a small serverless one. Copy its **Warehouse ID** (Connection details tab).
26. 🖱️ **MANUAL — Set `SQL_WAREHOUSE_ID` in `app/app.yaml`** to that ID, then redeploy (`databricks sync` + `databricks apps deploy`, same as steps 18-19).
27. 🖱️ **MANUAL — Grant the app "Can Use" on that warehouse.** SQL Warehouses → your warehouse → Permissions → add the app's service principal → **Can Use**. Without this, the Upload tab shows "Couldn't load upload policies" errors.
28. 🖱️ **MANUAL — Grant the app query access to the chat model endpoint** (`databricks-meta-llama-3-3-70b-instruct` by default) — same as step 17 but for this endpoint instead of the serving endpoint, since the Upload tab calls it directly to verify uploaded content matches a policy's allowed category (e.g. confirming a file really is a resume). Only needed for policies with `require_content_check = true`.

**How it behaves, end to end:**
- The Upload tab only shows policies whose `uploader_groups` overlaps the logged-in user's groups (or all active policies, for individually-allow-listed users). A Project Manager sees `resumes_pm_only`; a Data Scientist doesn't.
- On upload, the file's extension is checked against `allowed_extensions` first (cheap, immediate). If it fails: **"Can't upload this file — file type '.xyz' not allowed by policy '...'"** and nothing is written anywhere.
- If the policy also has `require_content_check = true`, the file's text is extracted and sent to the chat model with a strict classification prompt ("does this match {allowed_categories}? yes/no"). A non-match rejects the upload with a clear error (e.g. uploading a spreadsheet under the resumes-only policy) — again, nothing is written to the Volume.
- Every attempt — accepted or rejected — gets a row in `upload_manifest` for audit purposes.
- Accepted files land in the Volume under `/Volumes/news_agent/docs/raw_files/<policy_id>/<filename>`, tagged with that policy's `viewer_groups`.
- The next ingestion run (steps 01-04, or the scheduled job) carries `policy_id` / `category` / `viewer_groups` through bronze → silver → gold → the vector index.
- The agent (notebook 05/06) now takes the asking user's groups as a second input and drops any retrieved chunk whose `viewer_groups` doesn't overlap them — so a resume tagged `viewer_groups=[data-engineers, data-analysts]` never appears in an answer to a Project Manager or Business Analyst, even if it's the closest semantic match to their question.

**Adding a new policy later** (e.g. "only Data Scientists can upload training data, visible only to Data Scientists + Data Engineers"): use the Manage tab below (recommended), or just `INSERT` a row into `upload_policies` directly — no redeploy, no notebook re-run. It shows up in the Upload tab on next page load.

**Note on the security model:** this is application-layer enforcement (the app and the agent check group membership before showing content), not Unity Catalog row-level security on the underlying files — anyone with direct Catalog Explorer / `dbutils.fs` access to the Volume or the Delta tables can still see everything, same as any other UC object with standard grants. If you need the raw files themselves locked down per-role (not just "don't surface this in chat answers"), put restricted-content policies in their own schema/Volume with UC-level grants restricting who can even browse it, in addition to this feature.

### The admin "Manage" tab (edit policies without SQL)

A third tab, **⚙️ Manage**, is visible only to accounts listed in `ADMIN_USERS` in `app.yaml` (comma-separated emails — separate from `ALLOWED_USERS`; being in `ALLOWED_USERS` only grants normal app access, not admin capabilities). It has three sections:

- **Edit policies** — change any policy's `display_name`, `description`, `allowed_categories`, `allowed_extensions`, `uploader_groups`, `viewer_groups`, `require_content_check`, `active` from a form. Writes straight to `upload_policies`.
- **New policy** — create a policy from a form instead of a notebook `INSERT`.
- **Re-tag existing documents** — this is the one to reach for after editing a policy's `viewer_groups`. **Editing the policy table alone only changes what happens to *future* uploads** — it does NOT change documents already sitting in `bronze_documents` / `silver_documents` / `gold_chunks`, since those carry their own `viewer_groups` copied at ingestion time. This section runs the equivalent of:
  ```sql
  UPDATE bronze_documents SET viewer_groups = array(...) WHERE policy_id = '...';
  UPDATE silver_documents SET viewer_groups = array(...) WHERE policy_id = '...';
  UPDATE gold_chunks       SET viewer_groups = array(...) WHERE policy_id = '...';
  ```
  and then triggers a vector index resync (equivalent to re-running notebook `04`), so the change is queryable within a few minutes instead of waiting for the next scheduled ingestion run.

**One extra permission needed for the "re-tag + reindex" button specifically:** the app's service principal needs permission to trigger a sync on the vector search index — grant it the same way as step 17 (Serving endpoint permission), but on the **vector search endpoint** (`news_agent_vs_endpoint`) instead: left sidebar → **Compute** → **Vector Search** → your endpoint → grant the app's service principal permission to manage/sync the index. Without this, the re-tag itself succeeds but the reindex step fails — the tab will tell you this explicitly rather than failing silently.

### After the first run
Steps 4–7 (and the job in step 13) are all designed to be safe to re-run — they only process files/documents/chunks that are new since last time, so day-to-day you just drop new files in the volume and either wait for the schedule or manually trigger the job. Because everything is serverless, there's also no cluster sitting around costing you money between runs — compute only exists for the seconds/minutes each step actually takes.

### Common gotchas for beginners
- **"Catalog does not exist" errors**: you (or an admin) need `CREATE CATALOG` privilege, or ask an admin to run notebook 00 for you once.
- **Vector index stuck "Provisioning"**: this is normal for the first few minutes after creation — grab a coffee.
- **`ChatDatabricks` / embedding endpoint errors**: means Foundation Model APIs aren't enabled for your workspace yet — ask your workspace admin to enable "Foundation Model APIs — pay-per-token" under the Serving tab.
- **PDF parsing returns empty text**: some PDFs are scanned images rather than real text — those need OCR (not covered here); text-based PDFs and HTML/TXT work out of the box.
- **Serving endpoint fails with "MLflow raised an error loading the model" / "no replicas in a running state"**: almost always means the model was logged without declaring the `resources` it depends on (the vector index, the LLM endpoint) — see notebook 06's `resources=[...]` block. Re-run 06 then 06b.
- **App shows an error calling the agent / 403**: almost always step 17 — the app's service principal doesn't have "Can Query" on the serving endpoint yet.
- **`databricks apps create` fails with a permissions error**: Databricks Apps needs to be enabled for your workspace and you need at least "Can Create" apps permission — ask your workspace admin.
- **Don't see "Serverless" in the compute dropdown**: your workspace admin needs to turn on serverless compute for notebooks/jobs (Admin Console → Compute → Serverless). Everything in this project assumes it's on.
- **A library "not found" error on a notebook you re-ran**: serverless sessions are ephemeral — each time a notebook detaches/reattaches (e.g. after being idle, or in a fresh Job run) it starts clean, so the `%pip install` cell at the top of that notebook needs to run again. That's expected; just re-run the whole notebook rather than a single cell lower down.
- **Upload tab shows "Couldn't load upload policies"**: almost always `SQL_WAREHOUSE_ID` isn't set in `app.yaml`, or it's set but the app's service principal doesn't have "Can Use" on that warehouse yet — see "Document-level upload policies" step 27.
- **Upload rejects a file that looks correct**: if it's a content-check rejection, the model's classification is shown in the error/manifest row (`rejection_reason`) — check `upload_manifest` for the exact reply; the extraction step can also return empty text for scanned/image-only PDFs (same limitation noted above for the main pipeline), which fails closed (rejected) rather than silently letting it through.
- **A file uploaded through the app never shows up in answers**: uploading only writes to the Volume + manifest — it still needs the next ingestion run (steps 01-04, or the scheduled job) to actually get parsed, chunked, and synced into the vector index.
- **A `.docx` upload (e.g. a resume) never shows up in answers, no matter what `viewer_groups` says**: `02_parse_clean_silver.py` extracts text per file extension — if the extension it saw isn't one it knows how to parse, `extract_text()` returns `''`, and that document is silently dropped before silver (`.filter(col("text") != "")`), so it never reaches gold or the vector index at all. This project now handles `pdf`/`html`/`htm`/`txt`/`docx`; if you add a policy that allows a new extension (e.g. `.pptx`), add a matching extraction branch in `02_parse_clean_silver.py` too, or uploads under that extension will pass validation but silently vanish from the pipeline.