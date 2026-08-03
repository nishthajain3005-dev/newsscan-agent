# Databricks notebook source
# MAGIC %md
# MAGIC # 00b — Set up document-level upload policies
# MAGIC Run this once, after `00_setup_catalog_and_volume.py`. Creates two Delta
# MAGIC tables that power role-based upload restrictions and role-based document
# MAGIC visibility:
# MAGIC
# MAGIC - **`upload_policies`** — the rules. Each row says: which file types /
# MAGIC   content category are allowed under this policy, which groups are allowed
# MAGIC   to upload under it, and which groups are allowed to see (query) the
# MAGIC   resulting documents once ingested.
# MAGIC - **`upload_manifest`** — the audit log. One row per upload attempt made
# MAGIC   through the app's Upload tab (accepted AND rejected), so there's a record
# MAGIC   of who uploaded what, under which policy, and why anything was rejected.
# MAGIC
# MAGIC Safe to re-run — uses `IF NOT EXISTS` / `MERGE` so it won't duplicate seed
# MAGIC rows or wipe out policies you've edited since.

# COMMAND ----------

catalog = "news_agent"
schema = "docs"
policies_table = f"{catalog}.{schema}.upload_policies"
manifest_table = f"{catalog}.{schema}.upload_manifest"

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {policies_table} (
  policy_id             STRING NOT NULL,
  display_name          STRING,
  description           STRING,
  allowed_categories    ARRAY<STRING>,   -- e.g. ["resume"] -- content-check target
  allowed_extensions    ARRAY<STRING>,   -- e.g. ["pdf", "docx"]
  uploader_groups       ARRAY<STRING>,   -- who is allowed to upload under this policy
  viewer_groups         ARRAY<STRING>,   -- who is allowed to see/query the resulting docs
  require_content_check BOOLEAN,         -- if true, the LLM checks the doc actually matches allowed_categories
  active                BOOLEAN,
  created_by            STRING,
  created_at            TIMESTAMP
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {manifest_table} (
  manifest_id       STRING NOT NULL,
  file_path         STRING,   -- NULL for rejected uploads (nothing was ever written to the volume)
  original_filename STRING,
  policy_id         STRING,
  category          STRING,
  viewer_groups     ARRAY<STRING>,
  uploaded_by       STRING,
  uploaded_at       TIMESTAMP,
  status            STRING,   -- "accepted" or "rejected"
  rejection_reason  STRING
) USING DELTA
""")

print(f"Ready: {policies_table}")
print(f"Ready: {manifest_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Seed policies
# MAGIC Two demo policies, matching the walkthrough in the README:
# MAGIC 1. **`resumes_pm_only`** — Project Managers can upload, content must pass an
# MAGIC    LLM check that it's actually a resume/CV, and only Data Engineers +
# MAGIC    Data Analysts can ever see/query that content afterward.
# MAGIC 2. **`general_docs`** — the original behavior: any of the app's allowed
# MAGIC    roles can upload PDFs/HTML/TXT, no content check, visible to everyone
# MAGIC    with app access.
# MAGIC
# MAGIC Edit these rows any time in the Catalog Explorer's table editor, or with
# MAGIC plain `UPDATE`/`INSERT` SQL — no redeploy needed, the app reads this table
# MAGIC live on every visit to the Upload tab.

# COMMAND ----------

from pyspark.sql import Row
import datetime

seed_rows = [
    Row(
        policy_id="resumes_pm_only",
        display_name="Resumes (Project Manager upload -> Data Engineer / Data Analyst view only)",
        description="Only resumes/CVs. Uploaded by Project Managers. Visible only to Data Engineers and Data Analysts.",
        allowed_categories=["resume"],
        allowed_extensions=["pdf", "docx"],
        uploader_groups=["project-managers"],
        viewer_groups=["data-engineers", "data-analysts"],
        require_content_check=True,
        active=True,
        created_by="system",
        created_at=datetime.datetime.utcnow(),
    ),
    Row(
        policy_id="general_docs",
        display_name="General news / reports (all roles)",
        description="Newspaper articles, reports -- the original NewsScan use case. No content-type restriction.",
        allowed_categories=["general"],
        allowed_extensions=["pdf", "html", "htm", "txt"],
        uploader_groups=["data-engineers", "data-scientists", "business-analysts", "data-analysts", "project-managers"],
        viewer_groups=["data-engineers", "data-scientists", "business-analysts", "data-analysts", "project-managers"],
        require_content_check=False,
        active=True,
        created_by="system",
        created_at=datetime.datetime.utcnow(),
    ),
]

seed_df = spark.createDataFrame(seed_rows)

existing_ids = set()
if spark.catalog.tableExists(policies_table):
    existing_ids = {r["policy_id"] for r in spark.table(policies_table).select("policy_id").collect()}

new_seed_df = seed_df.filter(~seed_df.policy_id.isin(existing_ids)) if existing_ids else seed_df

if new_seed_df.limit(1).count() > 0:
    new_seed_df.write.format("delta").mode("append").saveAsTable(policies_table)
    print(f"Seeded {new_seed_df.count()} new policy row(s).")
else:
    print("Seed policies already present -- nothing to insert. Edit rows directly to change rules.")

display(spark.table(policies_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Next
# MAGIC - Make sure the 5 groups referenced above exist (see README "Access control"
# MAGIC   section) -- this project's demo now uses 5 role groups: `data-engineers`,
# MAGIC   `data-scientists`, `business-analysts`, `data-analysts`, `project-managers`.
# MAGIC - Grant the app's service principal **Can Use** on a SQL Warehouse (needed
# MAGIC   so the Upload tab can read `upload_policies` and write `upload_manifest`
# MAGIC   live) -- see README step "Grant the app a SQL Warehouse".
# MAGIC - Then use the app's **Upload** tab instead of dragging files straight into
# MAGIC   the Volume -- that's what actually applies these policies.

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from news_agent.docs.upload_policies