# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Create the Vector Search index
# MAGIC This is the "search engine" the agent will query. Databricks Vector Search
# MAGIC keeps it automatically in sync with the gold_chunks table (Delta Sync index)
# MAGIC and computes embeddings for you using a hosted embedding model — you don't
# MAGIC need to manage embeddings yourself.
# MAGIC
# MAGIC Run this after step 03 to create the endpoint + index the first time. It's
# MAGIC also safe (and necessary) to re-run any time after 03 adds new documents —
# MAGIC this index uses TRIGGERED sync, meaning it does NOT pick up new rows on its
# MAGIC own; re-running this notebook is what tells it to sync.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "news_agent"
schema = "docs"
gold_table = f"{catalog}.{schema}.gold_chunks"

vs_endpoint = "news_agent_vs_endpoint"
vs_index_name = f"{catalog}.{schema}.gold_chunks_index"
embedding_model_endpoint = "databricks-bge-large-en"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

existing_endpoints = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if vs_endpoint not in existing_endpoints:
    vsc.create_endpoint(name=vs_endpoint, endpoint_type="STANDARD")
    print(f"Creating endpoint {vs_endpoint} — this can take a few minutes.")
else:
    print(f"Endpoint {vs_endpoint} already exists.")

# COMMAND ----------

# Wait for the endpoint to be ONLINE before creating the index
import time

while True:
    status = vsc.get_endpoint(vs_endpoint)["endpoint_status"]["state"]
    print(f"Endpoint status: {status}")
    if status == "ONLINE":
        break
    time.sleep(30)

# COMMAND ----------

existing_indexes = [i["name"] for i in vsc.list_indexes(vs_endpoint).get("vector_indexes", [])]

if vs_index_name not in existing_indexes:
    vsc.create_delta_sync_index(
        endpoint_name=vs_endpoint,
        source_table_name=gold_table,
        index_name=vs_index_name,
        pipeline_type="TRIGGERED",  # sync manually / on schedule; use "CONTINUOUS" for near-real-time
        primary_key="chunk_id",
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name=embedding_model_endpoint,
    )
    print(f"Created index {vs_index_name}. Waiting for it to finish its first sync...")
else:
    # A TRIGGERED index does NOT pick up new/changed rows in gold_chunks on its
    # own — it only syncs when told to. Since this notebook is meant to be safe
    # to re-run any time after notebook 03 adds new chunks (including as part of
    # the scheduled Job), explicitly trigger a sync every time we get here,
    # rather than assuming "already exists" means "already up to date".
    print(f"Index {vs_index_name} already exists — triggering a sync to pick up any new rows.")
    vsc.get_index(endpoint_name=vs_endpoint, index_name=vs_index_name).sync()

# COMMAND ----------

# IMPORTANT: the endpoint being ONLINE (checked above) does NOT mean the index
# is ready — the endpoint is just the serving infrastructure, while the index
# has its own separate sync/embedding status. Always check the index itself.
#
# Also note: `ready` can be True even while a re-sync is still in progress
# (state ONLINE_TRIGGERED_UPDATE) — the index was already online from before,
# it's just now also processing new rows in the background. If you only check
# `ready`, you can move on before newly added documents have actually finished
# being embedded. Wait for the fully-settled state instead.
index = vsc.get_index(endpoint_name=vs_endpoint, index_name=vs_index_name)

while True:
    status = index.describe()["status"]
    state = status.get("detailed_state", "")
    print(f"{state} | {status.get('message')}")
    if state == "ONLINE_NO_PENDING_UPDATE":
        break
    time.sleep(30)

print(f"Indexed rows: {index.describe()['status'].get('indexed_row_count')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### To pick up newly chunked documents later
# MAGIC Just re-run this whole notebook (Run All) any time after step 03 adds new
# MAGIC rows — the cell above now detects the index already exists and triggers a
# MAGIC fresh sync automatically, then waits for it to fully finish. This is also
# MAGIC exactly what the scheduled Job does — see jobs/ingestion_workflow.json.
# MAGIC
# MAGIC ### Endpoint status vs. index status — don't confuse the two
# MAGIC The Catalog Explorer / Compute UI page for the endpoint shows the endpoint's
# MAGIC status (`ONLINE`/`OFFLINE`) — that's just whether the serving infrastructure
# MAGIC is up. It says nothing about whether a given index on it has finished syncing
# MAGIC and embedding your data. An endpoint can show `ONLINE` in the UI while an
# MAGIC index on it is still provisioning or has failed — always check
# MAGIC `index.describe()["status"]["ready"]` in code (as this notebook now does)
# MAGIC rather than relying on the endpoint's UI status alone.
