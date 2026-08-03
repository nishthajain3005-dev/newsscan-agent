# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest raw files into a Bronze table
# MAGIC Auto Loader watches the volume and picks up every new file (PDF, HTML, TXT)
# MAGIC as raw bytes + metadata. It remembers what it already processed, so re-running
# MAGIC this notebook only picks up NEW files — this is what makes the pipeline safe
# MAGIC to schedule and re-run automatically.
# MAGIC
# MAGIC **Access-policy tagging:** every file also gets looked up in
# MAGIC `upload_manifest` (written by the app's Upload tab — see
# MAGIC `00b_setup_access_policies.py`) to attach `policy_id`, `category`, and
# MAGIC `viewer_groups` — this is what lets the agent later restrict which roles can
# MAGIC see which documents. Files that land in the Volume WITHOUT going through the
# MAGIC app (e.g. dragged in via Catalog Explorer) won't have a manifest row, so they
# MAGIC fall back to `general_docs` / open visibility rather than silently
# MAGIC disappearing or crashing the pipeline.

# COMMAND ----------

catalog = "news_agent"
schema = "docs"
volume = "raw_files"

raw_path = f"/Volumes/{catalog}/{schema}/{volume}"
checkpoint_path = f"/Volumes/{catalog}/{schema}/{volume}/_checkpoints/bronze"
bronze_table = f"{catalog}.{schema}.bronze_documents"
manifest_table = f"{catalog}.{schema}.upload_manifest"

# Fallback tags for files that bypassed the app (no manifest row for their path)
FALLBACK_POLICY_ID = "legacy_manual_upload"
FALLBACK_CATEGORY = "general"
FALLBACK_VIEWER_GROUPS = ["data-engineers", "data-scientists", "business-analysts", "data-analysts", "project-managers"]

# COMMAND ----------

df = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .load(raw_path)
)

df = df.selectExpr(
    "path",
    "modificationTime",
    "length",
    "content",
    "lower(reverse(split(reverse(path), '\\\\.')[0])) as extension",
    "current_timestamp() as ingestion_time",
)

# COMMAND ----------

# trigger(availableNow=True) processes everything currently sitting in the folder
# and then stops — perfect for a scheduled Job. (Use .trigger(processingTime="5 minutes")
# instead if you want this notebook to run continuously.)
(
    df.writeStream
    .format("delta")
    .option("checkpointLocation", checkpoint_path)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(bronze_table)
)

# COMMAND ----------

# MAGIC %md ### Backfill access-policy columns for any rows missing them
# MAGIC Runs as a plain batch step after the stream above lands new rows — join to
# MAGIC the manifest on `path`, and fall back to the open/general defaults for
# MAGIC anything that has no manifest row (manual drops into the Volume).

# COMMAND ----------

from pyspark.sql.functions import col, lit, coalesce, array

if spark.catalog.tableExists(manifest_table):
    manifest_df = (
        spark.table(manifest_table)
        .filter("status = 'accepted'")
        .select(
            col("file_path").alias("path"),
            col("policy_id"),
            col("category"),
            col("viewer_groups"),
            col("uploaded_by"),
        )
    )
else:
    manifest_df = None

bronze_df = spark.table(bronze_table)

if manifest_df is not None:
    existing_extra_cols = [c for c in ["policy_id", "category", "viewer_groups", "uploaded_by"] if c in bronze_df.columns]
    tagged_df = (
        bronze_df.drop(*existing_extra_cols)
        .join(manifest_df, on="path", how="left")
        .withColumn("policy_id", coalesce(col("policy_id"), lit(FALLBACK_POLICY_ID)))
        .withColumn("category", coalesce(col("category"), lit(FALLBACK_CATEGORY)))
        .withColumn("viewer_groups", coalesce(col("viewer_groups"), array(*[lit(g) for g in FALLBACK_VIEWER_GROUPS])))
    )
    (
        tagged_df.write
        .format("delta")
        .mode("overwrite")
        .option("mergeSchema", "true")
        .saveAsTable(bronze_table)
    )

# COMMAND ----------

print(f"Bronze table ready: {bronze_table}")
display(spark.table(bronze_table).select("path", "extension", "length", "policy_id", "category", "viewer_groups", "ingestion_time"))