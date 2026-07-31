# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest raw files into a Bronze table
# MAGIC Auto Loader watches the volume and picks up every new file (PDF, HTML, TXT)
# MAGIC as raw bytes + metadata. It remembers what it already processed, so re-running
# MAGIC this notebook only picks up NEW files — this is what makes the pipeline safe
# MAGIC to schedule and re-run automatically.

# COMMAND ----------

catalog = "news_agent"
schema = "docs"
volume = "raw_files"

raw_path = f"/Volumes/{catalog}/{schema}/{volume}"
checkpoint_path = f"/Volumes/{catalog}/{schema}/_checkpoints/bronze"
bronze_table = f"{catalog}.{schema}.bronze_documents"

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

print(f"Bronze table ready: {bronze_table}")
display(spark.table(bronze_table).select("path", "extension", "length", "ingestion_time"))
