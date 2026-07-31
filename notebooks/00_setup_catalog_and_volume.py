# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — One-time setup
# MAGIC Creates the Unity Catalog catalog, schema, and Volume where your raw
# MAGIC documents will live. Run this once. Safe to re-run (uses IF NOT EXISTS).

# COMMAND ----------

catalog = "news_agent"
schema = "docs"
volume = "raw_files"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")

# a subfolder Auto Loader will use to remember what it already ingested
dbutils.fs.mkdirs(f"/Volumes/{catalog}/{schema}/_checkpoints/bronze")

print("Setup complete.")
print(f"Upload your PDF / HTML / TXT files here (drag-and-drop works in the Databricks UI, "
      f"under Catalog > {catalog} > {schema} > Volumes > {volume}):")
print(f"  /Volumes/{catalog}/{schema}/{volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick way to upload files from this notebook instead of the UI
# MAGIC If you already have files on your machine, easiest is: Catalog Explorer UI ->
# MAGIC navigate to the volume above -> "Upload to this volume" button.
# MAGIC
# MAGIC If you have files somewhere else Databricks can already see (e.g. DBFS, another
# MAGIC volume, a mounted path), copy them like this:
# MAGIC ```python
# MAGIC dbutils.fs.cp("file:/some/source/path/article.pdf",
# MAGIC               f"/Volumes/{catalog}/{schema}/{volume}/article.pdf")
# MAGIC ```
