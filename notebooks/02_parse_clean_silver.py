# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Parse raw bytes into clean plain text (Silver table)
# MAGIC Turns PDF/HTML/TXT bytes into readable text. Skips files it already parsed
# MAGIC (based on path), so it's safe to re-run.

# COMMAND ----------

# MAGIC %pip install pymupdf beautifulsoup4 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "news_agent"
schema = "docs"
bronze_table = f"{catalog}.{schema}.bronze_documents"
silver_table = f"{catalog}.{schema}.silver_documents"

# COMMAND ----------

import re
from pyspark.sql.functions import udf, col, sha2, concat_ws, current_timestamp
from pyspark.sql.types import StringType


def extract_text(content, extension):
    """Best-effort text extraction. Returns '' on failure rather than raising,
    so one bad file never stops the whole pipeline."""
    try:
        if extension == "pdf":
            import fitz  # PyMuPDF
            doc = fitz.open(stream=bytes(content), filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
        elif extension in ("html", "htm"):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(bytes(content), "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        elif extension == "txt":
            text = bytes(content).decode("utf-8", errors="ignore")
        else:
            text = ""
    except Exception:
        text = ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


extract_text_udf = udf(extract_text, StringType())

# COMMAND ----------

bronze_df = spark.table(bronze_table)

already_parsed = set()
if spark.catalog.tableExists(silver_table):
    already_parsed = {r["path"] for r in spark.table(silver_table).select("path").collect()}

new_docs = bronze_df.filter(~col("path").isin(already_parsed)) if already_parsed else bronze_df

silver_df = (
    new_docs
    .withColumn("text", extract_text_udf(col("content"), col("extension")))
    .filter(col("text") != "")
    .withColumn("doc_id", sha2(col("path"), 256))
    .withColumn("title", col("path"))
    .select("doc_id", "path", "title", "extension", "text", "ingestion_time")
    .withColumn("parsed_time", current_timestamp())
)

# COMMAND ----------

if silver_df.limit(1).count() > 0:
    (
        silver_df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(silver_table)
    )
    print(f"Parsed {silver_df.count()} new document(s) into {silver_table}")
else:
    print("No new documents to parse.")

display(spark.table(silver_table).select("doc_id", "path", "extension", "parsed_time"))
