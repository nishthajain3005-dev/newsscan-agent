# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Split documents into chunks (Gold table)
# MAGIC Vector search works on short passages, not whole articles. This splits each
# MAGIC document into ~800-character overlapping chunks so the agent can retrieve the
# MAGIC exact relevant passage instead of an entire article.

# COMMAND ----------

# MAGIC %pip install langchain langchain-text-splitters -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "news_agent"
schema = "docs"
silver_table = f"{catalog}.{schema}.silver_documents"
gold_table = f"{catalog}.{schema}.gold_chunks"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql.functions import udf, col, explode, sha2, concat_ws, current_timestamp
from pyspark.sql.types import ArrayType, StringType

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def split_text(text):
    return splitter.split_text(text)


split_udf = udf(split_text, ArrayType(StringType()))

# COMMAND ----------

already_chunked = set()
if spark.catalog.tableExists(gold_table):
    already_chunked = {r["doc_id"] for r in spark.table(gold_table).select("doc_id").distinct().collect()}

silver_df = spark.table(silver_table)
new_docs = silver_df.filter(~col("doc_id").isin(already_chunked)) if already_chunked else silver_df

gold_df = (
    new_docs
    .withColumn("chunk_text", explode(split_udf(col("text"))))
    .withColumn("chunk_id", sha2(concat_ws("||", col("doc_id"), col("chunk_text")), 256))
    .withColumn("chunked_time", current_timestamp())
    .select(
        "chunk_id", "doc_id", "path", "title", "chunk_text", "chunked_time",
        "policy_id", "category", "viewer_groups",  # access-policy metadata, carried through from silver
    )
)

# COMMAND ----------

if gold_df.limit(1).count() > 0:
    (
        gold_df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(gold_table)
    )
    # Vector Search's Delta Sync index requires Change Data Feed on the source table
    spark.sql(f"ALTER TABLE {gold_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"Wrote {gold_df.count()} new chunk(s) to {gold_table}")
else:
    print("No new documents to chunk.")

display(spark.table(gold_table).limit(10))