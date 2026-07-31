# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Ask the agent questions
# MAGIC Two ways: directly from a notebook (fastest for testing), or via the REST
# MAGIC endpoint (what a real app would use).

# COMMAND ----------

# MAGIC %md ### Option A — query the serving endpoint directly

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch databricks-langchain mlflow --no-cache-dir -v
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import requests

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
serving_endpoint_name = "news_qa_agent_endpoint"


def ask(question: str):
    resp = requests.post(
        f"https://{workspace_url}/serving-endpoints/{serving_endpoint_name}/invocations",
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [{"question": question}]},
    )
    return resp.json()


ask("What are the main topics covered in the documents I uploaded?")

# COMMAND ----------

# MAGIC %md ### Option B — query the MLflow model directly (no serving endpoint needed)
# MAGIC Loading the model this way actually runs its code in *this* notebook's
# MAGIC session — so this session needs the same libraries the agent imports
# MAGIC (`databricks-vectorsearch`, `databricks-langchain`), not just mlflow.

# COMMAND ----------

# MAGIC %pip install mlflow databricks-vectorsearch databricks-langchain -q
# MAGIC dbutils.library.restartPython()
# MAGIC # Note: this restart clears variables from Option A above (workspace_url, token,
# MAGIC # ask()) — if you want to rerun Option A after this, just re-run its cells too.

# COMMAND ----------

import mlflow
import pandas as pd

mlflow.set_registry_uri("databricks-uc")

registered_model_name = "news_agent.docs.news_qa_agent"
# Unity Catalog doesn't support "models:/name/latest" (that's a legacy Workspace
# Registry concept) — load by the "champion" alias set in notebook 06 instead.
model = mlflow.pyfunc.load_model(f"models:/{registered_model_name}@champion")

model.predict(pd.DataFrame({"question": ["Summarize the latest article in two sentences."]}))