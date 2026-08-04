# Databricks notebook source
# MAGIC %md
# MAGIC # 06b — Deploy the agent as a serving endpoint
# MAGIC Uses Databricks' own Agent Framework `agents.deploy()` helper instead of a
# MAGIC hand-built REST call. This matters for two reasons:
# MAGIC 1. It automatically wires up authentication for the `resources` (vector
# MAGIC    index + LLM endpoint) declared when the model was logged in notebook 06 —
# MAGIC    a raw `POST /api/2.0/serving-endpoints` call does NOT do this, which is
# MAGIC    almost always why a hand-rolled deploy fails with an opaque
# MAGIC    "MLflow raised an error loading the model" and "no replicas running".
# MAGIC 2. It gives synchronous, readable errors instead of a generic 200 response
# MAGIC    that only fails later, invisibly, when the container tries to start.
# MAGIC
# MAGIC Run notebook 06 first (log + register). This notebook only deploys whatever
# MAGIC version is currently tagged `@champion`.

# COMMAND ----------

# MAGIC %pip install -U "mlflow>=3.1.3" "databricks-agents>=1.1.0" -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

registered_model_name = "news_agent.docs.news_qa_agent"

model_version = client.get_model_version_by_alias(registered_model_name, "champion").version
print(f"Deploying {registered_model_name}, version {model_version} (currently tagged @champion)")

# COMMAND ----------

# This model uses a custom schema (question, user_groups) instead of the
# ChatCompletionRequest schema that agents.deploy() requires.
# Use mlflow.deployments instead (see cells below for the correct approach).

import mlflow.deployments

serving_endpoint_name = "news_qa_agent_endpoint"

deploy_client = mlflow.deployments.get_deploy_client("databricks")

endpoint_config = {
    "served_entities": [
        {
            "entity_name": registered_model_name,
            "entity_version": str(model_version),
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }
    ]
}

existing = None
try:
    existing = deploy_client.get_endpoint(endpoint=serving_endpoint_name)
except Exception:
    pass  # doesn't exist yet -- that's fine, we'll create it

if existing:
    print(f"Endpoint {serving_endpoint_name} already exists -- updating it to version {model_version}.")
    result = deploy_client.update_endpoint(endpoint=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Creating new endpoint {serving_endpoint_name}.")
    result = deploy_client.create_endpoint(name=serving_endpoint_name, config=endpoint_config)

print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### If this cell raises an error (rather than the endpoint failing silently later)
# MAGIC That's actually the point of using `agents.deploy()` — read the error message
# MAGIC directly, it's usually specific (e.g. a permissions issue on the vector index,
# MAGIC or the model version not found). Common causes at this stage:
# MAGIC - You skipped re-running notebook 06 after the `resources=` fix, so
# MAGIC   `@champion` still points at an older, broken version — check the version
# MAGIC   number printed above against what you expect.
# MAGIC - Your user account doesn't have `EXECUTE` / query permission on the vector
# MAGIC   search index or the `databricks-meta-llama-3-3-70b-instruct` endpoint —
# MAGIC   `agents.deploy()` uses YOUR permissions at deploy time to set up the
# MAGIC   endpoint's access, so you need to be able to reach both yourself.
# MAGIC
# MAGIC ### Checking status afterward
# MAGIC Left sidebar → **Serving** → `news_qa_agent_endpoint` (or whatever name
# MAGIC `agents.deploy()` printed above) → **Deployments** tab. First deployment
# MAGIC commonly takes 5-15 minutes.
# MAGIC
# MAGIC ### Re-deploying a new version later
# MAGIC Just re-run notebook 06 (registers a new version, moves `@champion`), then
# MAGIC re-run this notebook — `agents.deploy()` updates the existing endpoint in
# MAGIC place rather than requiring you to delete/recreate it.

# COMMAND ----------

# MAGIC %md
# MAGIC # 06b — Deploy the agent as a serving endpoint
# MAGIC Uses the `mlflow.deployments` SDK (the standard way to create/update a Model
# MAGIC Serving endpoint) rather than `databricks.agents.deploy()`. Agent Framework's
# MAGIC `deploy()` only accepts models with its standardized chat schema
# MAGIC (`ChatCompletionRequest`-style `{"messages": [...]}`) — our agent uses a
# MAGIC simpler `{"question": ...}` schema, which is fine for regular Model Serving
# MAGIC but gets rejected by that stricter helper. This approach has no such
# MAGIC restriction, and — now that notebook 06 declares `resources=[...]` when
# MAGIC logging the model — it correctly picks up authentication for the vector
# MAGIC index and LLM endpoint too, which was the actual root cause of the earlier
# MAGIC "no replicas running" failures (not how the endpoint gets created).
# MAGIC
# MAGIC Run notebook 06 first (log + register). This notebook deploys whatever
# MAGIC version is currently tagged `@champion`.

# COMMAND ----------

# MAGIC %pip install -U mlflow -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

registered_model_name = "news_agent.docs.news_qa_agent"
serving_endpoint_name = "news_qa_agent_endpoint"

model_version = client.get_model_version_by_alias(registered_model_name, "champion").version
print(f"Deploying {registered_model_name}, version {model_version} (currently tagged @champion)")

# COMMAND ----------

import mlflow.deployments

deploy_client = mlflow.deployments.get_deploy_client("databricks")

endpoint_config = {
    "served_entities": [
        {
            "entity_name": registered_model_name,
            "entity_version": str(model_version),
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }
    ]
}

existing = None
try:
    existing = deploy_client.get_endpoint(endpoint=serving_endpoint_name)
except Exception:
    pass  # doesn't exist yet -- that's fine, we'll create it

if existing:
    print(f"Endpoint {serving_endpoint_name} already exists -- updating it to version {model_version}.")
    result = deploy_client.update_endpoint(endpoint=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Creating new endpoint {serving_endpoint_name}.")
    result = deploy_client.create_endpoint(name=serving_endpoint_name, config=endpoint_config)

print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### If this cell raises an error (rather than the endpoint failing silently later)
# MAGIC Read the error message directly -- it's usually specific. Common causes:
# MAGIC - You skipped re-running notebook 06 after the `resources=` fix, so
# MAGIC   `@champion` still points at an older, broken version -- check the version
# MAGIC   number printed above against what you expect.
# MAGIC - Your account doesn't have permission to create serving endpoints -- ask
# MAGIC   your workspace admin.
# MAGIC
# MAGIC ### Checking status afterward
# MAGIC Left sidebar -> **Serving** -> `news_qa_agent_endpoint` -> **Deployments**
# MAGIC tab. First deployment commonly takes 5-15 minutes. If it fails, check the
# MAGIC **Logs** tab for the actual container error.
# MAGIC
# MAGIC ### Re-deploying a new version later
# MAGIC Just re-run notebook 06 (registers a new version, moves `@champion`), then
# MAGIC re-run this notebook -- it detects the endpoint already exists and updates
# MAGIC it in place rather than requiring you to delete/recreate it.