# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Log, register, and deploy the agent
# MAGIC Turns the agent from notebook 05 into a real endpoint you (or an app) can call
# MAGIC over REST, with versioning and governance via Unity Catalog.
# MAGIC
# MAGIC This notebook defines `NewsQAAgent` again directly (rather than `%run`-ing
# MAGIC notebook 05) on purpose: notebook 05 has its own `%pip install` +
# MAGIC `dbutils.library.restartPython()` cell, and restarting Python in the middle
# MAGIC of a `%run` wipes out the calling notebook's variables — the class would get
# MAGIC defined in a process that immediately gets discarded, causing a
# MAGIC `NameError: name 'NewsQAAgent' is not defined` right after. Keeping this
# MAGIC notebook self-contained avoids that entirely.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch mlflow databricks-langchain -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

chat_model_endpoint = "databricks-meta-llama-3-3-70b-instruct"
vs_endpoint = "news_agent_vs_endpoint"
vs_index_name = "news_agent.docs.gold_chunks_index"
NUM_RESULTS = 5

# COMMAND ----------

import mlflow


class NewsQAAgent(mlflow.pyfunc.PythonModel):
    """Retrieval-augmented Q&A agent over the ingested documents.
    (Same definition as notebook 05 — kept in sync manually; see the note above
    for why this isn't shared via %run.)"""

    def load_context(self, context):
        from databricks.vector_search.client import VectorSearchClient
        from databricks_langchain import ChatDatabricks

        self.vsc = VectorSearchClient(disable_notice=True)
        self.index = self.vsc.get_index(endpoint_name=vs_endpoint, index_name=vs_index_name)
        self.llm = ChatDatabricks(endpoint=chat_model_endpoint, max_tokens=1000, temperature=0.1)

    def _retrieve(self, question):
        results = self.index.similarity_search(
            query_text=question,
            columns=["chunk_text", "path", "title"],
            num_results=NUM_RESULTS,
        )
        rows = results.get("result", {}).get("data_array", [])
        return rows

    def _answer_one(self, question):
        rows = self._retrieve(question)
        if not rows:
            return {
                "answer": "I couldn't find anything about that in the ingested documents.",
                "sources": [],
            }

        context_block = "\n\n".join(f"[Source: {r[2]}]\n{r[0]}" for r in rows)
        prompt = f"""You are a research assistant answering questions using ONLY the newspaper
article excerpts below. If the excerpts don't contain the answer, say you don't know —
do not make anything up. Answer in clear, plain, human-friendly language (a few sentences,
not bullet points, unless the question asks for a list). End with the source titles you used.

Excerpts:
{context_block}

Question: {question}
"""
        response = self.llm.invoke(prompt)
        return {
            "answer": response.content,
            "sources": sorted({r[2] for r in rows}),
        }

    def predict(self, context, model_input):
        questions = model_input["question"].tolist()
        return [self._answer_one(q) for q in questions]

# COMMAND ----------

import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

mlflow.set_registry_uri("databricks-uc")

registered_model_name = "news_agent.docs.news_qa_agent"

input_schema = Schema([ColSpec("string", "question")])
output_schema = Schema([ColSpec("string", "answer"), ColSpec("string", "sources")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# COMMAND ----------

with mlflow.start_run(run_name="news_qa_agent"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=NewsQAAgent(),
        signature=signature,
        input_example=pd.DataFrame({"question": ["What happened in the latest article?"]}),
        pip_requirements=[
            "mlflow",
            "databricks-vectorsearch",
            "databricks-langchain",
        ],
        registered_model_name=registered_model_name,
    )

print(f"Registered as: {registered_model_name}, version {model_info.registered_model_version}")
print(f"Model URI: {model_info.model_uri}")

# Unity Catalog model registry doesn't use the old numeric-stage "latest" concept —
# instead you tag a specific version with an alias (e.g. "champion") and always
# refer to it by that alias. This is what lets notebook 07 and the app always
# point at "whichever version is current" without hardcoding a version number.
from mlflow import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(
    name=registered_model_name,
    alias="champion",
    version=model_info.registered_model_version,
)
print(f"Alias 'champion' now points to version {model_info.registered_model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy as a serving endpoint
# MAGIC This creates a REST endpoint you can call from anywhere (a web app, Slack bot,
# MAGIC or just curl) to get answers, pointed at the version tagged `@champion` above.

# COMMAND ----------

model_version = model_info.registered_model_version

# COMMAND ----------

import requests

serving_endpoint_name = "news_qa_agent_endpoint"
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

endpoint_config = {
    "name": serving_endpoint_name,
    "config": {
        "served_entities": [
            {
                "entity_name": registered_model_name,
                "entity_version": model_version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ]
    },
}

resp = requests.post(
    f"https://{workspace_url}/api/2.0/serving-endpoints",
    headers={"Authorization": f"Bearer {token}"},
    json=endpoint_config,
)
print(resp.status_code, resp.text)

# COMMAND ----------

# MAGIC %md
# MAGIC If the endpoint already exists and you're deploying a new model version,
# MAGIC use this instead (PUT the config to update it):
# MAGIC ```python
# MAGIC requests.put(
# MAGIC     f"https://{workspace_url}/api/2.0/serving-endpoints/{serving_endpoint_name}/config",
# MAGIC     headers={"Authorization": f"Bearer {token}"},
# MAGIC     json=endpoint_config["config"],
# MAGIC )
# MAGIC ```
# MAGIC Endpoint creation takes several minutes. Check status under
# MAGIC **Serving** in the left sidebar of the Databricks UI.
# MAGIC
# MAGIC Each time you re-run this whole notebook, it registers a brand-new model
# MAGIC version and moves the `@champion` alias to point at it — you never need to
# MAGIC hunt down a version number by hand.
