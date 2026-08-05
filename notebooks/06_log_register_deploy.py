# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Log and register the agent
# MAGIC Packages the agent from notebook 05 into a versioned, governed MLflow model
# MAGIC in Unity Catalog. Deployment as a serving endpoint happens in the next
# MAGIC notebook, `06b_deploy_serving_endpoint.py`.
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

chat_model_endpoint = "databricks-meta-llama-3-3-70b-instruct"  # <- must match notebook 05 and your Serving tab
vs_endpoint = "news_agent_vs_endpoint"
vs_index_name = "news_agent.docs.gold_chunks_index"
NUM_RESULTS = 5
OVER_FETCH_MULTIPLIER = 4  # fetch this many x NUM_RESULTS candidates before filtering by viewer_groups

# COMMAND ----------

import mlflow


class NewsQAAgent(mlflow.pyfunc.PythonModel):
    """Retrieval-augmented Q&A agent over the ingested documents, with
    role-based filtering of which chunks a given asker is allowed to see.
    (Same definition as notebook 05 — kept in sync manually; see the note above
    for why this isn't shared via %run.)"""

    def load_context(self, context):
        from databricks.vector_search.client import VectorSearchClient
        from databricks_langchain import ChatDatabricks

        self.vsc = VectorSearchClient(disable_notice=True)
        self.index = self.vsc.get_index(endpoint_name=vs_endpoint, index_name=vs_index_name)
        self.llm = ChatDatabricks(endpoint=chat_model_endpoint, max_tokens=1000, temperature=0.1)

    def _retrieve(self, question, user_groups):
        # Always over-fetch + filter -- including when user_groups is empty.
        # An empty/unknown group set must mean "can see nothing restricted",
        # NOT "no filter requested" -- see notebook 05's comment for why.
        results = self.index.similarity_search(
            query_text=question,
            columns=["chunk_text", "path", "title", "category", "viewer_groups"],
            num_results=NUM_RESULTS * OVER_FETCH_MULTIPLIER,
        )
        rows = results.get("result", {}).get("data_array", [])
        user_group_set = set(user_groups)
        rows = [r for r in rows if user_group_set & set(r[4] or [])]

        return rows[:NUM_RESULTS]

    def _answer_one(self, question, user_groups):
        rows = self._retrieve(question, user_groups)
        if not rows:
            return {
                "answer": "I couldn't find anything about that in the documents you have access to.",
                "sources": [],
            }

        context_block = "\n\n".join(f"[Source: {r[2]}]\n{r[0]}" for r in rows)
        prompt = f"""You are a research assistant answering questions using ONLY the document
excerpts below. If the excerpts don't contain the answer, say you don't know —
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
        if "user_groups" in model_input.columns:
            groups_col = model_input["user_groups"].fillna("").tolist()
        else:
            groups_col = [""] * len(questions)

        results = []
        for q, groups_str in zip(questions, groups_col):
            user_groups = [g.strip() for g in groups_str.split(",") if g.strip()]
            results.append(self._answer_one(q, user_groups))
        return results

# COMMAND ----------

import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

mlflow.set_registry_uri("databricks-uc")

registered_model_name = "news_agent.docs.news_qa_agent"

# "user_groups" is a comma-separated string of the asking user's Databricks
# group names (e.g. "data-engineers,project-managers"). Empty string = no
# role-based filtering (used by ad-hoc notebook testing in 07). The app
# ALWAYS sends the caller's real groups -- see app/app.py's ask_agent().
input_schema = Schema([ColSpec("string", "question"), ColSpec("string", "user_groups")])
output_schema = Schema([ColSpec("string", "answer"), ColSpec("string", "sources")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Tell Model Serving which OTHER Databricks resources this model calls at inference
# time (the vector index for retrieval, the LLM endpoint for generation). Without
# this, the deployed container has no credentials to reach either one and fails
# to load with an opaque "MLflow raised an error loading the model" — it's not
# enough for your own notebook session to have access; the serving container is
# a separate identity and needs this declared explicitly.
resources = [
    DatabricksVectorSearchIndex(index_name=vs_index_name),
    DatabricksServingEndpoint(endpoint_name=chat_model_endpoint),
]

# COMMAND ----------

with mlflow.start_run(run_name="news_qa_agent"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=NewsQAAgent(),
        signature=signature,
        input_example=pd.DataFrame({
            "question": ["What happened in the latest article?"],
            "user_groups": ["data-engineers,project-managers"],
        }),
        pip_requirements=[
            "mlflow",
            "databricks-vectorsearch",
            "databricks-langchain",
        ],
        resources=resources,
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
# MAGIC ### Next step
# MAGIC This notebook only logs and registers the model. Deploying it as a serving
# MAGIC endpoint is handled by **`06b_deploy_serving_endpoint.py`** — run that next.
# MAGIC (Splitting it out uses Databricks' own `agents.deploy()` helper, which is
# MAGIC far more reliable than hand-building the REST call.)
# MAGIC
# MAGIC Each time you re-run this whole notebook, it registers a brand-new model
# MAGIC version and moves the `@champion` alias to point at it — 06b will always
# MAGIC pick up whatever `@champion` currently points to.