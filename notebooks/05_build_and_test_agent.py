# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Build and test the agent
# MAGIC The agent does 3 things per question:
# MAGIC 1. Search the vector index for the most relevant chunks
# MAGIC 2. Hand those chunks + the question to an LLM
# MAGIC 3. Return a plain-language answer, with the source documents it used
# MAGIC
# MAGIC This notebook defines the agent as an MLflow "pyfunc" model — a standard
# MAGIC wrapper that step 06 will log and deploy as a serving endpoint.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch mlflow databricks-langchain -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "news_agent"
schema = "docs"
vs_endpoint = "news_agent_vs_endpoint"
vs_index_name = f"{catalog}.{schema}.gold_chunks_index"
chat_model_endpoint = "databricks-meta-llama-3-3-70b-instruct"
NUM_RESULTS = 5

# COMMAND ----------

import mlflow


class NewsQAAgent(mlflow.pyfunc.PythonModel):
    """Retrieval-augmented Q&A agent over the ingested documents."""

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
        return rows  # each row: [chunk_text, path, title]

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
        # model_input is a pandas DataFrame with a "question" column
        questions = model_input["question"].tolist()
        return [self._answer_one(q) for q in questions]

# COMMAND ----------

# MAGIC %md ### Quick local test (before logging/deploying anything)

# COMMAND ----------

import pandas as pd

agent = NewsQAAgent()
agent.load_context(None)

test_input = pd.DataFrame({"question": ["What is the most recent article about, in a sentence?"]})
result = agent.predict(None, test_input)
result