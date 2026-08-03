# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Build and test the agent
# MAGIC The agent does 4 things per question:
# MAGIC 1. Search the vector index for candidate chunks
# MAGIC 2. **Filter out any chunk the asking user's role isn't allowed to see**
# MAGIC    (role-based document visibility — see below)
# MAGIC 3. Hand the surviving chunks + the question to an LLM
# MAGIC 4. Return a plain-language answer, with the source documents it used
# MAGIC
# MAGIC This notebook defines the agent as an MLflow "pyfunc" model — a standard
# MAGIC wrapper that step 06 will log and deploy as a serving endpoint.
# MAGIC
# MAGIC ### Role-based document visibility
# MAGIC Every chunk in the gold table carries a `viewer_groups` array (set by the
# MAGIC policy the source document was uploaded under — see
# MAGIC `00b_setup_access_policies.py` and the app's Upload tab). The agent now takes
# MAGIC a second input, `user_groups` (the asking user's Databricks group
# MAGIC memberships, comma-separated), over-fetches candidates from the vector index,
# MAGIC and drops any chunk whose `viewer_groups` doesn't overlap the asker's groups
# MAGIC BEFORE it ever reaches the LLM or the answer — e.g. a résumé tagged
# MAGIC `viewer_groups=[data-engineers, data-analysts]` never surfaces for a Business
# MAGIC Analyst's question, even if it's the closest semantic match.
# MAGIC
# MAGIC If `user_groups` is empty/omitted, filtering is skipped (open access) — this
# MAGIC keeps ad-hoc notebook testing (step 07) simple, but means the *app* is what's
# MAGIC actually responsible for always passing the caller's real groups in
# MAGIC production. This is an application-layer control, not Unity Catalog row-level
# MAGIC security — good enough for "don't surface this in answers to the wrong role"
# MAGIC but not a substitute for UC-level ACLs if the underlying files themselves
# MAGIC need to be locked down too.

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
OVER_FETCH_MULTIPLIER = 4  # fetch this many x NUM_RESULTS candidates before filtering by viewer_groups

# COMMAND ----------

import mlflow


class NewsQAAgent(mlflow.pyfunc.PythonModel):
    """Retrieval-augmented Q&A agent over the ingested documents, with
    role-based filtering of which chunks a given asker is allowed to see."""

    def load_context(self, context):
        from databricks.vector_search.client import VectorSearchClient
        from databricks_langchain import ChatDatabricks

        self.vsc = VectorSearchClient(disable_notice=True)
        self.index = self.vsc.get_index(endpoint_name=vs_endpoint, index_name=vs_index_name)
        self.llm = ChatDatabricks(endpoint=chat_model_endpoint, max_tokens=1000, temperature=0.1)

    def _retrieve(self, question, user_groups):
        num_candidates = NUM_RESULTS * OVER_FETCH_MULTIPLIER if user_groups else NUM_RESULTS
        results = self.index.similarity_search(
            query_text=question,
            columns=["chunk_text", "path", "title", "category", "viewer_groups"],
            num_results=num_candidates,
        )
        rows = results.get("result", {}).get("data_array", [])
        # each row: [chunk_text, path, title, category, viewer_groups]

        if user_groups:
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
        # model_input is a pandas DataFrame with a "question" column and an
        # optional "user_groups" column (comma-separated group names; "" or
        # missing means no filtering -- see the role-based visibility note above)
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

# MAGIC %md ### Quick local test (before logging/deploying anything)

# COMMAND ----------

import pandas as pd

agent = NewsQAAgent()
agent.load_context(None)

test_input = pd.DataFrame({
    "question": ["What is the most recent article about, in a sentence?"],
    "user_groups": [""],  # "" = no role filtering, for a quick unrestricted sanity check
})
result = agent.predict(None, test_input)
result

# COMMAND ----------

# MAGIC %md ### Test role-based filtering
# MAGIC Try the same kind of question a Project Manager might ask about résumés
# MAGIC they uploaded -- they should NOT see résumé content back, since
# MAGIC `resumes_pm_only`'s `viewer_groups` is `[data-engineers, data-analysts]`.

# COMMAND ----------

pm_input = pd.DataFrame({
    "question": ["Summarize the resume that was uploaded."],
    "user_groups": ["project-managers"],
})
agent.predict(None, pm_input)

# COMMAND ----------

de_input = pd.DataFrame({
    "question": ["Summarize the resume that was uploaded."],
    "user_groups": ["data-engineers"],
})
agent.predict(None, de_input)