# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions).

# COMMAND ----------

# MAGIC %md
# MAGIC #Evaluate the RAG chains with and without caching
# MAGIC
# MAGIC (Write what this notebook does in one pargraph.)

# COMMAND ----------

from config import Config
config = Config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preparation

# COMMAND ----------

import pandas as pd
df = pd.read_csv('data/synthetic_questions_100.csv')
df = spark.createDataFrame(df)
df.write.mode('overwrite').saveAsTable(f'{config.CATALOG}.{config.SCHEMA}.synthetic_questions_100')

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {config.CATALOG}.{config.SCHEMA}.synthetic_questions_100_formatted AS
SELECT STRUCT(ARRAY(STRUCT(question AS content, "user" AS role)) AS messages) AS question, base as base
FROM {config.CATALOG}.{config.SCHEMA}.synthetic_questions_100;
""")

df = spark.table(f'{config.CATALOG}.{config.SCHEMA}.synthetic_questions_100_formatted')
display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Test standard rag chain endpoint

# COMMAND ----------

# DBTITLE 1,Load testing standard RAG chain
spark.sql(f"""
CREATE OR REPLACE TABLE {config.CATALOG}.{config.SCHEMA}.standard_rag_chain_results AS
SELECT question, ai_query(
  'standard_rag_chatbot',
  question,
  returnType => 'STRUCT<choices:ARRAY<STRING>>'
  ) AS prediction, base
FROM {config.CATALOG}.{config.SCHEMA}.synthetic_questions_100_formatted;
""")

standard_rag_chain_results = spark.table(f'{config.CATALOG}.{config.SCHEMA}.standard_rag_chain_results')
display(standard_rag_chain_results)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Test rag chain with cache endpoint

# COMMAND ----------

# DBTITLE 1,Load testing RAG chain with cache
spark.sql(f"""
CREATE OR REPLACE TABLE {config.CATALOG}.{config.SCHEMA}.rag_chain_with_cache_results AS
SELECT question, ai_query(
    'rag_chatbot_with_cache',
    question,
    returnType => 'STRUCT<choices:ARRAY<STRING>>'
  ) AS prediction, base
FROM {config.CATALOG}.{config.SCHEMA}.synthetic_questions_100_formatted;
""")

rag_chain_with_cache_results = spark.table(f'{config.CATALOG}.{config.SCHEMA}.rag_chain_with_cache_results')
display(rag_chain_with_cache_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate results using MLflow

# COMMAND ----------

import json
synthetic_qa = []
with open('data/synthetic_qa.txt', 'r') as file:
    for line in file:
        synthetic_qa.append(json.loads(line))

display(synthetic_qa)

# COMMAND ----------

evaluation_standard = spark.table(f'{config.CATALOG}.{config.SCHEMA}.standard_rag_chain_results').toPandas()
evaluation_cache = spark.table(f'{config.CATALOG}.{config.SCHEMA}.rag_chain_with_cache_results').toPandas()

evaluation_standard["question"] = evaluation_standard["question"].apply(lambda x: x["messages"][0]["content"])
evaluation_standard["prediction"] = evaluation_standard["prediction"].apply(lambda x: json.loads(x["choices"][0])["message"]["content"])

evaluation_cache["question"] = evaluation_cache["question"].apply(lambda x: x["messages"][0]["content"])
evaluation_cache["prediction"] = evaluation_cache["prediction"].apply(lambda x: json.loads(x["choices"][0])["message"]["content"])

labels = pd.DataFrame(synthetic_qa).drop(["question"], axis=1)

evaluation_standard = evaluation_standard.merge(labels, on='base')
evaluation_cache = evaluation_cache.merge(labels, on='base')

# COMMAND ----------

evaluation_standard

# COMMAND ----------

evaluation_cache

# COMMAND ----------

# MAGIC %md
# MAGIC https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# MAGIC

# COMMAND ----------

import mlflow
from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")
judge_model = "endpoints:/databricks-meta-llama-3-1-70b-instruct"


with mlflow.start_run(run_name="evaluation_standard"):
    standard_results = mlflow.evaluate(        
        data=evaluation_standard,
        targets="answer",
        predictions="prediction",
        model_type="question-answering",
        extra_metrics=[
          mlflow.metrics.genai.answer_similarity(model=judge_model), 
          mlflow.metrics.genai.answer_correctness(model=judge_model),
          mlflow.metrics.genai.answer_relevance(model=judge_model),
          ],
        evaluator_config={
            'col_mapping': {'inputs': 'question'}
        }
    )

with mlflow.start_run(run_name="evaluation_cache"):
    cache_results = mlflow.evaluate(        
        data=evaluation_cache,
        targets="answer",
        predictions="prediction",
        model_type="question-answering",
        extra_metrics=[
          mlflow.metrics.genai.answer_similarity(model=judge_model), 
          mlflow.metrics.genai.answer_correctness(model=judge_model),
          mlflow.metrics.genai.answer_relevance(model=judge_model),
          ],
        evaluator_config={
            'col_mapping': {'inputs': 'question'}
        }
    )

# COMMAND ----------

print(f"See aggregated evaluation results below: \n{standard_results.metrics}")

# COMMAND ----------

print(f"See aggregated evaluation results below: \n{cache_results.metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query inference log tables

# COMMAND ----------

standard_log = spark.read.table(f"{config.CATALOG}.{config.LOGGING_SCHEMA}.standard_rag_chatbot_payload").toPandas()
display(standard_log)

# COMMAND ----------

cache_log = spark.read.table(f"{config.CATALOG_CACHE}.{config.LOGGING_SCHEMA_CACHE}.rag_chatbot_with_cache_payload").toPandas()
display(cache_log)

# COMMAND ----------

print(f"standard rag chain mean execution time: {round(standard_log['execution_time_ms'].mean()/1000, 4)} seconds")
print(f"rag chain with cache mean execution time: {round(cache_log['execution_time_ms'].mean()/1000, 4)} seconds")

# COMMAND ----------

import json
import numpy as np

cache_trace = np.array(
    cache_log["response"].apply(lambda x: 1 if len(json.loads(x)["databricks_output"]["trace"]["data"]["spans"]) == 6 else 0)
)
print(f"Number of times the query hit the cache: {cache_trace.sum()}/100")

# COMMAND ----------


