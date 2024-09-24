# Databricks notebook source
# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from config import Config
config = Config()

# COMMAND ----------

# MAGIC %run ./99_init $reset_all_data=false

# COMMAND ----------

import utils
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

if not utils.vs_endpoint_exists(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME):
    utils.create_or_wait_for_endpoint(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME)

print(f"Endpoint named {config.VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient


if not utils.index_exists(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME, config.VS_INDEX_FULLNAME):
  
  print(f"Creating index {config.VS_INDEX_FULLNAME} on endpoint {config.VECTOR_SEARCH_ENDPOINT_NAME}...")
  
  vsc.create_delta_sync_index(
    endpoint_name=config.VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=config.VS_INDEX_FULLNAME,
    source_table_name=config.SOURCE_TABLE_FULLNAME,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column='content', # The column containing our text
    embedding_model_endpoint_name=config.EMBEDDING_MODEL_SERVING_ENDPOINT_NAME, #The embedding endpoint used to create the embeddings
  )
  
  # Let's wait for the index to be ready and all our embeddings to be created and indexed
  utils.wait_for_index_to_be_ready(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME, config.VS_INDEX_FULLNAME)
else:
  # Trigger a sync to update our vs content with the new data saved in the table
  utils.wait_for_index_to_be_ready(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME, config.VS_INDEX_FULLNAME)
  vsc.get_index(config.VECTOR_SEARCH_ENDPOINT_NAME, config.VS_INDEX_FULLNAME).sync()

print(f"index {config.VS_INDEX_FULLNAME} on table {config.SOURCE_TABLE_FULLNAME} is ready")

# COMMAND ----------

results = vsc.get_index(config.VECTOR_SEARCH_ENDPOINT_NAME, config.VS_INDEX_FULLNAME).similarity_search(
  query_text="What is Model Serving?",
  columns=["url", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | | | |
