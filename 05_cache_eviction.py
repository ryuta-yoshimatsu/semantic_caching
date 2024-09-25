# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions).

# COMMAND ----------

# MAGIC %md
# MAGIC #Cache eviction
# MAGIC
# MAGIC This notebook walks you through some of the eviction strategies you can employ to your semantic cache. 

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load parameters
from config import Config
config = Config()

# COMMAND ----------

# DBTITLE 1,Set environmental variables
import os

HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

os.environ['HOST'] = HOST
os.environ['TOKEN'] = TOKEN

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaning up the cache
# MAGIC
# MAGIC We instantiate a Vector Search client to interact with a Vector Search endpoint.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from cache import Cache

vsc = VectorSearchClient(
    workspace_url=HOST,
    personal_access_token=TOKEN,
    disable_notice=True,
    )

semantic_cache = Cache(vsc, config)

# COMMAND ----------

semantic_cache.evict(strategy='FIFO', max_documents=4, batch_size=4)

# COMMAND ----------

semantic_cache.evict(strategy='LRU', max_documents=49)

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | | | |
