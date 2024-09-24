# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions).

# COMMAND ----------

# MAGIC %md
# MAGIC #Evict the cache
# MAGIC
# MAGIC (Write what this notebook does in one pargraph.)

# COMMAND ----------

# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

import os
from config import Config
config = Config()

HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.secrets.get(scope="semantic_cache", key="token")

os.environ['HOST'] = HOST
os.environ['TOKEN'] = TOKEN

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
