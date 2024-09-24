# Databricks notebook source
# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from config import Config
config = Config()

# COMMAND ----------

import cache
cache.evict(strategy='FIFO', max_documents=4, batch_size=4)

# COMMAND ----------

cache.evict(strategy='LRU', max_documents=49)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | | | |
