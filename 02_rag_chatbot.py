# Databricks notebook source
# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from config import Config
config = Config()

# COMMAND ----------

# MAGIC %run ./99_init $reset_all_data=false

# COMMAND ----------

rag_chain_config = {
    "databricks_resources": {
        "llm_model_serving_endpoint_name": config.LLM_MODEL_SERVING_ENDPOINT_NAME,
        "vector_search_endpoint_name": config.VECTOR_SEARCH_ENDPOINT_NAME,
    },
    "input_example": {
        "messages": [{"content": "What is Databricks Model Serving?", "role": "user"}]
    },
    "retriever_config": {
        "vector_search_index": config.VS_INDEX_FULLNAME,
    },
    "llm_config": {
        "llm_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.\n\nContext: {context}""",
    },
}
try:
    with open('chain/chain_config.yaml', 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')

model_config = mlflow.models.ModelConfig(development_config='chain/chain_config.yaml')

# COMMAND ----------

HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.secrets.get(scope="semantic_cache", key="token")
os.environ['TOKEN'] = TOKEN
os.environ['HOST'] = HOST

# COMMAND ----------

# MAGIC %%writefile chain/chain.py
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from langchain_community.vectorstores import DatabricksVectorSearch
# MAGIC from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC import mlflow
# MAGIC import os
# MAGIC
# MAGIC ## Enable MLflow Tracing
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC model_config = mlflow.models.ModelConfig(development_config="chain_config.yaml")
# MAGIC
# MAGIC databricks_resources = model_config.get("databricks_resources")
# MAGIC retriever_config = model_config.get("retriever_config")
# MAGIC llm_config = model_config.get("llm_config")
# MAGIC
# MAGIC # Connect to the Vector Search Index
# MAGIC vs_index = VectorSearchClient(
# MAGIC     workspace_url=os.environ['HOST'],
# MAGIC     personal_access_token=os.environ['TOKEN'],
# MAGIC     disable_notice=True,
# MAGIC     ).get_index(
# MAGIC     endpoint_name=databricks_resources.get("vector_search_endpoint_name"),
# MAGIC     index_name=retriever_config.get("vector_search_index"),
# MAGIC )
# MAGIC
# MAGIC # Turn the Vector Search index into a LangChain retriever
# MAGIC vector_search_as_retriever = DatabricksVectorSearch(
# MAGIC     vs_index,
# MAGIC     text_column="content",
# MAGIC     columns=["id", "content", "url"],
# MAGIC ).as_retriever(search_kwargs={"k": 3}) # Number of search results that the retriever returns
# MAGIC # Enable the RAG Studio Review App and MLFlow to properly display track and display retrieved chunks for evaluation
# MAGIC mlflow.models.set_retriever_schema(primary_key="id", text_column="content", doc_uri="url")
# MAGIC
# MAGIC # Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
# MAGIC def format_context(docs):
# MAGIC     chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
# MAGIC     return "".join(chunk_contents)
# MAGIC
# MAGIC from langchain_core.prompts import ChatPromptTemplate
# MAGIC from langchain_community.chat_models import ChatDatabricks
# MAGIC from operator import itemgetter
# MAGIC
# MAGIC prompt = ChatPromptTemplate.from_messages(
# MAGIC     [
# MAGIC         ("system", f"{llm_config.get('llm_prompt_template')}"),
# MAGIC         ("user", "{question}"),
# MAGIC     ]
# MAGIC )
# MAGIC
# MAGIC # Our foundation model answering the final prompt
# MAGIC model = ChatDatabricks(
# MAGIC     endpoint=databricks_resources.get("llm_model_serving_endpoint_name"),
# MAGIC     extra_params={"temperature": 0.01, "max_tokens": 500}
# MAGIC )
# MAGIC
# MAGIC # Return the string contents of the most recent messages: [{...}] from the user to be used as input question
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]["content"]
# MAGIC
# MAGIC # RAG Chain
# MAGIC chain = (
# MAGIC     {
# MAGIC         "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
# MAGIC         "context": itemgetter("messages")
# MAGIC         | RunnableLambda(extract_user_query_string)
# MAGIC         | vector_search_as_retriever
# MAGIC         | RunnableLambda(format_context),
# MAGIC     }
# MAGIC     | prompt
# MAGIC     | model
# MAGIC     | StrOutputParser()
# MAGIC )
# MAGIC
# MAGIC # Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=chain)

# COMMAND ----------

# Log the model to MLflow
with mlflow.start_run(run_name=f"rag_chatbot"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(os.getcwd(), 'chain/chain.py'),  # Chain code file e.g., /path/to/the/chain.py 
        model_config='chain/chain_config.yaml',  # Chain configuration 
        artifact_path="chain",  # Required by MLflow
        input_example=model_config.get("input_example"),  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
    )

# Test the chain locally
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(model_config.get("input_example"))

# COMMAND ----------

# Register to UC
uc_registered_model_info = mlflow.register_model(
  model_uri=logged_chain_info.model_uri, 
  name=config.MODEL_FULLNAME
  )

# COMMAND ----------

import utils
utils.deploy_model_serving_endpoint(
  spark, 
  config.MODEL_FULLNAME,
  config.CATALOG,
  config.LOGGING_SCHEMA,
  config.ENDPOINT_NAME,
  HOST,
  )

# COMMAND ----------

utils.wait_for_model_serving_endpoint_to_be_ready(config.ENDPOINT_NAME)

# COMMAND ----------

import utils
data = {
    "inputs": {
        "messages": [
            {
                "content": "What is Model Serving?",
                "role": "user"
            }
        ]
    }
}
# Now, call the function with the correctly formatted data
utils.send_request_to_endpoint(
    config.ENDPOINT_NAME, 
    data,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | | | |

# COMMAND ----------


