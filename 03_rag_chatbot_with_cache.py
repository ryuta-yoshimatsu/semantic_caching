# Databricks notebook source
# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from config import Config
config = Config()

# COMMAND ----------

# MAGIC %run ./99_init $reset_all_data=false

# COMMAND ----------

import cache
cache.clear_cache()

# COMMAND ----------

cache.create_cache()

# COMMAND ----------

cache.warm_cache()

# COMMAND ----------

rag_chain_config = {
    "databricks_resources": {
        "llm_model_serving_endpoint_name": config.LLM_MODEL_SERVING_ENDPOINT_NAME,
        "embedding_model_serving_endpoint_name": config.EMBEDDING_MODEL_SERVING_ENDPOINT_NAME,
        "vector_search_endpoint_name": config.VECTOR_SEARCH_ENDPOINT_NAME,
        "vector_search_endpoint_name_cache": config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE,
    },
    "input_example": {
        "messages": [{"content": "What is Databricks Model Serving?", "role": "user"}]
    },
    "retriever_config": {
        "vector_search_index": config.VS_INDEX_FULLNAME,
        "vector_search_index_cache": config.VS_INDEX_FULLNAME_CACHE,
        "similarity_threshold": config.SIMILARITY_THRESHOLD,
    },
    "llm_config": {
        "llm_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.\n\nContext: {context}""",
    },
}

try:
    with open('chain/chain_config_cache.yaml', 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')

model_config = mlflow.models.ModelConfig(development_config='chain/chain_config_cache.yaml')

# COMMAND ----------

HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.secrets.get(scope="semantic_cache", key="token")
os.environ['TOKEN'] = TOKEN
os.environ['HOST'] = HOST

# COMMAND ----------

# MAGIC %%writefile chain/chain_cache.py
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from langchain_community.vectorstores import DatabricksVectorSearch
# MAGIC from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.prompts import ChatPromptTemplate
# MAGIC from langchain_community.chat_models import ChatDatabricks
# MAGIC from operator import itemgetter
# MAGIC from datetime import datetime
# MAGIC from uuid import uuid4
# MAGIC import mlflow
# MAGIC import cache
# MAGIC import os
# MAGIC
# MAGIC
# MAGIC # Set up logging
# MAGIC import logging
# MAGIC logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# MAGIC logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
# MAGIC logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
# MAGIC
# MAGIC ## Enable MLflow Tracing
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC model_config = mlflow.models.ModelConfig(development_config="chain_config_cache.yaml")
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
# MAGIC
# MAGIC def retrieve_context(qa):
# MAGIC     return vector_search_as_retriever.invoke(qa["question"])
# MAGIC
# MAGIC # Connect to the Vector Search Index Cache
# MAGIC vs_index_cache = VectorSearchClient(
# MAGIC     workspace_url=os.environ['HOST'],
# MAGIC     personal_access_token=os.environ['TOKEN'],
# MAGIC     disable_notice=True,
# MAGIC     ).get_index(
# MAGIC     endpoint_name=databricks_resources.get("vector_search_endpoint_name_cache"),
# MAGIC     index_name=retriever_config.get("vector_search_index_cache"),
# MAGIC )
# MAGIC
# MAGIC # Enable the RAG Studio Review App and MLFlow to properly display track and display retrieved chunks for evaluation
# MAGIC mlflow.models.set_retriever_schema(primary_key="id", text_column="content", doc_uri="url")
# MAGIC
# MAGIC # Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
# MAGIC def format_context(docs):
# MAGIC     chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
# MAGIC     return "".join(chunk_contents)
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
# MAGIC def call_model(prompt):
# MAGIC     response = model.invoke(prompt)
# MAGIC     store_in_cache(
# MAGIC         question = prompt.dict()['messages'][1]['content'], 
# MAGIC         answer = response.content
# MAGIC     )
# MAGIC     return response
# MAGIC
# MAGIC # Return the string contents of the most recent messages: [{...}] from the user to be used as input question
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]["content"]
# MAGIC
# MAGIC def get_from_cache(question, creator="user", access_level=0):    
# MAGIC     # Check if the question exists in the cache
# MAGIC     qa = {"question": question, "answer": ""}
# MAGIC     
# MAGIC     results = vs_index_cache.similarity_search(
# MAGIC         query_vector=cache.get_embedding(question),
# MAGIC         columns=["id", "question", "answer"],
# MAGIC         num_results=1
# MAGIC     )
# MAGIC     if results and results['result']['row_count'] > 0:
# MAGIC         score = results['result']['data_array'][0][3]  # Get the score
# MAGIC         logging.info(f"Score: {score}")
# MAGIC         try:
# MAGIC             if float(score) >= retriever_config.get("similarity_threshold"): 
# MAGIC                 # Cache hit
# MAGIC                 qa["answer"] = results['result']['data_array'][0][2]
# MAGIC                 record_id = results['result']['data_array'][0][0]  # Assuming 'id' is the first column
# MAGIC                 logging.info("Cache hit: True")
# MAGIC             else:
# MAGIC                 logging.info("Cache hit: False")
# MAGIC         except ValueError:
# MAGIC             logging.info(f"Warning: Invalid score value: {score}")
# MAGIC     return qa
# MAGIC
# MAGIC def store_in_cache(question, answer, creator="user", access_level=0):
# MAGIC     # Prepare the document for upsert
# MAGIC     document = {
# MAGIC         "id": str(uuid4()),
# MAGIC         "creator": creator,
# MAGIC         "question": question,
# MAGIC         "answer": answer,
# MAGIC         "access_level": access_level,
# MAGIC         "created_at": datetime.now().isoformat(),
# MAGIC         "text_vector": cache.get_embedding(question),
# MAGIC     }
# MAGIC     vs_index_cache.upsert([document])
# MAGIC
# MAGIC def route(qa):
# MAGIC     if qa["answer"] != "":
# MAGIC         return (qa["answer"])
# MAGIC     else:
# MAGIC         return rag_chain
# MAGIC
# MAGIC rag_chain = (
# MAGIC     {
# MAGIC         "question": lambda x: x["question"],
# MAGIC         "context": RunnablePassthrough()
# MAGIC         | RunnableLambda(retrieve_context)
# MAGIC         | RunnableLambda(format_context),
# MAGIC     }
# MAGIC     | prompt
# MAGIC     | RunnableLambda(call_model)
# MAGIC )
# MAGIC
# MAGIC # RAG Chain with cache
# MAGIC full_chain = (
# MAGIC     itemgetter("messages")
# MAGIC     | RunnableLambda(extract_user_query_string)
# MAGIC     | RunnableLambda(get_from_cache)
# MAGIC     | RunnableLambda(route)
# MAGIC     | StrOutputParser()
# MAGIC )
# MAGIC
# MAGIC # Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=full_chain)

# COMMAND ----------

# Log the model to MLflow
config_file_path = "config.py"
cache_file_path = "cache.py"
utils_file_path = "utils.py"

with mlflow.start_run(run_name=f"rag_chatbot"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(os.getcwd(), 'chain/chain_cache.py'),  # Chain code file e.g., /path/to/the/chain.py 
        model_config='chain/chain_config_cache.yaml',  # Chain configuration 
        artifact_path="chain",  # Required by MLflow
        input_example=model_config.get("input_example"),  # MLflow will execute the chain before logging & capture it's output schema.
        code_paths = [cache_file_path, config_file_path, utils_file_path],
    )

# Test the chain locally
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(model_config.get("input_example"))

# COMMAND ----------

chain.invoke({'messages': [{'content': "How does Databricks' feature Genie automate feature engineering for machine learning models?", 'role': 'user'}]})

# COMMAND ----------

chain.invoke({'messages': [{'content': "What is the role of Databricks' feature Genie in automating feature engineering for machine learning models?", 'role': 'user'}]})

# COMMAND ----------

# Register to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=config.MODEL_FULLNAME_CACHE)

# COMMAND ----------

# MAGIC %md
# MAGIC Need to put a token (either PAT or Service Principal Token) in the Secret.

# COMMAND ----------

import utils
utils.deploy_model_serving_endpoint(
  spark, 
  config.MODEL_FULLNAME_CACHE,
  config.CATALOG_CACHE,
  config.LOGGING_SCHEMA_CACHE,
  config.ENDPOINT_NAME_CACHE,
  HOST,
  )

# COMMAND ----------

utils.wait_for_model_serving_endpoint_to_be_ready(config.ENDPOINT_NAME_CACHE)

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
utils.send_request_to_endpoint(config.ENDPOINT_NAME_CACHE, data)

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | | | |
