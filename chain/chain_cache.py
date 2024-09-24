from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks
from operator import itemgetter
from datetime import datetime
from uuid import uuid4
import os
import mlflow
from cache import Cache
from config import Config


# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

## Enable MLflow Tracing
mlflow.langchain.autolog()

# Get configuration
config = Config()

vsc = VectorSearchClient(
    workspace_url=os.environ['HOST'],
    personal_access_token=os.environ['TOKEN'],
    disable_notice=True,
)

semantic_cache = Cache(vsc, config)

# Connect to the RAG Vector Search Index
vs_index = vsc.get_index(
    endpoint_name=config.VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=config.VS_INDEX_FULLNAME,
)

# Turn the Vector Search index into a LangChain retriever
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column="content",
    columns=["id", "content", "url"],
).as_retriever(search_kwargs={"k": 3}) # Number of search results that the retriever returns

def retrieve_context(qa):
    return vector_search_as_retriever.invoke(qa["question"])

# Connect to the cache Vector Search Index
vs_index_cache = vsc.get_index(
    endpoint_name=config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE,
    index_name=config.VS_INDEX_FULLNAME_CACHE,
)

# Enable the RAG Studio Review App and MLFlow to properly display track and display retrieved chunks for evaluation
mlflow.models.set_retriever_schema(primary_key="id", text_column="content", doc_uri="url")

# Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
def format_context(docs):
    chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
    return "".join(chunk_contents)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"{config.LLM_PROMPT_TEMPLATE}"),
        ("user", "{question}"),
    ]
)

# Define our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint=config.LLM_MODEL_SERVING_ENDPOINT_NAME,
    extra_params={"temperature": 0.01, "max_tokens": 500}
)

# Call the foundation model
def call_model(prompt):
    response = model.invoke(prompt)
    semantic_cache.store_in_cache(
        question = prompt.dict()['messages'][1]['content'], 
        answer = response.content
    )
    return response

# Return the string contents of the most recent messages: [{...}] from the user to be used as input question
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

def router(qa):
    if qa["answer"] != "":
        return (qa["answer"])
    else:
        return rag_chain

# RAG chain
rag_chain = (
    {
        "question": lambda x: x["question"],
        "context": RunnablePassthrough()
        | RunnableLambda(retrieve_context)
        | RunnableLambda(format_context),
    }
    | prompt
    | RunnableLambda(call_model)
)

# Full chain with cache
full_chain = (
    itemgetter("messages")
    | RunnableLambda(extract_user_query_string)
    | RunnableLambda(semantic_cache.get_from_cache)
    | RunnableLambda(router)
    | StrOutputParser()
)

# Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=full_chain)
