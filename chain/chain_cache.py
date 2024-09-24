from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks
from operator import itemgetter
from datetime import datetime
from uuid import uuid4
import mlflow
import cache
import os


# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

## Enable MLflow Tracing
mlflow.langchain.autolog()

model_config = mlflow.models.ModelConfig(development_config="chain_config_cache.yaml")

databricks_resources = model_config.get("databricks_resources")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")

# Connect to the Vector Search Index
vs_index = VectorSearchClient(
    workspace_url=os.environ['HOST'],
    personal_access_token=os.environ['TOKEN'],
    disable_notice=True,
    ).get_index(
    endpoint_name=databricks_resources.get("vector_search_endpoint_name"),
    index_name=retriever_config.get("vector_search_index"),
)
    
# Turn the Vector Search index into a LangChain retriever
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column="content",
    columns=["id", "content", "url"],
).as_retriever(search_kwargs={"k": 3}) # Number of search results that the retriever returns

def retrieve_context(qa):
    return vector_search_as_retriever.invoke(qa["question"])

# Connect to the Vector Search Index Cache
vs_index_cache = VectorSearchClient(
    workspace_url=os.environ['HOST'],
    personal_access_token=os.environ['TOKEN'],
    disable_notice=True,
    ).get_index(
    endpoint_name=databricks_resources.get("vector_search_endpoint_name_cache"),
    index_name=retriever_config.get("vector_search_index_cache"),
)

# Enable the RAG Studio Review App and MLFlow to properly display track and display retrieved chunks for evaluation
mlflow.models.set_retriever_schema(primary_key="id", text_column="content", doc_uri="url")

# Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
def format_context(docs):
    chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
    return "".join(chunk_contents)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"{llm_config.get('llm_prompt_template')}"),
        ("user", "{question}"),
    ]
)

# Our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint=databricks_resources.get("llm_model_serving_endpoint_name"),
    extra_params={"temperature": 0.01, "max_tokens": 500}
)

def call_model(prompt):
    response = model.invoke(prompt)
    store_in_cache(
        question = prompt.dict()['messages'][1]['content'], 
        answer = response.content
    )
    return response

# Return the string contents of the most recent messages: [{...}] from the user to be used as input question
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

def get_from_cache(question, creator="user", access_level=0):    
    # Check if the question exists in the cache
    qa = {"question": question, "answer": ""}
    
    results = vs_index_cache.similarity_search(
        query_vector=cache.get_embedding(question),
        columns=["id", "question", "answer"],
        num_results=1
    )
    if results and results['result']['row_count'] > 0:
        score = results['result']['data_array'][0][3]  # Get the score
        logging.info(f"Score: {score}")
        try:
            if float(score) >= retriever_config.get("similarity_threshold"): 
                # Cache hit
                qa["answer"] = results['result']['data_array'][0][2]
                record_id = results['result']['data_array'][0][0]  # Assuming 'id' is the first column
                logging.info("Cache hit: True")
            else:
                logging.info("Cache hit: False")
        except ValueError:
            logging.info(f"Warning: Invalid score value: {score}")
    return qa

def store_in_cache(question, answer, creator="user", access_level=0):
    # Prepare the document for upsert
    document = {
        "id": str(uuid4()),
        "creator": creator,
        "question": question,
        "answer": answer,
        "access_level": access_level,
        "created_at": datetime.now().isoformat(),
        "text_vector": cache.get_embedding(question),
    }
    vs_index_cache.upsert([document])

def route(qa):
    if qa["answer"] != "":
        return (qa["answer"])
    else:
        return rag_chain

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

# RAG Chain with cache
full_chain = (
    itemgetter("messages")
    | RunnableLambda(extract_user_query_string)
    | RunnableLambda(get_from_cache)
    | RunnableLambda(route)
    | StrOutputParser()
)

# Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=full_chain)
