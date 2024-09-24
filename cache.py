import json
import utils
import mlflow
import logging
from uuid import uuid4
from datetime import datetime

mlflow.set_tracking_uri("databricks")
from databricks.vector_search.client import VectorSearchClient

from config import Config
config = Config()


def create_cache():
    vsc = VectorSearchClient(disable_notice=True)
    logging.info("VectorSearchClient initialized successfully")
    
    # Create or wait for the endpoint
    utils.create_or_wait_for_endpoint(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE)
    logging.info(f"Vector search endpoint '{config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE}' is ready")

    # Create or update the main index
    utils.create_or_update_direct_index(
        vsc, 
        config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE, 
        config.VS_INDEX_FULLNAME_CACHE, 
        config.VECTOR_SEARCH_INDEX_SCHEMA_CACHE,
        config.VECTOR_SEARCH_INDEX_CONFIG_CACHE,
    )
    logging.info(f"Main index '{config.VS_INDEX_FULLNAME_CACHE}' created/updated and is ready")
    logging.info("Environment setup completed successfully")


def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def get_embedding(text):
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    response = client.predict(
      endpoint=config.EMBEDDING_MODEL_SERVING_ENDPOINT_NAME,
      inputs={"input": [text]})
    return response.data[0]['embedding']


def warm_cache(batch_size=100):
    # Set up the VectorSearchClient
    vsc = VectorSearchClient(disable_notice=True)
    logging.info("VectorSearchClient initialized successfully")
    
    # Load dataset
    data = load_data(config.CACHE_WARMING_FILE_PATH)
    logging.info(f"Loaded {len(data)} documents from {config.CACHE_WARMING_FILE_PATH}")

    # Use the endpoint directly
    endpoint = vsc.get_endpoint(config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE)
    index = vsc.get_index(
      index_name=config.VS_INDEX_FULLNAME_CACHE, 
      endpoint_name=config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE)
    logging.info(f"Retrieved index '{config.VS_INDEX_FULLNAME_CACHE}' from endpoint '{config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE}'")

    documents = []
    for idx, item in enumerate(data):
        if 'question' in item and 'answer' in item:
            embedding = get_embedding(item['question']) 
            doc = {
                "id": str(idx),
                "creator": "system",
                "question": item["question"],
                "answer": item["answer"],
                "access_level": 0,
                "created_at": datetime.now().isoformat(),
                "text_vector": embedding
            }
            documents.append(doc)
        
        # Upsert when batch size is reached
        if len(documents) >= batch_size:
            try:
                index.upsert(documents)
                print(f"Successfully upserted batch of {len(documents)} documents.")
            except Exception as e:
                print(f"Error upserting batch: {str(e)}")
            documents = []  # Clear the batch

    # Upsert any remaining documents
    if documents:
        try:
            index.upsert(documents)
            print(f"Successfully upserted final batch of {len(documents)} documents.")
        except Exception as e:
            print(f"Error upserting final batch: {str(e)}")

    logging.info("Index details:")
    logging.info(f"  Type: {type(index)}")
    logging.info(f"  Name: {index.name}")
    logging.info(f"  Endpoint name: {index.endpoint_name}")
    logging.info(f"Finished loading documents into the index.")
    logging.info("Cache warming completed successfully")


def evict(strategy='FIFO', max_documents=1000, batch_size=100):
    vsc = VectorSearchClient(disable_notice=True)
    index = vsc.get_index(
        index_name=config.VS_INDEX_FULLNAME_CACHE,
        endpoint_name=config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE
    )
    
    total_docs = get_indexed_row_count(
        vsc,
        config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE,
        config.VS_INDEX_FULLNAME_CACHE
    )
    
    if total_docs <= max_documents:
        logging.info(f"Cache size ({total_docs}) is within limit ({max_documents}). No eviction needed.")
        return
    
    docs_to_remove = total_docs - max_documents
    
    logging.info(f"Evicting {docs_to_remove} documents from cache using {strategy} strategy...")
    
        
    if strategy == 'FIFO':
        _evict_fifo(index, docs_to_remove, batch_size)
    elif strategy == 'LRU':
        _evict_lru(index, docs_to_remove, batch_size)
    else:
        raise ValueError(f"Unknown eviction strategy: {strategy}")
    
    logging.info("Cache eviction completed.")


def _evict_fifo(self, index, docs_to_remove, batch_size):
    while docs_to_remove > 0:
        results = index.similarity_search(
            query_vector=[0] * config.EMBEDDING_DIMENSION,
            columns=["id", "created_at"],
            num_results=min(docs_to_remove, batch_size),
        )
        
        if not results or results['result']['row_count'] == 0:
            break
        
        ids_to_remove = [row[0] for row in results['result']['data_array']]
        index.delete(ids_to_remove)
        
        docs_to_remove -= len(ids_to_remove)
        logging.info(f"Removed {len(ids_to_remove)} documents from cache (FIFO).")

def _evict_lru(self, index, docs_to_remove, batch_size):
    while docs_to_remove > 0:
        results = index.similarity_search(
            query_vector=[0] * config.EMBEDDING_DIMENSION,
            columns=["id", "last_accessed"],
            num_results=min(docs_to_remove, batch_size),
        )
        
        if not results or results['result']['row_count'] == 0:
            break
        
        ids_to_remove = [row[0] for row in results['result']['data_array']]
        index.delete(ids_to_remove)
        
        docs_to_remove -= len(ids_to_remove)
        logging.info(f"Removed {len(ids_to_remove)} documents from cache (LRU).")

def get_indexed_row_count(vsc, endpoint_name, index_name):
    index = vsc.get_index(
        index_name=index_name,
        endpoint_name=endpoint_name
    )
    description = index.describe()
    return description.get('status', {}).get('indexed_row_count', 0)
    
def clear_cache():
    logging.info(f"Cleaning cache on endpoint {config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE}...")
    vsc = VectorSearchClient(disable_notice=True)
    if utils.index_exists(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE, config.VS_INDEX_FULLNAME_CACHE):
        try:
            vsc.delete_index(config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE, config.VS_INDEX_FULLNAME_CACHE)
            logging.info(f"Cache index {config.VS_INDEX_FULLNAME_CACHE} deleted successfully")
        except Exception as e:
            logging.error(f"Error deleting cache index {config.VS_INDEX_FULLNAME_CACHE}: {str(e)}")
    else:
        logging.info(f"Cache index {config.VS_INDEX_FULLNAME_CACHE} does not exist")

