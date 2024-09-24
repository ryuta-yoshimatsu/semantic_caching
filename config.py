class Config:
    def __init__(self):

        self.LLM_MODEL_SERVING_ENDPOINT_NAME = "databricks-dbrx-instruct"
        self.EMBEDDING_MODEL_SERVING_ENDPOINT_NAME = "databricks-gte-large-en"
        self.DATABRICKS_SITEMAP_URL = "https://docs.databricks.com/en/doc-sitemap.xml"

        # For vector search index
        self.CATALOG = "semantic_cache_solacc"
        self.SCHEMA = "chatbot_rag"
        self.SOURCE_TABLE_FULLNAME = f"{self.CATALOG}.{self.SCHEMA}.databricks_documentation"
        self.EVALUATION_TABLE_FULLNAME = f"{self.CATALOG}.{self.SCHEMA}.eval_databricks_documentation"
        self.VECTOR_SEARCH_ENDPOINT_NAME = "one-env-shared-endpoint-12"
        self.VS_INDEX_FULLNAME = f"{self.CATALOG}.{self.SCHEMA}.databricks_documentation_vs_index"
        self.MODEL_FULLNAME = f"{self.CATALOG}.{self.SCHEMA}.standard_rag_chatbot"
        self.ENDPOINT_NAME = "standard_rag_chatbot"
        self.LOGGING_SCHEMA = f"logging"

        # For semantic cache
        self.CATALOG_CACHE = "semantic_cache_solacc"
        self.SCHEMA_CACHE = "chatbot_cache"
        self.SOURCE_TABLE_FULLNAME_CACHE = f"{self.CATALOG}.{self.SCHEMA}.cache"
        self.VECTOR_SEARCH_ENDPOINT_NAME_CACHE = "one-env-shared-endpoint-12"
        self.VS_INDEX_FULLNAME_CACHE = f"{self.CATALOG}.{self.SCHEMA}.cache_vs_index"
        self.VS_METRICS_INDEX_FULLNAME_CACHE = f"{self.CATALOG}.{self.SCHEMA}.metrics"
        self.MODEL_FULLNAME_CACHE = f"{self.CATALOG}.{self.SCHEMA}.rag_chatbot_with_cache"
        self.ENDPOINT_NAME_CACHE = "rag_chatbot_with_cache"
        self.LOGGING_SCHEMA_CACHE = f"logging"
        self.CACHE_WARMING_FILE_PATH="data/synthetic_qa.txt"
        self.INDEX_NAME = "cache_index"
        self.SIMILARITY_THRESHOLD = 0.01

        self.VECTOR_SEARCH_INDEX_SCHEMA_CACHE = {
            "id": "string",
            "creator": "string",
            "question": "string",
            "answer": "string",
            "access_level": "int",
            "created_at": "timestamp",
            "text_vector": "array<float>"
        }

        self.EMBEDDING_DIMENSION = 1024
        self.VECTOR_SEARCH_INDEX_CONFIG_CACHE = {
            "primary_key": "id",
            "embedding_dimension": self.EMBEDDING_DIMENSION,
            "embedding_vector_column": "text_vector"
        }
