import os

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

from src.utils.configuration import LoaderConfiguration, RAGConfiguration


class VectorStoreManager:
    # Cache instances keyed by index_name to avoid reinitializing connections unnecessarily
    _instances = {}

    def __new__(cls, index_name: str, *args, **kwargs):
        # If an instance already exists for the given index_name, return it.
        if index_name in cls._instances:
            return cls._instances[index_name]
        instance = super().__new__(cls)
        cls._instances[index_name] = instance
        return instance


    def __init__(self, index_name: str, configuration: LoaderConfiguration | RAGConfiguration,
                 api_key=os.getenv("PINECONE_API_KEY"), cloud='aws', region='us-east-1', metric='cosine'):
        """
        Handles vector_store initialization, metadata tracking, and index operations.
        """
        print(f"Initializing VectorStoreManager for index '{index_name}'.")
        # Prevent reinitialization if already done
        if hasattr(self, "_initialized") and self._initialized:
            print(f"VectorStoreManager for index '{index_name}' already initialized.")
            return

        self.index_name = index_name
        self.configuration = configuration

        self.api_key = api_key
        self.cloud = cloud
        self.region = region
        self.metric = metric

        self._initialize_connections()
        self._initialized = True

    def _initialize_connections(self):
        # Initialize embedder
        model_provider, model_name = self.configuration.embedding_model.split('/')
        if model_provider == "openai":
            self.embeddings = OpenAIEmbeddings(model=model_name)
            self.dimension = len(self.embeddings.embed_query(""))
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.configuration.embedding_model)
            self.dimension = len(self.embeddings.embed_query(""))


        # Initialize Pinecone
        self.pinecone_client = Pinecone(api_key=self.api_key)

        # Create or load index
        if self.index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region)
            )
            print(f"Index '{self.index_name}' created.")
        else:
            print(f"Index '{self.index_name}' loaded.")

        self.pinecone_index = self.pinecone_client.Index(self.index_name)
        self.vector_store = PineconeVectorStore(index=self.pinecone_index, embedding=self.embeddings)

        # Track vector count
        self.total_count = self._get_vector_count()
        print(f"Total vectors: {self.total_count}")

    def _get_vector_count(self):
        """Fetches the total vector count from the index."""
        try:
            index_stats = self.pinecone_index.describe_index_stats()
            return sum(items['vector_count'] for items in index_stats['namespaces'].values())
        except Exception:
            raise RuntimeError("Failed to retrieve index stats.")

    def delete_by_ids(self, ids: list) -> None:
        if ids:
            self.pinecone_index.delete(ids=ids)
            self.total_count -= len(ids)

    def get_all_documents(self, namespace: str = None):
        """
        Retrieve all documents

        :param namespace: (Optional) Namespace for filtering documents.
        :return: List of documents with their metadata.
        """
        try:
            dummy_query = ""
            results = self.vector_store.similarity_search(query=dummy_query, k=self.total_count, namespace=namespace)
            documents = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
            print(f"Found {len(documents)} documents.")
            return documents
        except Exception as e:
            print(f"Failed to retrieve documents: {e}")
            return []