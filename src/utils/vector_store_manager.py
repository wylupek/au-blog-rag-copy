import os

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

class VectorStoreManager:
    def __init__(self, index_name: str, api_key=os.getenv("PINECONE_API_KEY"),
                 dimension=1536, cloud='aws', region='us-east-1', metric='cosine'):
        """
        Handles vector_store initialization, metadata tracking, and index operations.
        """

        # Initialize Pinecone
        self.pinecone_client = Pinecone(api_key=api_key)

        # Create or load index
        if index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            print(f"Index '{index_name}' created.")
        else:
            print(f"Index '{index_name}' loaded.")

        self.pinecone_index = self.pinecone_client.Index(index_name)

        # Setup embedding model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
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
        if len(ids) != 0:
            self.pinecone_index.delete(ids=ids)
            self.total_count -= len(ids)
