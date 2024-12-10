from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.data_loaders.docling_loader import DoclingHTMLLoader


class DocumentProcessor:
    def __init__(self, index_name: str, api_key=os.getenv("PINECONE_API_KEY"),
                 dimension=1536, cloud='aws', region='us-east-1', metric='cosine'):
        """
        Initialize Pinecone vectorstore

        :param index_name: Name of the Pinecone index.
        :param api_key: API key for Pinecone.
        :param dimension: Embedding dimension.
        :param cloud: Cloud provider for the index (e.g., 'aws').
        :param region: Region for the index (e.g., 'us-east-1').
        :param metric: Distance metric for the index.
        :return: LangChain Pinecone vectorstore object.
        """

        pc = Pinecone(api_key=api_key)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)

            )
            print(f"Index '{index_name}' created.")
        else:
            print(f"Index '{index_name}' loaded.")

        pinecone_index = pc.Index(index_name)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
        self.vectorstore = vectorstore


    def load_documents(self, sitemap_entries: list) -> None:
        """
        Load and index documents into the Pinecone vector database.

        :param sitemap_entries: List of URLs to load and index.
        """

        # Load documents using your custom loader
        loader = DoclingHTMLLoader(sitemap_entry=sitemap_entries)
        documents = loader.load()

        # Initialize MarkdownHeaderTextSplitter
        markdown_separators = [
            "\n#{1,6} ",
            "```\n",
            "\n\\*\\*\\*+\n",
            "\n---+\n",
            "\n___+\n",
            "\n\n",
            "\n",
            " ",
            "",
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=markdown_separators,
        )

        # Split the documents into chunks
        processed_documents = text_splitter.split_documents(documents)

        # for i, doc in enumerate(processed_documents):
        #     print(f"***** {i + 1} *****\n"
        #           f"{doc.metadata}\n"
        #           f"{doc.page_content}\n")

        self.vectorstore.add_documents(documents=processed_documents)
        print(f"Loaded {len(processed_documents)} chunks into the Pinecone vectorstore.")


    def get_all_documents(self, namespace: str = None):
        """
        Retrieve all documents in a really dumb way, use only for testing.

        :param namespace: (Optional) Namespace for filtering documents.
        :return: List of documents with their metadata.
        """
        try:
            dummy_query = ""
            results = self.vectorstore.similarity_search(query=dummy_query, k=10000, namespace=namespace)
            documents = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
            print(f"Found {len(documents)} documents.")
            return documents
        except Exception as e:
            print(f"Failed to retrieve documents: {e}")
            return []
