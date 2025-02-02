from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
from src.loader_graph.sitemap_entry import SitemapEntry
from src.loader_graph.docling_loader import DoclingHTMLLoader


class DocumentProcessor:
    def __init__(self, index_name: str, api_key=os.getenv("PINECONE_API_KEY"),
                 dimension=1536, cloud='aws', region='us-east-1', metric='cosine'):
        """
        Initialize Pinecone vectorstore and database

        :param index_name: Name of the Pinecone index.
        :param api_key: API key for Pinecone.
        :param dimension: Embedding dimension.
        :param cloud: Cloud provider for the index.
        :param region: Region for the index.
        :param metric: Distance metric for the index.
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

        self.pinecone_index = pc.Index(index_name)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = PineconeVectorStore(index=self.pinecone_index, embedding=embeddings)

        index_stats = self.pinecone_index.describe_index_stats()
        total_count = 0
        for key, items in index_stats['namespaces'].items():
            total_count += items['vector_count']
        self.total_count = total_count
        print(f"Total vectors: {total_count}")


    def _load_entries(self, sitemap_entries: List[SitemapEntry]) -> None:
        """
        Scrape and split url's from sitemap_entries.
        Add embedded vectors with metadata to Pinecone database.

        :param sitemap_entries: List of SitemapEntry to load.
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
            add_start_index=True,   # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=markdown_separators,
        )

        processed_documents = text_splitter.split_documents(documents)

        # for i, doc in enumerate(processed_documents):
        #     print(f"***** {i + 1} *****\n"
        #           f"{doc.metadata}\n"
        #           f"{doc.page_content}\n")

        self.vectorstore.add_documents(documents=processed_documents)
        self.total_count += len(processed_documents)

        print(f"Loaded {len(processed_documents)} vectors into database.")
        print(f"Total vector count: {self.total_count}")


    def update_database(self, sitemap_entries: List[SitemapEntry]) -> None:
        """
        Main function to add data to Pinecone database.
        Load new or updated vectors with metadata. Delete outdated vectors.

        :param sitemap_entries: List of SitemapEntry to load.
        """

        metadata_list = self.get_all_metadata()
        db_entries = set([SitemapEntry(url=metadata['source'], lastmod=datetime.fromisoformat(metadata['lastmod']))
                          for metadata in metadata_list])

        new_entries = []
        delete_ids = []
        for entry in sitemap_entries:
            vector_flag = 0
            for db_entry in db_entries:
                if entry.url == db_entry.url:
                    if entry.lastmod <= db_entry.lastmod:
                        vector_flag = 1
                        break
                    vector_flag = 2
                    break

            # New entry - just add it
            if vector_flag == 0:
                print(f"<Adding> {entry.url}")
                new_entries.append(entry)

            # Already added entry - just skip it
            elif vector_flag == 1:
                print(f"<Skipping> {entry.url}")
                pass

            # Updated entry - delete old vectors and add new
            elif vector_flag == 2:
                print(f"<Updating> {entry.url}")
                for metadata in metadata_list:
                    if metadata["source"] == entry.url:
                        delete_ids.append(metadata["id"])
                new_entries.append(entry)

        print(f"Ids to delete [{len(delete_ids)}]: {delete_ids}")

        self.delete_by_ids(delete_ids)
        self._load_entries(sitemap_entries=new_entries)


    def get_all_metadata(self, namespace=None):
        try:
            k = (1 if self.total_count == 0 else self.total_count)
            dummy_query = ""
            results = self.vectorstore.similarity_search(query=dummy_query, k=k, namespace=namespace)
            return [{"id": doc.id, "source": doc.metadata['source'], "lastmod": doc.metadata['lastmod']} for doc in results]
        except Exception as e:
            print(f"Failed to retrieve documents: {e}")
            return []


    def delete_by_ids(self, ids: list) -> None:
        if len(ids) != 0:
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
            results = self.vectorstore.similarity_search(query=dummy_query, k=self.total_count, namespace=namespace)
            documents = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
            print(f"Found {len(documents)} documents.")
            return documents
        except Exception as e:
            print(f"Failed to retrieve documents: {e}")
            return []
