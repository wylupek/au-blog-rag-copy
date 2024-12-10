from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from data_loaders.docling_loader import DoclingHTMLLoader

class DocumentProcessor:
    def __init__(self, vector_db: VectorStore):
        """
        Initializes the Document Processor.

        Parameters:
            vector_db (str): Path to store/load the vector database.
        """

        self.vector_db = vector_db
                
        # # Delete all collections
        # for collection in self.vector_db._client.list_collections():
        #     self.vector_db._client.delete_collection(collection.name)
        


    # def load_and_index_documents(self, urls: List[str]) -> None:
    #     loader = DoclingHTMLLoader(file_path=urls)
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1000,
    #         chunk_overlap=100
    #     )
    
    #     docs = loader.load()
    #     splits = text_splitter.split_documents(docs)
        
    #     texts = [doc.page_content for doc in splits]
    #     metadatas = [doc.metadata for doc in splits]
    
    #     # Add summaries to the vector store
    #     self.vector_db.add_texts(
    #         texts=texts,
    #         metadatas=metadatas
    #     )

    def load_and_index_documents(self, urls: List[str]) -> None:
        # Load documents using your custom loader
        loader = DoclingHTMLLoader(file_path=urls)
        docs = loader.load()
        
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
            chunk_size=1000,  # The maximum number of characters in a chunk: we selected this value arbitrarily
            chunk_overlap=100,  # The number of characters to overlap between chunks
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=markdown_separators,
        )

        # Split the documents
        splits = []
        docs_processed = []
        for doc in docs:
            docs_processed += text_splitter.split_documents([doc])

        # for i, doc in enumerate(docs_processed):
        #     print(f"***** {i + 1} *****\n"
        #           f"{doc.metadata}\n"
        #           f"{doc.page_content}\n")


        # Extract texts and metadata for storage in the vector database
        texts = [doc.page_content for doc in docs_processed]
        metadatas = [doc.metadata for doc in docs_processed]

        # Add the splits to the vector store
        self.vector_db.add_texts(
            texts=texts,
            metadatas=metadatas
        )

    def load_documents(self, docs: List[Document]) -> None:
        self.vector_db.add_texts(
            texts=[doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs]
        )

    def test_db(self) -> None:
        import json

        # Get all collections
        collections = self.vector_db._client.list_collections()

        # Iterate over each collection and print documents with metadata
        for collection in collections:
            print(f"\nCollection Name: {collection.name}")
            # Fetch the collection object
            col = self.vector_db._client.get_collection(collection.name)
    
            # Retrieve documents and metadata
            docs = col.get(include=["metadatas", "documents"])  # Adjust 'include' as per your structure

            # Iterate over each document and print it nicely formatted
            for i, (doc, metadata) in enumerate(zip(docs["documents"], docs["metadatas"])):
                print(f"\nDocument {i+1}")
                print("Content:", doc)
                print("Metadata:", json.dumps(metadata, indent=4))  # Nicely formatted metadata

    def get_docs(self):
        collections = self.vector_db._client.list_collections()

        all_documents = []
        for collection in collections:
            print(f"\nCollection Name: {collection.name}")
            col = self.vector_db._client.get_collection(collection.name)

            docs = col.get(include=["metadatas", "documents"])
            all_documents.append({collection.name: docs})
        return all_documents
