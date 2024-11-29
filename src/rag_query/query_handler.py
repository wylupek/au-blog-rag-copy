from collections import Counter
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loaders.docling_loader import DoclingHTMLLoader

class QueryHandler:
    def __init__(self, vector_db: VectorStore):
        """
        Initializes the Query Handler.

        Parameters:
            vector_db_path (str): Path to load the vector database.
        """
        self.vector_db = vector_db
        
        # Check if the database is initialized and get the document count
        print(f"ChromaDB initialized successfully. Document count: {self.vector_db._collection.count()}")
        
    def get_answer(self, question):
        # return documents based on query
        results = self.search_documents(user_query=question) 

        # Get unique URLs sorted by frequency
        sources = self.get_sorted_urls_by_frequency(results)

        # Generate summary and query answer for each article
        return self.analyze_summaries(sources, question)

    def search_documents(self, user_query: str) -> List[Document]:
        """Retrieve and return top-k most relevant documents."""
        # Generate alternative queries
        template = ("You are an AI language model assistant. Your task is to generate five different versions of the "
                    "given user question to retrieve relevant documents from a vector database. By generating "
                    "multiple perspectives on the user question, your goal is to help the user overcome some of the "
                    "limitations of the distance-based similarity search. Provide these alternative questions "
                    "separated by newlines. Original question: {question}")
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_perspectives
            | ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        def get_unique_union(documents: list[list]):
            """Unique union of retrieved docs."""
            flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
            unique_docs = list(set(flattened_docs))
            return [loads(doc) for doc in unique_docs]

        # Retrieve documents
        retriever = self.vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.2}
            # search_type="similarity",
            # search_kwargs={"k": 4}
        )
        retrieval_chain = generate_queries | retriever.map() | get_unique_union
        docs = retrieval_chain.invoke({"question": user_query})

        # Debugging
        print(f"DEBUG: Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"Document {i + 1}:")
            print(f"  Page Content: {doc.page_content}...")  # Print first 200 characters
            print(f"  Metadata: {doc.metadata}")

        # Sort by relevance score (if available)
        docs.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)

        # Return top-k matches
        return docs


    def analyze_summaries(self, urls: [str], question: str) -> List[dict]:
        """
        Uses the LLM to analyze how each document summary can answer the user's query.
        This version parallelizes the API calls for efficiency.
        """

        # Template for the LLM
        template = """
        You are an AI language model tasked with generating concise responses to user queries based on provided article content. 

        Your goal is to:
        1. Generate a concise summary of the article.
        2. Address the user's question strictly based on the article content in simple, non-technical language.
        3. Format the output in two clear sections separated by two new lines: 
            - A summary of the article.
            - A response to the user's query.

        Additional Notes:
        - Keep the response brief (maximum 150 words).
        - Ensure the output is easily understandable for non-technical users.
        - If the article does not answer the question, don't generate summary and return an empty response.
        - Do not include indentations or extra formatting.

        <question>{query}<\question>
        <article>{context}<\article>
        """

        llm = ChatOpenAI(temperature=0.4, model_name="gpt-4o-mini")

        def process_url(url):
            """
            Process a single URL: Load the document, send it to the LLM, and return the analysis.
            """
            try:
                # Load the document using the custom loader
                loader = DoclingHTMLLoader(file_path=url)
                document = loader.load()

                # Format the prompt with the query and document content
                prompt = ChatPromptTemplate.from_template(template)
                messages = prompt.format_messages(query=question, context=document[0].page_content)

                # Call the LLM
                result = llm(messages)

                # Return the processed result
                return {
                    "url": url,
                    "analysis": result.content
                }
            except Exception as e:
                # Handle and log any errors
                return {
                    "url": url,
                    "analysis": f"Error processing URL {url}: {e}"
                }

        # Use ThreadPoolExecutor to parallelize URL processing
        analyses = []
        with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust `max_workers` based on your needs
            future_to_url = {executor.submit(process_url, url): url for url in urls}

            for future in as_completed(future_to_url):
                try:
                    analyses.append(future.result())
                except Exception as e:
                    analyses.append({
                        "url": future_to_url[future],
                        "analysis": f"Error in processing: {e}"
                    })

        return analyses

    def display_results(self, results: List[dict]) -> None:
        """
        Nicely prints the summaries with their corresponding URLs.

        Parameters:
            results (List[dict]): List of dictionaries containing 'url' and 'analysis'.
        """

        print("Analysis Results:\n" + "=" * 50)
        for i, result in enumerate(results, 1):
            print(f"URL: {result['url']}")
            print(f"\n{result['analysis']}")
            print("-" * 50)

    def get_sorted_urls_by_frequency(self, documents: List[Document]) -> List[str]:
        """
        Extracts and sorts a list of unique URLs from document metadata by their frequency.

        Parameters:
            documents (list): List of Document objects, each with a metadata dictionary.

        Returns:
            list: Sorted list of unique URLs (most frequent first).
        """
        from collections import Counter

        # Collect all sources
        sources = [doc.metadata.get("source") for doc in documents if "source" in doc.metadata]

        # Count occurrences of each source
        source_counts = Counter(sources)

        # Sort by count in descending order and extract only the URLs
        sorted_urls = [url for url, count in source_counts.most_common()]

        return sorted_urls
