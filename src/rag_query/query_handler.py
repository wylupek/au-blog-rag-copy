from typing import List
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data_loaders.docling_loader import DoclingHTMLLoader
from langchain_pinecone import PineconeVectorStore
from src.data_loaders.sitemap_entry import SitemapEntry
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone



class QueryHandler:
    def __init__(self, vectorstore: PineconeVectorStore, pinecone_index: Pinecone.Index):
        self.vectorstore = vectorstore
        self.pinecone_index = pinecone_index
        print(f"Pinecone initialized successfully.")


    def get_answer(self, question):
        # return documents based on query
        results = self.search_documents(user_query=question) 

        # Get unique URLs sorted by frequency
        entries = self.get_entries_with_score(results, question)

        # Generate summary and query answer for each article
        return sorted(self.analyze_summaries(entries, question), key=lambda x: x["score"], reverse=True)


    def search_documents(self, user_query: str) -> List[Document]:
        """Retrieve and return top-k most relevant documents."""
        template = """
        You are an AI language model assistant. Your task is to generate four different versions of the given
        user question to retrieve relevant documents from a vector database. By generating multiple perspectives
        on the user question, your goal is to help the user overcome limitations of the distance-based 
        similarity search. Provide only these alternative questions, each one in a new line without numbering. 
        Original question: {question}
        """
        template = " ".join(line.strip() for line in template.strip().splitlines())
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_queries_pipeline = (
            prompt_perspectives
            | ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
            | StrOutputParser()
            | (lambda x: [user_query] + [query.strip() for query in x.split("\n") if query.strip()])
        )

        # generated_queries = generate_queries_pipeline.invoke({"question": user_query})
        # print("Generated queries:")
        # for i, generated_query in enumerate(generated_queries):
        #     print(f"{i + 1}. {generated_query}")

        def get_unique_documents(docs: list[list]):
            """Unique union of retrieved docs."""
            flattened_docs = [dumps(doc) for sublist in docs for doc in sublist]
            unique_docs = list(set(flattened_docs))
            return [loads(doc) for doc in unique_docs]

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )

        retrieval_chain = (
                generate_queries_pipeline
                | retriever.map()
                | get_unique_documents
        )
        documents = retrieval_chain.invoke({"question": user_query})

        print(f"Retrieved {len(documents)} documents:")
        for i, document in enumerate(documents):
            print(f"*** Document {i + 1} ***")
            print(f"Source: {document.metadata['source']}")
            print(f"Page Content: {document.page_content[:200]}...\n\n")  # Print first 200 characters

        return documents


    def analyze_summaries(self, sitemap_entries: [SitemapEntry], question: str) -> List[dict]:
        """
        Uses the LLM to analyze how each document summary can answer the user's query.
        This version parallelizes the API calls for efficiency.
        """

        template = """
        You are an AI language model tasked with generating concise and precise responses to user queries based on the provided article content.

        Your response should include three distinct sections, separated by two blank lines:

        1. A single-word answer ('True' or 'False') indicating whether the article might be relevant to the user's query.
        2. A concise summary of the article in plain, non-technical language.
        3. A direct, clear response to the user's question, strictly based on the article content, written in simple terms.

        Guidelines:
        - Limit the entire response to a maximum of 150 words.
        - Avoid extra formatting, unnecessary details, or information outside the article's scope.
        - Write in a clear, user-friendly tone suitable for non-technical readers.

        <question>{query}</question>
        <article>{context}</article>
        """

        llm = ChatOpenAI(temperature=0.4, model_name="gpt-4o-mini")

        def process_url(sitemap_entry: SitemapEntry):
            """
            Process a single URL: Load the document, send it to the LLM, and return the analysis.
            """
            try:
                # Load the document using the custom loader
                loader = DoclingHTMLLoader(sitemap_entry)
                document = loader.load()

                # Format the prompt with the query and document content
                prompt = ChatPromptTemplate.from_template(template)
                messages = prompt.format_messages(query=question, context=document[0].page_content)

                # Call the LLM
                result = llm(messages)

                # Return the processed result
                return {
                    "url": sitemap_entry.url,
                    "analysis": result.content,
                    "score": sitemap_entry.score
                }
            except Exception as e:
                # Handle and log any errors
                return {
                    "url": sitemap_entry.url,
                    "analysis": f"Error processing URL {sitemap_entry.url}: {e}",
                    "score": sitemap_entry.score
                }

        # Use ThreadPoolExecutor to parallelize URL processing
        analyses = []
        with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust `max_workers` based on your needs
            future_to_url = {executor.submit(process_url, sitemap_entry): sitemap_entry for sitemap_entry in sitemap_entries}

            for future in as_completed(future_to_url):
                try:
                    analyses.append(future.result())
                except Exception as e:
                    analyses.append({
                        "url": future_to_url[future],
                        "analysis": f"Error in processing: {e}",
                        "score": 0
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


    def get_entries_with_score(self, documents: List[Document], query: str, threshold=0.35) -> List[SitemapEntry]:
        """
        Calculate score and get list of unique sources with best score from given source based on Document object.
        Sort unique sources by score. Apply threshold on score.
        :param documents: list of Document objects
        :param query: user query to calculate score with
        :param threshold: threshold for score
        :return: list of unique SitemapEntry objects
        """
        embedder = self.vectorstore.embeddings
        query_vector = embedder.embed_query(query)

        unique_sources = {}
        for document in documents:
            doc_vector = embedder.embed_query(document.page_content)
            score = cosine_similarity([query_vector], [doc_vector])[0][0]
            if document.metadata["source"] not in unique_sources or score > unique_sources[document.metadata["source"]]:
                unique_sources[document.metadata["source"]] = score
        unique_sources = sorted(unique_sources.items(), key=lambda x: x[1], reverse=True)

        for i, source in enumerate(unique_sources):
            print(f"{i + 1}. {source[1]}, {source[0]}")

        return [SitemapEntry(url=source[0], lastmod=None, score=source[1]) for source in unique_sources if source[1] > threshold]

