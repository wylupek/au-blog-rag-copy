from typing import List
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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


    def get_answer(self, question, threshold=0.35, top_k=10, filter_false=False,
                   query_generation_model="gpt-4o-mini", analysis_model="gpt-4o-mini"):
        # return documents based on query
        results = self.search_documents(user_query=question, top_k=top_k, query_generation_model=query_generation_model)

        # Get unique URLs sorted by frequency
        entries = self.get_entries_with_score(results, threshold=threshold)

        # Generate summary and analysis for each article
        return self.analyze_summaries(entries, question, filter_false=filter_false, analysis_model=analysis_model)


    def search_documents(self, user_query: str, top_k=10,
                         query_generation_model="gpt-4o-mini") -> List[Document]:
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
            | ChatOpenAI(temperature=0, model_name=query_generation_model)
            | StrOutputParser()
            | (lambda x: [user_query] + [q.strip() for q in x.split("\n") if q.strip()])
        )

        generated_queries = generate_queries_pipeline.invoke({"question": user_query})
        # print("Generated queries:")
        # for i, generated_query in enumerate(generated_queries):
        #     print(f"{i + 1}. {generated_query}")

        # Perform searches for all query embeddings and merge results
        results = []
        query_embeddings = [self.vectorstore.embeddings.embed_query(q) for q in generated_queries]
        for query in query_embeddings:
            response = self.pinecone_index.query(
                vector=query,
                top_k=top_k,
                include_metadata=True,
                include_values=True,
            )
            results.extend(response.matches)

        # Remove duplicate documents by ID
        seen_ids = set()
        unique_results = []
        for match in results:
            if match.id not in seen_ids:
                seen_ids.add(match.id)
                unique_results.append(match)

        # Convert Pinecone matches to LangChain Documents
        documents = [
            Document(
                page_content=match.metadata.get("text", ""),
                metadata={**match.metadata,
                          "score": cosine_similarity([query_embeddings[0]], [match.values])[0][0]},
            )
            for match in unique_results
        ]

        documents = sorted(documents, key=lambda x: x.metadata["score"], reverse=True)
        print(f"Retrieved {len(documents)} documents\n")
        for i, document in enumerate(documents):
            print(f"*** Document {i + 1} ***\n"
                  f"Source: {document.metadata['source']}\n"
                  f"Score: {document.metadata['score']}\n"
                  f"Page Content: {document.page_content[:200]}...\n\n")

        return documents


    @staticmethod
    def analyze_summaries(sitemap_entries: [SitemapEntry], question: str, max_workers=5,
                          filter_false=False, analysis_model="gpt-4o-mini") -> List[dict]:
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

        llm = ChatOpenAI(temperature=0.4, model_name=analysis_model)

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
                result = llm.invoke(messages)

                result_split = [x.strip() for x in result.content.split("\n") if x.strip()]
                if len(result_split) == 3:
                    decision, summary, response = result_split
                else:
                    decision = ""
                    summary = ""
                    response = result

                # Return the processed result
                return {
                    "url": sitemap_entry.url,
                    "score": sitemap_entry.score,
                    "decision": decision,
                    "summary": summary,
                    "response": response,
                }
            except Exception as e:
                return {
                    "url": sitemap_entry.url,
                    "score": sitemap_entry.score,
                    "decision": True,
                    "summary": "",
                    "response": f"Error processing URL {sitemap_entry.url}: {e}",
                }

        # Use ThreadPoolExecutor to parallelize URL processing
        analyses = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(process_url, sitemap_entry): sitemap_entry for sitemap_entry in sitemap_entries}

            for future in as_completed(future_to_url):
                try:
                    analyses.append(future.result())
                except Exception as e:
                    analyses.append({
                        "url": future_to_url[future],
                        "score": 0,
                        "decision": True,
                        "summary": "",
                        "response": f"Error in processing: {e}",
                    })

        analyses = sorted(analyses, key=lambda x: x["score"], reverse=True)
        if filter_false:
            return [analysis for analysis in analyses if analysis['decision'] != "False"]

        return ([analysis for analysis in analyses if analysis['decision'] == "True"] +
                [analysis for analysis in analyses if analysis['decision'] != "True"])


    @staticmethod
    def display_results(results: List[dict]) -> None:
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


    @staticmethod
    def get_entries_with_score(documents: List[Document], threshold=0.35) -> List[SitemapEntry]:
        """
        Sort unique sources by score and apply threshold on it.
        :param documents: list of Document objects
        :param threshold: threshold for score
        :return: list of unique SitemapEntry objects
        """

        unique_sources = {}
        for document in documents:
            if (document.metadata["source"] not in unique_sources
                    or document.metadata["score"] > unique_sources[document.metadata["source"]]):
                unique_sources[document.metadata["source"]] = document.metadata["score"]

        for i, (source, score) in enumerate(unique_sources.items()):
            print(f"{i + 1}. {score:.3f} - {source}")

        return [SitemapEntry(url=source, lastmod=None, score=score)
                for source, score in unique_sources.items() if score > threshold]
