from pinecone import Pinecone
from typing import List
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loaders.sitemap_entry import SitemapEntry



class QueryHandler:
    """
    A class for handling user queries, retrieving documents,
    generating summaries and analyses, and filtering results.
    """
    def __init__(self, vectorstore: PineconeVectorStore, pinecone_index: Pinecone.Index, dimension=384):
        """
        Initialize the QueryHandler with a Pinecone vectorstore and index.

        :param vectorstore: The Pinecone vectorstore instance used for document retrieval.
        :param pinecone_index: The Pinecone index for querying vector embeddings.
        :param dimension: The dimension of the vector embedding.
        """
        self.dimension = dimension
        self.vectorstore = vectorstore
        self.pinecone_index = pinecone_index
        print(f"Pinecone initialized successfully.")


    def get_answer(self, question, threshold=0.35, top_k=10, filter_false=False, max_workers=5,
                   query_generation_model="gpt-4o-mini", analysis_model="gpt-4o-mini"):
        """
        Retrieve and analyze documents relevant to the user's question.

        :param question: The user question.
        :param threshold: The score threshold for determining relevant documents.
        :param top_k: The number of top documents to retrieve for each query variant.
        :param filter_false: Whether to filter out documents flagged as irrelevant ('False').
        :param max_workers: Maximum number of threads for parallel processing.
        :param query_generation_model: The model name for generating alternative queries.
        :param analysis_model: The model name for summarizing and analyzing the documents.
        :return: A list of dictionaries containing analysis results for each article.
        """
        results = self.search_documents(user_query=question, top_k=top_k,
                                        query_generation_model=query_generation_model)
        entries = self.get_entries_with_score(results, threshold=threshold)
        return self.analyze_summaries(entries, question, filter_false=filter_false,
                                      analysis_model=analysis_model, max_workers=max_workers)


    def search_documents(self, user_query: str, top_k=10,
                         query_generation_model="gpt-4o-mini") -> List[Document]:
        """
        Retrieve documents relevant to the given user query.
        Generates multiple query variations to improve retrieval diversity, embeds them,
        queries the Pinecone index, merges results, removes duplicates, and sorts by similarity score.

        :param user_query: The original user query.
        :param top_k: The number of top documents to retrieve for each query variant.
        :param query_generation_model: The model name for generating alternative queries.
        :return: A list of Documents.
        """
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

        print(f"Retrieved {len(documents)} documents\n")
        for i, document in enumerate(documents):
            print(f"*** Document {i + 1} ***\n"
                  f"Source: {document.metadata['source']}\n"
                  f"Score: {document.metadata['score']}\n"
                  f"Page Content: {document.page_content[:200]}...\n\n")

        return documents


    def analyze_summaries(self, sitemap_entries: [SitemapEntry], question: str, max_workers=5,
                          filter_false=False, analysis_model="gpt-4o-mini") -> List[dict]:
        """
        Analyze each sitemap entry by loading the corresponding article, summarizing it,
        and determining if it can help the user based on query. Calls the LLM in parallel for efficiency.

        :param sitemap_entries: A list of SitemapEntry objects representing documents to analyze.
        :param question: The user question.
        :param max_workers: Maximum number of threads for parallel processing.
        :param filter_false: Whether to filter out documents flagged as irrelevant ('False').
        :param analysis_model: The model name for summarizing and analyzing the documents.
        :return: A list of dictionaries containing analysis results for each article.
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

        def process_url(sitemap_entry: SitemapEntry) -> dict:
            """
            Process a single sitemap entry: load the article, call the LLM, and return the analysis.

            :param sitemap_entry: A SitemapEntry object representing the article to analyze.
            :return: A dictionary containing 'url', 'score', 'decision', 'summary', and 'response'.
            """
            result_dict = {
                "url": sitemap_entry.url,
                "score": sitemap_entry.score,
                "decision": "False",
                "summary": "",
                "response": "",
            }

            # Load the document from database and process it with the LLM
            try:
                results = self.pinecone_index.query(
                    vector=[0] * self.dimension,
                    filter={"source": {"$eq": sitemap_entry.url}},
                    top_k=10000,
                    include_metadata=True,
                    include_values=False
                )
                documents = sorted(
                    (match['metadata'] for match in results['matches']),
                    key=lambda x: x.get('start_index', 0)
                )
                if not documents:
                    raise ValueError("No documents found.")

                reconstructed_string = ""
                current_position = 0
                for doc in documents:
                    start_index = int(doc.get('start_index', 0))
                    content = doc.get('text', "")
                    if start_index >= current_position:
                        reconstructed_string += content
                    else:
                        overlap_length = current_position - start_index
                        reconstructed_string += content[overlap_length:]
                    current_position = max(current_position, start_index + len(content))
                if not reconstructed_string:
                    raise ValueError("Couldn't reconstruct the document.")

                prompt = ChatPromptTemplate.from_template(template)
                messages = prompt.format_messages(query=question, context=reconstructed_string)
                result = llm.invoke(messages)
            except Exception as error:
                result_dict["response"] = f"Error processing URL {sitemap_entry.url}: {error}"
                return result_dict

            # Parse the LLM response
            try:
                result_split = [x.strip() for x in result.content.split("\n") if x.strip()]
                if len(result_split) == 3:
                    result_dict["decision"], result_dict["summary"], result_dict["response"] = result_split
                else:
                    result_dict["response"] = result.content
            except Exception as error:
                result_dict["response"] = f"Error processing LLM response for {sitemap_entry.url}: {error}"

            return result_dict


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
    def get_entries_with_score(documents: List[Document], threshold=0.35) -> List[SitemapEntry]:
        """
        Extract unique document sources, assign them scores, and apply a minimum score threshold.

        :param documents: A list of Document objects returned from search.
        :param threshold: The minimum score threshold for filtering documents.
        :return: A list of SitemapEntry objects passing the score threshold.
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
