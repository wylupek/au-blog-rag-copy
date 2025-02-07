from typing import Optional, List, Literal
from dataclasses import dataclass, field

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import create_model, Field

from src.utils.configuration import RAGConfiguration
from src.utils.state import RAGState, RAGOutputState
from src.utils.vector_store_manager import VectorStoreManager
from src.utils.sitemap_entry import SitemapEntry


@dataclass(kw_only=True)
class Result:
    url: str = field(
        metadata={"description": "The URL of the article being analyzed"}
    )
    score: float = field(
        default=0,
        metadata={"description": "The similarity score of the article to the user's query"},
    )

    decision: bool = field(
        default=False,
        metadata={"description": "The decision of the analysis"},
    )
    summary: str = field(
        default="",
        metadata={"description": "The summary of the article"},
    )
    analysis: str = field(
        default="",
        metadata={"description": "The analysis of the article and user query"},
    )

    def __str__(self):
        return (f"Result(\n"
                f"\tdecision = {self.decision} | score = {self.score:.3f} | url = {self.url}\n"
                f"\tanalysis = {self.analysis}\n"
                f"\tsummary = {self.summary}\n"
                f")")


async def initialize_vector_store(
    state: RAGState, *, config: Optional[RunnableConfig] = None
) -> dict[str, VectorStoreManager]:
    print("Initializing vector store")
    if not config:
        raise ValueError("Configuration required to run initialize_vector_store.")
    configuration = RAGConfiguration.from_runnable_config(config)
    return {"vector_store_manager": VectorStoreManager(configuration.index_name)}


async def generate_query_variants(
        state: RAGState, *, config: Optional[RunnableConfig] = None
) -> dict[str, List[str]]:
    print("Generating query variants")
    if not config:
        raise ValueError("Configuration required to run initialize_vector_store.")
    configuration = RAGConfiguration.from_runnable_config(config)

    # RAGInputState doesn't have the generated_queries attribute, but RAGState does.
    queries_to_generate = (configuration.num_query_variants -
                           (len(state.generated_queries) if state.generated_queries else 0))
    previous_queries_text = "\n".join(state.generated_queries if state.generated_queries else [])

    template = configuration.query_variants_prompt
    template = " ".join(line.strip() for line in template.strip().splitlines())
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0, model_name=configuration.query_variants_model)
    messages = prompt.format_messages(
        num_variants=queries_to_generate,
        question=state.query + "\n",
        previous_queries=previous_queries_text
    )
    result = llm.invoke(messages)

    generated_queries = [q.strip() for q in result.content.split("\n") if q.strip()]
    return {"generated_queries": generated_queries}


async def check_query_variants(
    state: RAGState, *, config: Optional[RunnableConfig] = None
) -> Literal["retrieve_documents", "generate_query_variants"]:
    if not config:
        raise ValueError("Configuration required to run initialize_vector_store.")
    configuration = RAGConfiguration.from_runnable_config(config)
    num_query_variants = configuration.num_query_variants

    current_count = len(state.generated_queries) if state.generated_queries else 0
    sufficient = current_count >= num_query_variants
    if sufficient:
        return "retrieve_documents"
    return "generate_query_variants"


async def retrieve_documents(
        state: RAGState, *, config: Optional[RunnableConfig] = None
) -> dict[str, List[Document]]:
    print("Retrieving documents")
    if not state.generated_queries or len(state.generated_queries) == 0:
        raise ValueError("No generated queries found in state.")
    if not state.vector_store_manager:
        raise ValueError("No vector store manager found in state.")
    if not config:
        raise ValueError("Configuration required to run initialize_vector_store.")
    configuration = RAGConfiguration.from_runnable_config(config)

    vsm = state.vector_store_manager

    # Perform searches for all query embeddings and merge results
    query_embeddings = [vsm.embeddings.embed_query(q) for q in state.generated_queries]
    results = []
    for query in query_embeddings:
        response = vsm.pinecone_index.query(
            vector=query,
            top_k=configuration.top_k,
            include_metadata=True,
            include_values=True,
        )
        results.extend(response.matches)

    # Remove duplicate documents by match ID.
    seen_ids = set()
    unique_results = []
    for match in results:
        if match.id not in seen_ids:
            seen_ids.add(match.id)
            unique_results.append(match)

    # Convert the unique matches into LangChain Documents,
    # computing a similarity score using the first query embedding.
    first_query_embedding = query_embeddings[0] if query_embeddings else None
    documents = []
    for match in unique_results:
        if first_query_embedding is None:
            score = 0
        else:
            score = cosine_similarity([first_query_embedding], [match.values])[0][0]
        doc = Document(
            page_content=match.metadata.get("text", ""),
            metadata={**match.metadata, "score": score},
        )
        documents.append(doc)

    # Print some diagnostic information.
    # print(f"Retrieved {len(documents)} documents")
    # for i, doc in enumerate(documents):
    #     print(f"*** Document {i + 1} ***")
    #     print(f"Source: {doc.metadata.get('source', 'unknown')}")
    #     print(f"Score: {doc.metadata.get('score')}")
    #     print(f"Page Content: {doc.page_content[:200]}...\n")

    return {"retrieved_documents": documents}


async def get_entries_with_score(
        state: RAGState, *, config: Optional[RunnableConfig] = None
) -> dict[str, List[SitemapEntry]]:
    print("Getting entries with score")
    if not state.retrieved_documents or len(state.retrieved_documents) == 0:
        raise ValueError("No retrieved documents found in state.")
    if not config:
        raise ValueError("Configuration required to run initialize_vector_store.")
    configuration = RAGConfiguration.from_runnable_config(config)

    documents: List[Document] = state.retrieved_documents
    threshold = configuration.threshold

    unique_sources = {}
    for document in documents:
        if (document.metadata["source"] not in unique_sources
                or document.metadata["score"] > unique_sources[document.metadata["source"]]):
            unique_sources[document.metadata["source"]] = document.metadata["score"]

    # for i, (source, score) in enumerate(unique_sources.items()):
    #     print(f"{i + 1}. {score:.3f} - {source}")

    sitemap_entries = [
        SitemapEntry(url=source, lastmod=None, score=score)
        for source, score in unique_sources.items() if score > threshold
    ]
    return {"sitemap_entries": sitemap_entries}


async def analyze_summaries(
    state: RAGState, *, config: Optional[RunnableConfig] = None
) -> dict[str, List[Result]]:
    print("Analyzing summaries")
    if not config:
        raise ValueError("Configuration required to run initialize_vector_store.")
    if not state.sitemap_entries:
        raise ValueError("No sitemap entries found in state.")
    if not state.vector_store_manager:
        raise ValueError("No vector store manager found in state.")

    configuration = RAGConfiguration.from_runnable_config(config)
    sitemap_entries: List[SitemapEntry] = state.sitemap_entries
    vsm = state.vector_store_manager

    analysis_model = configuration.analysis_model
    filter_false = configuration.filter_false
    template = configuration.analysis_prompt

    analysis_scheme = create_model(
        "AnalysisModel",
        decision=(bool, Field(default=False, description=configuration.result_decision_prompt)),
        summary=(str, Field(default="", description=configuration.result_summary_prompt)),
        analysis=(str, Field(default="", description=configuration.result_analysis_prompt)),
    )

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0.4, model_name=analysis_model)
    structured_llm = llm.with_structured_output(analysis_scheme)

    # Define a helper function that processes a single sitemap entry.
    def process_entry(sitemap_entry: SitemapEntry) -> Result:
        result = Result(url=sitemap_entry.url, score=sitemap_entry.score)
        # Reconstruct the article text by merging fragments.
        try:
            response = vsm.pinecone_index.query(
                vector=[0] * vsm.dimension,
                filter={"source": {"$eq": sitemap_entry.url}},
                top_k=10000,
                include_metadata=True,
                include_values=False
            )
            docs = sorted(
                (match["metadata"] for match in response["matches"]),
                key=lambda x: x.get("start_index", 0)
            )
            if not docs:
                raise ValueError("No documents found.")

            reconstructed = ""
            current_pos = 0
            for doc in docs:
                start_index = int(doc.get("start_index", 0))
                content = doc.get("text", "")
                if start_index >= current_pos:
                    reconstructed += content
                else:
                    overlap = current_pos - start_index
                    reconstructed += content[overlap:]
                current_pos = max(current_pos, start_index + len(content))
            if not reconstructed:
                raise ValueError("Couldn't reconstruct the document.")
        except Exception as error:
            result.analysis = f"Error processing URL {sitemap_entry.url}: {error}"
            return result

        # Format the prompt with the original query and the reconstructed article text.
        messages = prompt.format_messages(query=state.query, context=reconstructed)
        llm_result = structured_llm.invoke(messages)

        try:
            result.decision = llm_result.decision
            result.summary = llm_result.summary
            result.analysis = llm_result.analysis
        except Exception as error:
            result.analysis = f"Error processing LLM response for {sitemap_entry.url}: {error}"
        return result

    # Process each sitemap entry concurrently using ThreadPoolExecutor.
    max_workers = 5
    analyses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {executor.submit(process_entry, entry): entry for entry in sitemap_entries}
        for future in as_completed(future_to_entry):
            try:
                analyses.append(future.result())
            except Exception as e:
                analyses.append(Result(url=future_to_entry[future].url, analysis=f"Error in processing: {e}"))


    analyses = sorted(analyses, key=lambda x: x.score, reverse=True)
    if filter_false:
        analyses = [analysis for analysis in analyses if analysis.decision != False]
    else:
        analyses = ([analysis for analysis in analyses if analysis.decision == True] +
                    [analysis for analysis in analyses if analysis.decision != True])
    return {"analyses": analyses}


builder = StateGraph(RAGState, output=RAGOutputState, config_schema=RAGConfiguration)
builder.add_node(initialize_vector_store)
builder.add_node(generate_query_variants)
builder.add_node(retrieve_documents)
builder.add_node(get_entries_with_score)
builder.add_node(analyze_summaries)

builder.add_edge("__start__", "initialize_vector_store")
builder.add_edge("__start__", "generate_query_variants")
builder.add_edge("initialize_vector_store", "retrieve_documents")
builder.add_conditional_edges(
    "generate_query_variants",
    check_query_variants
)
builder.add_edge("retrieve_documents", "get_entries_with_score")
builder.add_edge("get_entries_with_score", "analyze_summaries")

graph = builder.compile()
graph.name = "RagGraph"