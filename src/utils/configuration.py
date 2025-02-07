from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from src.utils import prompts


# noinspection PyUnresolvedReferences
@dataclass(kw_only=True)
class LoaderConfiguration:
    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}}
    ] = field(
        default="wylupek/au-blog-rag-embedder",
        metadata={"description": "Name of the embedding model to use. "
                                 "Must be a 'wylupek/au-blog-rag-embedder' or 'openai/text-embedding-3-small.'"},
    )

    index_name: str = field(
        default="au-blog-rag-fine-tuned",
        metadata={"description": "Pinecone index name for vectorstore."},
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=LoaderConfiguration)


# noinspection PyUnresolvedReferences
@dataclass(kw_only=True)
class RAGConfiguration(LoaderConfiguration):
    num_query_variants: int = field(
        default=4,
        metadata={"description": "The number of alternative retrieval queries to generate."}
    )

    query_variants_prompt: str = field(
        default=prompts.QUERY_VARIANTS_PROMPT,
        metadata={"description": "The system prompt used for generating alternative queries for retrieval."},
    )

    query_variants_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-4o-mini",
        metadata={"description": "The OpenAI language model used for generating alternative queries."},
    )

    top_k: int = field(
        default=10,
        metadata={"description": "The number of documents to retrieve from the vector store for each query."}
    )

    threshold: float = field(
        default=0.30,
        metadata={"description": "The threshold for cosine similarity between the query and retrieved documents."}
    )

    analysis_prompt: str = field(
        default=prompts.ANALYSIS_PROMPT,
        metadata={"description": "The system prompt used for analyzing selected article."},
    )

    analysis_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-4o",
        metadata={"description": "The OpenAI language model used for analyzing selected articles."},
    )

    filter_false: bool = field(
        default=True,
        metadata={"description": "Filter out articles with decision 'False' from the retrieved articles."}
    )

    result_decision_prompt: str = field(
        default=prompts.RESULT_DECISION_PROMPT,
        metadata={"description": "The system prompt used for generating the decision for the retrieved article."},
    )

    result_summary_prompt: str = field(
        default=prompts.RESULT_SUMMARY_PROMPT,
        metadata={"description": "The system prompt used for generating the summary of the retrieved article."},
    )

    result_analysis_prompt: str = field(
        default=prompts.RESULT_ANALYSIS_PROMPT,
        metadata={"description": "The system prompt used for generating the analysis how the retrieved article is relevant to user's question."},
    )
