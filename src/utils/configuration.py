from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from . import prompts


# noinspection PyUnresolvedReferences
@dataclass(kw_only=True)
class LoaderConfiguration:
    index_name: str = field(
        default="au-blog-rag-fine-tuned",
        metadata={"description": "Pinecone index name for vectorstore."},
    )

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}}
    ] = field(
        default="wylupek/au-blog-rag-embedder",
        metadata={"description": "Name of the embedding model to use. Must be in provider/model_name format. "
                                 "Only HuggingFace or OpenAi models available, e.g., 'wylupek/au-blog-rag-embedder' or 'openai/text-embedding-3-small'."
                                 "Must match the index dimensions."},
    )

    load_documents_batch_size: int = field(
        default=10,
        metadata={"description": "The number of documents to load in a single batch/node. It should be low to prevent timeouts."},
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
        metadata=
        {
            "description": "The system prompt used for generating alternative queries for retrieval.",
            "text_type": "prompt"
        },
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

    filter_false: bool = field(
        default=True,
        metadata={"description": "Filter out articles with decision 'False' from the retrieved articles."}
    )

    analysis_prompt: str = field(
        default=prompts.ANALYSIS_PROMPT,
        metadata=
        {
            "description": "The system prompt used for analyzing selected article.",
            "text_type": "prompt"
        },
    )

    analysis_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-4o",
        metadata={"description": "The OpenAI language model used for analyzing selected articles."},
    )

    result_decision_prompt: str = field(
        default=prompts.RESULT_DECISION_PROMPT,
        metadata=
        {
            "description": "The system prompt used for generating the decision for the retrieved article.",
            "text_type": "prompt"
        },
    )

    result_summary_prompt: str = field(
        default=prompts.RESULT_SUMMARY_PROMPT,
        metadata=
        {
            "description": "The system prompt used for generating the summary of the retrieved article.",
            "text_type": "prompt"
        },
    )

    result_analysis_prompt: str = field(
        default=prompts.RESULT_ANALYSIS_PROMPT,
        metadata=
        {
            "description": "The system prompt used for generating the analysis how the retrieved article is relevant to user's question.",
            "text_type": "prompt"
        },
    )
