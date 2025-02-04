from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from src.utils import prompts


# noinspection PyUnresolvedReferences
@dataclass(kw_only=True)
class IndexConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including user identification, embedding model selection,
    retriever provider choice, and search parameters.
    """

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    index_name: str = field(
        default="default",
        metadata={
            "description": "Pinecone index name for vectorstore."
        },
    )


    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=IndexConfiguration)

# noinspection PyUnresolvedReferences
@dataclass(kw_only=True)
class RAGConfiguration(IndexConfiguration):
    """The configuration for the agent."""

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
        metadata={
            "description": "The OpenAI language model used for generating alternative queries."
        },
    )

    analysis_prompt: str = field(
        default=prompts.ANALYSIS_PROMPT,
        metadata={"description": "The system prompt used for analyzing the retrieved documents."},
    )

    analysis_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-4o",
        metadata={
            "description": "The OpenAI language model used for analyzing the retrieved documents."
        },
    )

    filter_false: bool = field(
        default=False,
        metadata={"description": "Filter out articles with decision 'False' from the retrieved articles."}
    )

    # response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
    #     default="openai/gpt-4-mini",
    #     metadata={
    #         "description": "The language model used for generating responses. Should be in the form: provider/model-name."
    #     },
    # )
    #
    # query_system_prompt: str = field(
    #     default=prompts.QUERY_SYSTEM_PROMPT,
    #     metadata={
    #         "description": "The system prompt used for processing and refining queries."
    #     },
    # )
    #
    # query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
    #     default="openai/gpt-4-mini",
    #     metadata={
    #         "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name."
    #     },
    # )
