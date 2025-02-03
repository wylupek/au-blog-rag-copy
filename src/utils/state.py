from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional, Sequence, Union, List

from langchain_core.documents import Document

from src.loader_graph.vector_store_manager import VectorStoreManager
from src.loader_graph.sitemap_entry import SitemapEntry


def reduce_sitemap_entries(
    existing: Optional[Sequence[Document]],
    new: Union[
        List[SitemapEntry],
        Literal["delete"],
    ],
) -> Sequence[Document]:
    if new == "delete":
        return []
    return existing or []


@dataclass(kw_only=True)
class LoaderState:
    sitemap_entries: Annotated[List[SitemapEntry], reduce_sitemap_entries] = field(default_factory=list)
    vector_store_manager: Optional[VectorStoreManager] = None # Maybe initialize this earlier, so RAG graph can use it


@dataclass(kw_only=True)
class LoaderInputState:
    sitemap: str


def reduce_generated_queries(
    existing: List[str],
    new: List[str]
) -> List[str]:
    return existing + new


@dataclass(kw_only=True)
class RAGState(LoaderState):
    query: str
    generated_queries: Annotated[List[str], reduce_generated_queries] = field(default_factory=list)

