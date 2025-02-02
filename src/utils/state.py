from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional, Sequence, Union, List

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
    sitemap_entries: Optional[Annotated[List[SitemapEntry], reduce_sitemap_entries]] = None
    vector_store_manager: Optional[VectorStoreManager] = None


@dataclass(kw_only=True)
class LoaderInputState:
    sitemap: str
