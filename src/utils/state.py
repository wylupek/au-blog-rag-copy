from dataclasses import dataclass, field
from typing import Annotated, Optional, List

from langchain_core.documents import Document
from src.utils.vector_store_manager import VectorStoreManager
from src.utils.sitemap_entry import SitemapEntry



@dataclass(kw_only=True)
class LoaderState:
    sitemap_entries: List[SitemapEntry] = field(default_factory=list)
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
    retrieved_documents: List[Document] = field(default_factory=list)

class RAGOutputState:
    analyses: List[dict] =  field(default_factory=list)
