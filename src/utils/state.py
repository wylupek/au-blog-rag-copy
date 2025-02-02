import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional, Sequence, Union, List
from src.loader_graph.sitemap_entry import SitemapEntry
from langchain_pinecone import PineconeVectorStore

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

############################  Doc Indexing State  #############################


def reduce_sitemap_entries(
    existing: Optional[Sequence[Document]],
    new: Union[
        Sequence[Document],
        Sequence[dict[str, Any]],
        Sequence[str],
        str,
        Literal["delete"],
    ],
) -> Sequence[Document]:
    if new == "delete":
        return []
    # if isinstance(new, str):
    #     return [Document(page_content=new, metadata={"id": str(uuid.uuid4())})]
    # if isinstance(new, list):
    #     coerced = []
    #     for item in new:
    #         if isinstance(item, str):
    #             coerced.append(
    #                 Document(page_content=item, metadata={"id": str(uuid.uuid4())})
    #             )
    #         elif isinstance(item, dict):
    #             coerced.append(Document(**item))
    #         else:
    #             coerced.append(item)
    #     return coerced
    return existing or []


@dataclass(kw_only=True)
class LoaderState:
    sitemap_entries: Optional[Annotated[List[SitemapEntry], reduce_sitemap_entries]] = None
    # docs: Annotated[List[Document], reduce_sitemap_entries]
    # vectorstore: Optional[PineconeVectorStore] = No

@dataclass(kw_only=True)
class LoaderInputState:
    sitemap: str
