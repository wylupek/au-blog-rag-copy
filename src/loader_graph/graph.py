from typing import Optional, List
from .sitemap_entry import SitemapEntry

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from src.utils.configuration import IndexConfiguration
from src.utils.state import LoaderState, LoaderInputState
from .sitemap_entry import Sitemap
from .document_processor import DocumentProcessor


async def extract_sitemap_entries(state: LoaderInputState) -> dict[str, List[SitemapEntry]]:
    sitemap = Sitemap(sitemap=state.sitemap)
    return {"sitemap_entries": sitemap.load()}


async def load_entries(
    state: LoaderState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    if not config:
        raise ValueError("Configuration required to run load_entries.")
    configuration = IndexConfiguration.from_runnable_config(config)
    document_processor = DocumentProcessor(configuration.index_name)
    document_processor.update_database(state.sitemap_entries)
    return {"docs": "delete"}


builder = StateGraph(LoaderState, input=LoaderInputState, config_schema=IndexConfiguration)
builder.add_node(extract_sitemap_entries)
builder.add_node(load_entries)
builder.add_edge("__start__", "extract_sitemap_entries")
builder.add_edge("extract_sitemap_entries", "load_entries")
graph = builder.compile()
graph.name = "LoaderGraph"
