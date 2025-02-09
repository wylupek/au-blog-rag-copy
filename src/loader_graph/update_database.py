import asyncio
from langchain_core.runnables import RunnableConfig

from src.loader_graph.graph import graph
from src.utils.state import LoaderInputState


async def run_loader():
    config = RunnableConfig(
        configurable={
            "index_name": "au-blog-rag-fine-tuned",
            "embedding_model": "openai/text-embedding-3-small"
        }
    )
    input_data = LoaderInputState(sitemap="https://tech.appunite.com/blog/blog-sitemap.xml")
    output = await graph.ainvoke(input_data, config=config)
    print(output)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(run_loader())