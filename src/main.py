import os
from dotenv import load_dotenv
from slack_integration.slack_bot import SlackBot
from rag_query.query_handler import QueryHandler
from data_loaders.sitemap_entry import Sitemap
from data_loaders.document_processor import DocumentProcessor
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def main():
    dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv()

    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")

    if not bot_token or not app_token:
        raise ValueError("Missing Slack tokens in environment variables.")

    # load sitemaps entries
    sitemap = Sitemap(sitemap="https://tech.appunite.com/blog/blog-sitemap.xml")
    sitemap_entries = sitemap.load()

    # debug entries
    # for entry in sitemap_entries:
    #     print(f"URL: {entry.url}, Last Modified: {entry.lastmod}")

    # Extract only URLs
    urls = [entry.url for entry in sitemap_entries]

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_db = Chroma(
        persist_directory="./vector_db",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # create vector database
    document_processor = DocumentProcessor(vector_db=vector_db)
    document_processor.load_and_index_documents(urls=urls)

    # 
    rag_system = QueryHandler(
        vector_db=vector_db
    )

    #
    slack_bot = SlackBot(
        rag_system=rag_system,
        slack_bot_token=bot_token,
        slack_app_token=app_token
    )

    slack_bot.start()

if __name__ == "__main__":
    main()
