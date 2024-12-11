import os
from dotenv import load_dotenv
from slack_integration.slack_bot import SlackBot
from rag_query.query_handler import QueryHandler
from data_loaders.sitemap_entry import Sitemap
from data_loaders.document_processor import DocumentProcessor



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

    document_processor = DocumentProcessor('au-blog-rag')
    document_processor.update_database(sitemap_entries)

    # 
    rag_system = QueryHandler(
        vectorstore=document_processor.vectorstore,
        pinecone_index=document_processor.pinecone_index
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
