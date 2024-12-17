import os
from dotenv import load_dotenv
from slack_integration.slack_bot import SlackBot
from rag_query.query_handler import QueryHandler
from data_loaders.sitemap_entry import Sitemap
from data_loaders.document_processor import DocumentProcessor

def main():
    load_dotenv()

    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")
    if not bot_token or not app_token:
        raise ValueError("Missing Slack tokens in environment variables.")

    #
    document_processor = DocumentProcessor('au-blog-rag')

    rag_system = QueryHandler(
        vectorstore=document_processor.vectorstore,
        pinecone_index=document_processor.pinecone_index
    )

    slack_bot = SlackBot(
        rag_system=rag_system,
        slack_bot_token=bot_token,
        slack_app_token=app_token
    )
    slack_bot.start()


if __name__ == "__main__":
    main()
