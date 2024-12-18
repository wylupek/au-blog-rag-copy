import os
from dotenv import load_dotenv
from slack_integration.slack_bot import SlackBot
from rag_query.query_handler import QueryHandler
from data_loaders.document_processor import DocumentProcessor

from flask import Flask

def main():
    load_dotenv()

    # Load Slack tokens from environment variables
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    signing_secret = os.getenv("SLACK_SIGNING_SECRET")
    if not bot_token or not signing_secret:
        raise ValueError("Missing Slack tokens or signing secret in environment variables.")

    # Initialize the document processor and query handler
    document_processor = DocumentProcessor('au-blog-rag')

    rag_system = QueryHandler(
        vectorstore=document_processor.vectorstore,
        pinecone_index=document_processor.pinecone_index
    )

    # Initialize the SlackBot
    slack_bot = SlackBot(
        rag_system=rag_system,
        slack_bot_token=bot_token,
        signing_secret=signing_secret
    )

    # Start the Flask app (used for webhooks)
    slack_bot.start()


if __name__ == "__main__":
    main()
