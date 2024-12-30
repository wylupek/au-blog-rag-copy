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

    text = rag_system.get_answer("We’re discussing with a prospect client that wants to implement an AI feature in the app. Please show me article about our experience and knowledge", filter_false=False, analysis_model="gpt-4o")

    print("RESULTS:")
    print(text)

if __name__ == "__main__":
    main()
