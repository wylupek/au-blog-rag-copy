import os
from dotenv import load_dotenv
from slack_integration.slack_bot import SlackBot
from rag_query.query_handler import QueryHandler

def main():
    dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path)

    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")

    if not bot_token or not app_token:
        raise ValueError("Missing Slack tokens")

    rag_system = QueryHandler(
        llm_model_name="gpt-4o-mini",
        embeddings_model_name="text-embedding-3-small",
        vector_db_path="./vector_db"
    )

    slack_bot = SlackBot(
        rag_system,
        slack_bot_token=bot_token,
        slack_app_token=app_token
    )

    slack_bot.start()

if __name__ == "__main__":
    main()
