import os
import logging
import time
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

from rag_query.query_handler import QueryHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlackBot:
    def __init__(self, rag_system, slack_bot_token, slack_app_token):
        self.app = App(token=slack_bot_token)
        self.rag_system = rag_system
        self.socket_handler = SocketModeHandler(app=self.app, app_token=slack_app_token)

        self._register_handlers()

    def _register_handlers(self):
        @self.app.event("message")
        def handle_message(event, say, ack, client):
            # Acknowledge the event
            ack()

            # Filter for private channels or direct messages
            if event.get('channel_type') in ['im', 'group']:
                text = event.get('text', '').strip()

                # Ignore bot's own messages
                if event.get('bot_id'):
                    return

                try:
                    # Extract the channel ID
                    channel_id = event['channel']

                    # Send a "Thinking..." placeholder message
                    thinking_message = client.chat_postMessage(
                        channel=channel_id,
                        text="_Thinking..._"
                    )

                    # Process the question and generate the response
                    response = self._process_question(text)

                    # Update the "Thinking..." message with the actual response
                    if isinstance(response, dict) and "blocks" in response:
                        client.chat_update(
                            channel=channel_id,
                            ts=thinking_message['ts'],  # Timestamp of the placeholder message
                            blocks=response["blocks"]
                        )
                    else:
                        client.chat_update(
                            channel=channel_id,
                            ts=thinking_message['ts'],
                            text=response
                        )
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    say("I encountered an error processing your request.")

    def _process_question(self, question):
        """Process question through RAG system and format response."""
        results = self.rag_system.get_answer(question)

        if not results:
            return "No relevant information found for your query."

        # Build Slack message using Block Kit
        blocks = self._build_slack_message_blocks(results)
        return {"blocks": blocks}

    def _build_slack_message_blocks(self, results):
        """Build Slack Block Kit message blocks from results."""
        print("DEBUG: Results Type:", type(results))
        print("DEBUG: Results Content:", results)
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Analysis Results:*"
                }
            },
            {"type": "divider"}
        ]

        for result in results[:20]:
            # Extract the URL
            url = result.get('url', 'URL not available')

            # Safely handle 'analysis'
            analysis = result.get('analysis', 'Analysis not available')
            if isinstance(analysis, str) and "Analysis:" in analysis:
                analysis = analysis.replace("Analysis:", "").strip()

            # Add to Slack blocks
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"<{url}>\n{analysis}"
                    }
                }
            )
            blocks.append({"type": "divider"})

        return blocks

    def start(self):
        """Start the bot"""
        try:
            logger.info("Starting Private Channel RAG Bot...")
            self.socket_handler.start()
        except Exception as e:
            logger.error(f"Bot startup error: {e}")
            raise

