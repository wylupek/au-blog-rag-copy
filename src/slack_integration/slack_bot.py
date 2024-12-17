import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class SlackBot:
    """
    SlackBot class for handling Slack messages, interacting with a RAG system, and responding using Slack Block Kit.
    """
    def __init__(self, rag_system, slack_bot_token, slack_app_token):
        """
        Initialize the SlackBot with a RAG system, Slack bot token, and Slack app token.

        :param rag_system: The QueryHandler object for answering user queries.
        :param slack_bot_token: The Slack bot token for authentication.
        :param slack_app_token: The Slack app-level token for Socket Mode.
        """
        self.app = App(token=slack_bot_token)
        self.rag_system = rag_system
        self.socket_handler = SocketModeHandler(app=self.app, app_token=slack_app_token)
        self._register_handlers()


    def _register_handlers(self):
        @self.app.event("message")
        def handle_message(event, say, ack, client):
            """Main entry point for handling Slack messages."""
            ack()
            try:
                if not self._is_valid_event(event):
                    return

                channel_id = event['channel']
                text = event.get('text', '').strip()

                # Send typing indicator
                user_id = event.get('user')
                self._send_typing_placeholder(client, channel_id, user_id)

                # Process the user's question and generate a response
                response = self._generate_response(text)

                # Send the response back to the user
                self._send_response(client, channel_id, response)

            except ValueError as ve:
                logger.error(f"Validation error: {ve}")
                say(str(ve))
            except Exception as e:
                logger.error(f"Unexpected error while processing message: {e}")
                say("An unexpected error occurred. Please try again later.")


    @staticmethod
    def _is_valid_event(event):
        """Validate the Slack event."""
        if event.get('channel_type') not in ['im', 'group']:
            return False
        if event.get('bot_id'):  # Ignore bot messages
            return False
        if not event.get('text', '').strip():
            raise ValueError("Message text is empty.")
        return True


    @staticmethod
    def _send_typing_placeholder(client, channel_id, user):
        """Send a placeholder 'Thinking...' message."""
        try:
            return client.chat_postEphemeral(
                channel=channel_id,
                text="_Thinking..._",
                user=user
            )
        except Exception as e:
            logger.error(f"Failed to send 'Thinking...' placeholder: {e}")
            raise ValueError("Unable to send 'Thinking...' placeholder.")


    def _generate_response(self, text):
        """
        Generate a response to the user's question using the RAG system.

        :param text: The user's query text.
        :return: A dictionary containing Slack Block Kit message blocks or a plain text message.
        """
        results = self.rag_system.get_answer(text, filter_false=False, analysis_model="gpt-4o")
        if not results:
            return "No relevant information found for your query."
        return {"blocks": self._build_slack_message_blocks(results)}


    @staticmethod
    def _send_response(client, channel_id, response):
        """
        Send the generated response back to the Slack channel.

        :param client: The Slack WebClient for API interactions.
        :param channel_id: The ID of the Slack channel where the response will be sent.
        :param response: The response to send, either as plain text or Block Kit message blocks.
        :raises ValueError: If sending the response message fails.
        """
        try:
            if isinstance(response, dict) and "blocks" in response:
                client.chat_postMessage(
                    channel=channel_id,
                    blocks=response["blocks"],
                    unfurl_links=False  # Disable link unfurling
                )
            else:
                client.chat_postMessage(
                    channel=channel_id,
                    text=response,
                    unfurl_links=False  # Disable link unfurling
                )
        except Exception as e:
            logger.error(f"Failed to send response message: {e}")
            raise ValueError("Unable to send the Slack response.")


    @staticmethod
    def _build_slack_message_blocks(results):
        """
        Build Slack Block Kit message blocks from query results.

        :param results: List of dictionaries containing the following fields:
            - url: The URL of the article.
            - score: The retriever score of the article.
            - decision: The decision about article relevance.
            - summary: A brief summary of the article.
            - response: The response to the user's question, based on the article content.
        :return: A list of Slack Block Kit message blocks.
        """
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
            url = result.get('url', 'URL not available')
            # score = result.get('score', 'Score not available')
            decision = result.get('decision', 'Decision not available')
            summary = result.get('summary', 'Summary not available')
            response = result.get('response', 'Response not available')

            result_text = (
                f"*< {url} >*\n"
                # f"• *Score:* `{score}`\n"
                f"• *Decision:* `{decision}`\n"
                f"• *Summary:* {summary}\n"
                f"• *Response:* {response}"
            )

            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": result_text
                    }
                }
            )
            blocks.append({"type": "divider"})
        return blocks


    def start(self):
        """Start the bot."""
        try:
            logger.info("Starting Private Channel RAG Bot...")
            self.socket_handler.start()
        except Exception as e:
            logger.error(f"Bot startup error: {e}")
            raise

