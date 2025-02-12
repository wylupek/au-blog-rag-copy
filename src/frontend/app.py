from typing import Type
from dataclasses import fields
import sys
import os
import requests

from dotenv import load_dotenv
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.configuration import LoaderConfiguration, RAGConfiguration

load_dotenv()
api_key = os.getenv("LANGCHAIN_API_KEY")
api_url = os.getenv("API_URL")



def generate_config_ui(config_class: Type[LoaderConfiguration]|Type[RAGConfiguration], is_main_ui: bool, main_keys: list) -> dict:
    """
    Dynamically generates a configuration UI based on the dataclass fields.
    Returns a dictionary with the field names and the user-provided values.
    """
    config_values = {}
    for field_obj in fields(config_class):
        name = field_obj.name
        if (name in main_keys) ^ is_main_ui: # XOR
            continue
        label = name.replace("_", " ").capitalize()
        default = field_obj.default
        description = field_obj.metadata.get("description", "")
        text_type = field_obj.metadata.get("text_type", None)
        field_type = field_obj.type

        if field_type == "int":
            config_values[name] = st.number_input(
                label=label,
                value=default,
                help=description,
                step=1
            )
        elif field_type == "float":
            config_values[name] = st.number_input(
                label=label,
                value=default,
                help=description,
                step=0.01
            )
        elif field_type == "bool":
            config_values[name] = st.checkbox(
                label=label,
                value=default,
                help=description
            )
        elif field_type == "str":
            if text_type == "prompt":
                config_values[name] = st.text_area(
                    label=label,
                    value=default,
                    help=description
                )
            else:
                config_values[name] = st.text_input(
                    label=label,
                    value=default,
                    help=description
                )
        else:
            config_values[name] = st.text_input(
                label=label,
                value=str(default),
                help=description
            )
    return config_values


def create_thread() -> str | None:
    url = f"{api_url}/threads/"
    headers = {"X-Api-Key": api_key}
    payload = {
        "thread_id": "",
        "metadata": {},
        "if_exists": "raise"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["thread_id"]
    else:
        st.error(f"Error creating thread: {response.status_code} {response.text}")
        return None


def create_background_run(thread_id: str, graph_option: str, user_input: str, config_data: dict):
    # Choose the assistant and input key based on the graph option.
    if graph_option == "Loader Graph":
        assistant_id = "loader_graph"
        input_data = {"sitemap": user_input}
    else:
        assistant_id = "retrieval_graph"
        input_data = {"query": user_input}
    url = f"{api_url}/threads/{thread_id}/runs"
    headers = {"X-Api-Key": api_key}
    payload = {
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "input": input_data,
        "metadata": {"debug_info": "Streamlit run"},
        "config": config_data
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["run_id"]
    else:
        st.error(f"Error creating background run: {response.status_code} {response.text}")
        return None


def join_run(thread_id, run_id):
    url = f"{api_url}/threads/{thread_id}/runs/{run_id}/join"
    headers = {"X-Api-Key": api_key}
    payload = {
        "thread_id": thread_id,
        "run_id": run_id
    }
    response = requests.get(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error joining run: {response.status_code} {response.text}")
        return None


def format_rag_response(analysis_list):
    response = ""
    for i, analysis in enumerate(analysis_list):
        response += f"### Information\n"
        response += f"* **URL:** [{analysis['url']}]({analysis['url']})\n"
        response += f"* **Score:** {analysis['score']:.2f}\n"
        response += f"* **Decision:** {analysis['decision']}\n"
        response += f"### Summary\n{analysis['summary']}\n"
        response += f"### Analysis\n{analysis['analysis']}\n"

        if i < len(analysis_list) - 1:
            response += "\n---\n\n"
    return response


def main(main_config_keys):
    with st.sidebar:
        st.title('AppUnite Blog Agent')
        st.markdown(
            "This agent is designed to help sales teams during client discussions by quickly retrieving relevant blog articles. It implements two main components: \n"
            "* **RAG Graph** - Uses a language model to retrieve ranked articles from the database and analyze them. "
            "Each article response includes a one-word relevance decision, a short summary, and an explanation of how it addresses the client's problem.\n"
            "* **Loader Graph** - Scrapes articles from a given sitemap, processes them, and updates a Pinecone vector database with the documents. "
            "It can be used for an already loaded sitemap, for new articles or updates, or for an entirely new sitemap.\n"
        )

        st.write("## Select Graph")
        graph_option = st.radio(
            "Select Graph",
            options=["RAG Graph", "Loader Graph"],
            index=0,
            help = "Select the graph to use for generating responses.",
            label_visibility="collapsed",
            # disabled=True
        )

        st.write(f"## Graph Configuration")
        if graph_option == "Loader Graph":
            config_data = generate_config_ui(LoaderConfiguration, is_main_ui=True, main_keys=main_config_keys)
        else:
            config_data = generate_config_ui(RAGConfiguration, is_main_ui=True, main_keys=main_config_keys)

        with st.expander("Optional Configuration", expanded=False):
            if graph_option == "Loader Graph":
                config_data.update(generate_config_ui(LoaderConfiguration, is_main_ui=False, main_keys=main_config_keys))
            else:
                config_data.update(generate_config_ui(RAGConfiguration, is_main_ui=False, main_keys=main_config_keys))

    # Create a thread for the session if it doesn't exist.
    if "thread_id" not in st.session_state:
        thread_id = create_thread()
        if thread_id:
            st.session_state.thread_id = thread_id
        else:
            st.stop()  # Stop execution if thread creation failed.

    # Initialize chat messages if they don't exist.
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to the AppUnite Blog RAG! How can I help you today?"}]

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    # User-provided prompt
    if prompt := st.chat_input(
            placeholder="Ask what articles do you need." if graph_option == "RAG Graph"
            else "Provide a sitemap URL of the blog."
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                run_id = create_background_run(st.session_state.thread_id, graph_option, prompt, config_data)
                if run_id:
                    response = join_run(st.session_state.thread_id, run_id)
                    if graph_option == "Loader Graph":
                        response = response["documents_count"]
                    else:
                        response = format_rag_response(response["analyses"])
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main(main_config_keys=["top_k", "threshold", "filter_false"])
