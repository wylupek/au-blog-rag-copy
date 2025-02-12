import streamlit as st
from dataclasses import fields
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.configuration import LoaderConfiguration, RAGConfiguration



def generate_config_ui(config_class):
    """
    Dynamically generates a configuration UI based on the dataclass fields.
    Returns a dictionary with the field names and the user-provided values.
    """
    config_values = {}
    if config_class.__name__ == "LoaderConfiguration":
        st.write(f"### Loader Graph Configuration")
    elif config_class.__name__ == "RAGConfiguration":
        st.write(f"### RAG Graph Configuration")
    else:
        st.write("### Configuration")

    for field_obj in fields(config_class):
        name = field_obj.name
        label = name.replace("_", " ").capitalize()
        default = field_obj.default
        description = field_obj.metadata.get("description", "")
        text_type = field_obj.metadata.get("text_type", None)
        field_type = field_obj.type

        # Handle common types:
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
            # Fallback for any other type—convert default to string.
            config_values[name] = st.text_input(
                label=label,
                value=str(default),
                help=description
            )

    return config_values


def main():
    with st.sidebar:
        st.title('AppUnite Blog RAG')

        graph_option = st.radio(
            "Select Graph",
            options=["RAG Graph", "Loader Graph"],
            index=0
        )

        with st.expander("Optional Configuration", expanded=False):
            if graph_option == "Loader Graph":
                config_data = generate_config_ui(LoaderConfiguration)
            else:
                config_data = generate_config_ui(RAGConfiguration)


    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to the AppUnite Blog RAG! How can I help you today?"}]

    # Display chat messages
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
                response = "### Markdown output" # TODO
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
