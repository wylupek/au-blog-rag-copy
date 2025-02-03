from typing import Optional, List

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from src.utils.configuration import RAGConfiguration
from src.utils.state import RAGState


async def generate_query_variants(
        state: RAGState, *, config: Optional[RunnableConfig] = None
) -> dict[str, List[str]]:
    if not config:
        raise ValueError("Configuration required to run initialize_vector_store.")
    configuration = RAGConfiguration.from_runnable_config(config)

    # RAGInputState doesn't have the generated_queries attribute, but RAGState does.
    queries_to_generate = (configuration.num_query_variants -
                           (len(state.generated_queries) if state.generated_queries else 0))
    previous_queries_text = "\n".join(state.generated_queries if state.generated_queries else [])

    template = configuration.query_variants_prompt
    template = " ".join(line.strip() for line in template.strip().splitlines())
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    messages = prompt.format_messages(
        num_variants=queries_to_generate,
        question=state.query + "\n",
        previous_queries=previous_queries_text
    )
    result = llm.invoke(messages)

    generated_queries = [q.strip() for q in result.content.split("\n") if q.strip()]
    return {"generated_queries": generated_queries}


async def check_query_variants(
    state: RAGState, *, config: Optional[RunnableConfig] = None
) -> str:
    if not config:
        raise ValueError("Configuration required to run initialize_vector_store.")
    configuration = RAGConfiguration.from_runnable_config(config)
    num_query_variants = configuration.num_query_variants

    current_count = len(state.generated_queries) if state.generated_queries else 0
    sufficient = current_count >= num_query_variants
    if sufficient:
        return END
    return "generate_query_variants"



builder = StateGraph(RAGState, config_schema=RAGConfiguration)
builder.add_node(generate_query_variants)
builder.add_edge("__start__", "generate_query_variants")
builder.add_conditional_edges(
    "generate_query_variants",
    check_query_variants
)
graph = builder.compile()
graph.name = "RagGraph"