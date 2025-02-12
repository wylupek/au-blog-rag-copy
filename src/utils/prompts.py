QUERY_VARIANTS_PROMPT = """You are an AI language model assistant. 
Your task is to generate exactly {num_variants} alternative versions of the given user question to help retrieve relevant documents from a vector database. 
Provide only these alternative questions, each on a new line without numbering. Do not repeat any of the following previously generated queries. 
Original question: {question}
Previous queries: {previous_queries}"""

ANALYSIS_PROMPT="""You are an AI language model tasked with generating concise and precise responses to user queries based on the provided article content.
Guidelines:
- Limit the entire response to a maximum of 150 words.
- Avoid extra formatting, unnecessary details, or information outside the article's scope.
- Write in a clear, user-friendly tone suitable for non-technical readers.

<question>{query}</question>
<article>{context}</article>"""

RESULT_DECISION_PROMPT="A single-word answer (True or False) indicating whether the article is relevant to the user's query and might be used as a showcase for potential client."

RESULT_SUMMARY_PROMPT="A concise summary of the article in plain, non-technical language."

RESULT_ANALYSIS_PROMPT="A direct, clear response to the user's question, strictly based on the article content, written in simple terms."