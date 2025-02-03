QUERY_VARIANTS_PROMPT = """You are an AI language model assistant. 
Your task is to generate exactly {num_variants} alternative versions of the given user question to help retrieve relevant documents from a vector database. 
Provide only these alternative questions, each on a new line without numbering. Do not repeat any of the following previously generated queries. 
Original question: {question}
Previous queries: {previous_queries}"""