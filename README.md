Fine-tuned embedder https://huggingface.co/wylupek/au-blog-rag-embedder

## Configuration:
#### Loader Graph
* `index_name` - Pinecone vectorstore index name, if it does not exist, it will be created. The index name must match the embedding model dimensions.
* `embedding_model` - The general embedding model in provider/model_name format used throughout the project. It must be either a HuggingFace or OpenAI model, e.g., 'wylupek/au-blog-rag-embedder', 'openai/text-embedding-3-small'.
* `load_documents_batch_size` - The number of documents to load into the vector store in a single batch (single node execution).

#### RAG Graph
* `num_query_variants` - The number of alternative retrieval queries to generate.
* `query_variants_prompt` - The system prompt used for generating alternative queries for retrieval.
* `query_variants_model` - The OpenAI language model used for generating alternative queries. Must be an LLM model, e.g., 'gpt-4o-mini'.
* `top_k` - The number of documents retrieved from the vector store for each query.
* `threshold` - The cosine similarity threshold between the query and retrieved documents. Documents below this threshold are not considered for analysis.
* `filter_false` - If enabled, articles with a relevance decision marked as "False" will be filtered out from the retrieved results.
* `analysis_prompt` - The system prompt used for analyzing the selected article.
* `analysis_model` - The OpenAI language model used for analyzing selected articles, e.g., 'gpt-4o'.
* `result_decision_prompt` - The system prompt used for generating a decision on whether a retrieved article is relevant to user's question.
* `result_summary_prompt` - The system prompt used for generating a summary of the retrieved article.
* `result_analysis_prompt` - The system prompt used for analyzing how the retrieved article is relevant to the user's question.
