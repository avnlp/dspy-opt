# Advanced RAG Pipeline Optimization with DSPy

This repository implements several Retrieval-Augmented Generation (RAG) pipelines on diverse question answering datasets using the [DSPy](https://dspy.ai/) framework. The prompts and few-shot examples in the DSPy modules are optimized using the MIPRO, COPRO, BootstrapFewShot optimizers and DeepEval metrics.

The RAG pipelines are built using:

- [**DSPy**](https://dspy.ai/) for modular pipeline design and optimization.
- [**Weaviate**](https://weaviate.io/) vector database for hybrid search and retrieval.
- [**DeepEval**](https://deepeval.com/) for comprehensive evaluation metrics.
- [**Confident AI**](https://www.confident-ai.com/) for logging of metrics during optimization.

Each pipeline is configured through YAML files that allow for flexible customization of language models, embedding models, and optimizer hyperparameters.

## Datasets

The project includes implementations for several question answering datasets:

- [**FreshQA (SealQA)**](https://huggingface.co/datasets/vtllms/sealqa): FreshQA is a dynamic QA benchmark that covers a diverse range of question and answer types, with questions that require world knowledge as well as questions with false premises that need to be debunked. SealQA builds on FreshQA with a stronger focus on reasoning.
- [**HotpotQA**](https://hotpotqa.github.io/): HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts.
- [**PubMedQA**](https://pubmedqa.github.io/): Biomedical question answering dataset based on PubMed abstracts.
- [**TriviaQA**](https://nlp.cs.washington.edu/triviaqa/): TriviaQA is a reading comprehension dataset containing question-answer-evidence triples authored by trivia enthusiasts and independently gathered evidence documents, providing high quality distant supervision for answering the questions.
- [**Wikipedia**](https://huggingface.co/datasets/wikipedia): Wikipedia is a large-scale dataset of cleaned articles from all language editions of Wikipedia, sourced from the official Wikipedia dumps.

## Pipeline Architecture

<img src="plots/pipeline.png" alt="DSPy RAG Pipeline" align="middle" height=950>

Each pipeline follows a consistent architecture with the following components:

- **Query Rewriting**: The initial `question` is passed to the `QueryRewriter` to generate a search-optimized query by expanding it with synonyms, clarifying ambiguous terms, and removing conversational noise.

- **Sub-Query Generation**: The rewritten query is then passed to the `SubQueryGenerator` to decompose it into multiple, more specific sub-queries. This breaks down multi-faceted questions into smaller, self-contained queries that can be executed in parallel, improving retrieval coverage.

- **Metadata Extraction**: The `MetadataExtractor` uses an LLM to parse both the rewritten query and each sub-query to extract structured metadata based on a predefined JSON schema. This structured metadata can then be used for filtering in the retriever to improve retrieval precision.

- **Document Retrieval**: The `WeaviateRetriever` is called for the main query and each sub-query, using the extracted metadata for filtering. It performs hybrid search combining vector search with keyword-based filtering. The results are aggregated into a single list of passages.

- **Answer Generation**: The unique, retrieved passages are fed into a `dspy.ChainOfThought` module to generate a final answer and the reasoning behind it.

- **Optimization**: DSPy optimizers (MIPROv2, COPRO, BootstrapFewShot) automatically tune prompts and select few-shot examples by exploring the space of possible configurations and evaluating them using DeepEval metrics.

- **Logging**: Confident AI is used for logging of metrics during optimization.

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management. First, ensure uv is installed:

```bash
# Install uv (if not already installed)
pip install uv
```

Then install the project dependencies:

```bash
# Install dependencies with all extras and dev dependencies
uv sync --all-extras --dev

# Activate the virtual environment
source .venv/bin/activate
```

## Usage

### Environment Setup

Create a `.env` file in the project root with the required environment variables:

```env
WEAVIATE_URL=your_weaviate_cluster_url
WEAVIATE_API_KEY=your_weaviate_api_key
GROQ_API_KEY=your_groq_api_key
```

For tracing of evaluation runs:

Create a `.env.local` file in the project root and add your Confident AI API key:

```env
API_KEY=CONFIDENT_API_KEY
```

### Indexing

Each dataset module includes an indexing script to process and store documents in the vector database. The indexing process:

1. Loads the dataset from Hugging Face.
2. Extracts metadata from each document using an LLM based on the metadata schema defined in the config file.
3. Generates vector embeddings using SentenceTransformer model.
4. Stores documents, embeddings, and metadata in Weaviate.

Example for FreshQA:

```bash
cd src/dspy_opt/freshqa
python freshqa_indexing.py
```

### Evaluation

Each dataset module includes an evaluation script to test the pipeline performance. The evaluation script:

1. Loads the pipeline from the saved state.
2. Runs predictions on the test dataset.
3. Evaluates using DeepEval metrics configured in the YAML file.
4. Reports aggregated scores and individual metric results.

Example for FreshQA:

```bash
cd src/dspy_opt/freshqa
python freshqa_rag_evaluation.py
```

### Pipeline Optimization

Each dataset module includes optimization scripts for different DSPy optimizers. The optimization process:

1. Loads the configuration from the YAML file (e.g., `freshqa_rag_mipro_config.yml`).
2. Initializes all DSPy modules (QueryRewriter, SubQueryGenerator, MetadataExtractor, WeaviateRetriever).
3. Loads the training and evaluation datasets.
4. Runs the optimizer to compile the pipeline with optimized prompts and few-shot examples.
5. Evaluates the optimized pipeline using DeepEval metrics.

Example for FreshQA RAG pipeline optimized using the MIPROv2 optimizer:

```bash
cd src/dspy_opt/freshqa
python freshqa_rag_mipro.py
```

## Components

### Query Rewriter

The QueryRewriter optimizes user queries for better retrieval performance.

- Rewrites queries to be more effective for search engines.
- Expands queries with relevant synonyms and concepts.
- Clarifies ambiguous terms and removes conversational noise.
- Maintains conciseness while preserving key entities and constraints.

### Sub-Query Generator

The SubQueryGenerator decomposes complex user queries into simpler, more focused sub-queries.

- Breaks down multi-faceted questions into smaller queries.
- Each sub-query addresses a distinct aspect of the original query.
- Sub-queries are self-contained for parallel search execution.
- Improves retrieval coverage for complex information needs.

### Metadata Extractor

The MetadataExtractor extracts structured metadata from text using a language model and a user-specified JSON schema.

- Uses LLMs with structured-output generation for metadata extraction.
- Dynamically converts JSON schema into validation structures.
- Only includes successfully extracted (non-null) fields in results.
- Extracted metadata is used for filtering during retrieval.

### Weaviate Retriever

The WeaviateRetriever connects to a Weaviate vector database for document retrieval.

- Performs hybrid search combining vector search with keyword-based filtering.
- Filters results based on extracted metadata.

### Metrics

The Metrics module integrates [DeepEval](https://deepeval.com/) evaluation metrics into the DSPy optimization framework.

- Creates metric functions compatible with DSPy optimizers.
- Evaluates pipeline performance using multiple metrics:
  - **Answer Relevancy**: Measures how relevant the answer is to the question.
  - **Faithfulness**: Ensures the answer is grounded in the retrieved context.
  - **Contextual Precision**: Evaluates precision of retrieved context.
  - **Contextual Recall**: Measures recall of retrieved context.
  - **Contextual Relevancy**: Assesses overall relevance of retrieved passages.
- Aggregates scores across metrics for optimization objectives.
- Supports async evaluation with configurable throttling.

## Project Structure

```text
src/dspy_opt/
├── utils/                          # Shared reusable components
│   ├── query_rewriter.py           # Query optimization module
│   ├── sub_query_generator.py      # Multi-query decomposition
│   ├── metadata_extractor.py       # Structured metadata extraction
│   ├── weaviate_retriever.py       # Hybrid Search retriever
│   └── metrics.py                  # DeepEval Metrics Integration
│
├── freshqa/                        # FreshQA dataset pipelines
│   ├── freshqa_indexing.py         # Index documents to Weaviate
│   ├── freshqa_indexing_config.yml
│   ├── freshqa_rag_module.py       # Complete RAG pipeline definition
│   ├── freshqa_rag_mipro.py        # MIPRO Optimization
│   ├── freshqa_rag_mipro_config.yml
│   ├── freshqa_rag_copro.py        # COPRO Optimization
│   ├── freshqa_rag_copro_config.yml
│   ├── freshqa_rag_bootstrap_few_shot.py
│   ├── freshqa_rag_bootstrap_few_shot_config.yml
│   └── freshqa_rag_evaluation.py   # Evaluate optimized pipeline
│
├── hotpotqa/                       # HotpotQA dataset pipelines
│   └── ... (similar structure)
│
├── triviaqa/                       # TriviaQA dataset pipelines
│   └── ... (similar structure)
│
├── pubmedqa/                       # PubMedQA dataset pipelines
│   └── ... (similar structure)
│
└── wikipedia/                      # Wikipedia dataset pipelines
    └── ... (similar structure)
```

## Contributing

Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
