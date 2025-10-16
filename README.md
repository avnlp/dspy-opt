# Advanced RAG pipelines optimized with DSPy

This repository implements optimized question-answering (QA) pipelines using DSPy (Differentiable Programming for Language Models) for multiple QA datasets. The project focuses on building efficient Retrieval-Augmented Generation (RAG) systems with advanced prompt and retrieval optimizations.

- **Datasets**:
  - **HotpotQA**: Multi-hop reasoning over Wikipedia articles
  - **FreshQA**: Time-sensitive and fact-based QA
  - **PubMedQA**: Biomedical question answering
  - **TriviaQA**: Open-domain QA with complex questions
  - **Wikipedia**: General knowledge QA

- **Advanced Pipeline Components**:
  - Optimized retrieval and generation pipelines
  - Context-aware prompt optimization
  - Metadata-enhanced document retrieval
  - Customizable evaluation framework

- **Integration**:
  - Weaviate vector database for efficient document retrieval
  - DeepEval for comprehensive model evaluation
  - Support for various embedding models

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/avnlp/test-dspy-opt.git
   cd test-dspy-opt
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   ```bash
   export WEAVIATE_URL=your_weaviate_url
   export WEAVIATE_API_KEY=your_weaviate_api_key
   ```

## Project Structure

```
test-dspy-opt/
├── dspy_opt/
│   ├── freshqa/               # FreshQA implementation
│   │   ├── evaluate_deepeval.py  # DeepEval integration
│   │   ├── indexing.py        # Weaviate document indexing
│   │   ├── metadata_extractor.py # Context enhancement
│   │   ├── prompt_optimization.py # DSPy prompt optimization
│   │   ├── prompts.py         # Prompt templates
│   │   └── run_deepeval.py    # Main execution script
│   ├── hotpotqa/              # HotpotQA implementation
│   ├── pubmedqa/              # PubMedQA implementation
│   ├── triviaqa/              # TriviaQA implementation
│   ├── wikipedia/             # Wikipedia QA implementation
│   ├── metrics.py             # Evaluation metrics
│   └── prompts.py             # Common prompt utilities
├── tests/                    # Test cases
└── requirements.txt          # Dependencies
```

### 1. Question-Answering Pipelines

Each dataset has a dedicated pipeline implementation with the following common components:

1. **Document Indexing**:
   - Processes and indexes documents into Weaviate
   - Generates embeddings using Sentence Transformers
   - Stores metadata for enhanced retrieval

2. **Retrieval-Augmented Generation (RAG)**:
   - Implements context-aware retrieval
   - Uses vector similarity for relevant document fetching
   - Supports metadata filtering

3. **Prompt Optimization**:
   - Utilizes DSPy's `BootstrapFewShotWithRandomSearch`
   - Optimizes few-shot examples for better performance
   - Adapts prompts based on dataset characteristics

### 2. Evaluation

The RAG pipelines use DeepEval for comprehensive evaluation with two key metrics:

1. **Faithfulness Metric**: Measures if the generated answer is factually consistent with the retrieved context
2. **Answer Relevancy Metric**: Evaluates how relevant the answer is to the question

These metrics are combined into a composite score for overall performance evaluation.

### 3. Weaviate Integration

- **Vector Database**: Stores document embeddings for efficient similarity search
- **Schema Management**: Custom schema for each dataset with metadata support
- **Query Processing**: Handles complex queries with metadata filtering
- **Performance**: Optimized for high-throughput retrieval

## Usage

### Running with FreshQA

1. Prepare the dataset:

   ```python
   from dspy_opt.freshqa.run_deepeval import preprocess_freshqa
   from datasets import load_dataset

   dataset = load_dataset("fresh_qa", "fresh_qa_w_docs")
   processed_data, passages = preprocess_freshqa(dataset['train'])
   ```

2. Initialize Weaviate and index documents:

   ```python
   from dspy_opt.freshqa.indexing import get_weaviate_client, create_schema, upload_documents
   from sentence_transformers import SentenceTransformer

   client = get_weaviate_client()
   class_name = "FreshQA"
   create_schema(client, class_name)
   embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
   upload_documents(client, class_name, passages, embedding_model)
   ```

3. Run the optimized pipeline:

   ```python
   from dspy_opt.freshqa.run_deepeval import main

   main()
   ```

### Evaluation

Evaluate model performance using DeepEval metrics:

```python
from dspy_opt.freshqa.evaluate_deepeval import evaluate_pipeline

results = evaluate_pipeline(pipeline, test_data)
print(f"Faithfulness: {results['faithfulness']}")
print(f"Answer Relevancy: {results['relevancy']}")
```
