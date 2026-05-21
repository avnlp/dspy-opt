# Wikipedia Pipeline

RAG pipeline for the [Wikipedia](https://huggingface.co/datasets/wikipedia) dataset - a large-scale dataset of cleaned articles from all language editions of Wikipedia. Uses [WikiQA](https://huggingface.co/datasets/microsoft/wiki_qa) for question-answer pairs during optimization.

- **HuggingFace dataset (indexing):** [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (subset: `20231101.en`, split: `train`)
- **HuggingFace dataset (QA):** [`microsoft/wiki_qa`](https://huggingface.co/datasets/microsoft/wiki_qa) (split: `train`, test_size: `0.1`)
- **Weaviate collection:** `Wikipedia`
- **Complexity type:** General knowledge QA

## Pipeline Class

**`WikipediaRAG`** (defined in [`wikipedia_rag_module.py`](wikipedia_rag_module.py))

```python
class WikipediaRAG(dspy.Module):
    def __init__(
        self,
        query_rewriter: QueryRewriter,
        sub_query_generator: SubQueryGenerator,
        metadata_extractor: MetadataExtractor,
        metadata_schema: Dict[str, Any],
        weaviate_retriever: WeaviateRetriever,
        embedding_model: SentenceTransformer,
        top_k: int = 3,
    ): ...

    def forward(self, question: str) -> dspy.Prediction:
        """Returns Prediction with: question, rewritten_query, sub_queries,
        retrieved_context, answer, reasoning."""
```

## Metadata Schema

| Field | Type | Description |
| :---- | :--- | :---------- |
| `title` | string | The main title or name of the subject |
| `category` | string | Primary category or type of content |

## Models

| Role | Model |
| :--- | :---- |
| Answer LLM | `groq/qwen3-32b` |
| Extractor LLM | `groq/llama-3.3-70b-versatile` |
| Embedding | `Qwen/Qwen3-Embedding-0.6B` |
| Evaluator LLM | `groq/qwen3-32b` |

## Configuration Files

| File | Description |
| :--- | :---------- |
| [`wikipedia_indexing_config.yml`](wikipedia_indexing_config.yml) | Indexing parameters (embedding model, metadata schema, collection name) |
| [`wikipedia_rag_mipro_config.yml`](wikipedia_rag_mipro_config.yml) | MIPROv2 optimizer parameters |
| [`wikipedia_rag_copro_config.yml`](wikipedia_rag_copro_config.yml) | COPRO optimizer parameters |
| [`wikipedia_rag_bootstrap_few_shot_config.yml`](wikipedia_rag_bootstrap_few_shot_config.yml) | BootstrapFewShot optimizer parameters |
| [`wikipedia_rag_simba_config.yml`](wikipedia_rag_simba_config.yml) | SIMBA optimizer parameters |
| [`wikipedia_rag_gepa_config.yml`](wikipedia_rag_gepa_config.yml) | GEPA optimizer parameters |
| [`wikipedia_rag_evaluation_config.yml`](wikipedia_rag_evaluation_config.yml) | Evaluation settings and DeepEval metric thresholds |

## Scripts

| Script | Description | Usage |
| :----- | :---------- | :---- |
| [`wikipedia_indexing.py`](wikipedia_indexing.py) | Load dataset from HuggingFace, extract metadata, embed, and store in Weaviate | `python wikipedia_indexing.py` |
| [`wikipedia_rag_module.py`](wikipedia_rag_module.py) | Pipeline definition (`WikipediaRAG` class) | Imported by optimizer and evaluation scripts |
| [`wikipedia_rag_mipro.py`](wikipedia_rag_mipro.py) | Run MIPROv2 optimization | `python wikipedia_rag_mipro.py` |
| [`wikipedia_rag_copro.py`](wikipedia_rag_copro.py) | Run COPRO optimization | `python wikipedia_rag_copro.py` |
| [`wikipedia_rag_bootstrap_few_shot.py`](wikipedia_rag_bootstrap_few_shot.py) | Run BootstrapFewShot optimization | `python wikipedia_rag_bootstrap_few_shot.py` |
| [`wikipedia_rag_simba.py`](wikipedia_rag_simba.py) | Run SIMBA optimization | `python wikipedia_rag_simba.py` |
| [`wikipedia_rag_gepa.py`](wikipedia_rag_gepa.py) | Run GEPA optimization | `python wikipedia_rag_gepa.py` |
| [`wikipedia_rag_evaluation.py`](wikipedia_rag_evaluation.py) | Evaluate optimized pipeline with DeepEval metrics | `python wikipedia_rag_evaluation.py` |

All scripts must be run from the `wikipedia/` directory for config file resolution:

```bash
cd src/dspy_opt/wikipedia
python wikipedia_rag_mipro.py
```
