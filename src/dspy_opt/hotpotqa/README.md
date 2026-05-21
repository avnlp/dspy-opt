# HotpotQA Pipeline

RAG pipeline for the [HotpotQA](https://hotpotqa.github.io/) dataset - a question answering dataset featuring natural, multi-hop questions with strong supervision for supporting facts.

- **HuggingFace dataset:** [`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa) (subset: `distractor`, split: `test`)
- **Weaviate collection:** `HotpotQA`
- **Complexity type:** Multi-hop reasoning

## Pipeline Class

**`HotpotQARAG`** (defined in [`hotpotqa_rag_module.py`](hotpotqa_rag_module.py))

```python
class HotpotQARAG(dspy.Module):
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
| [`hotpotqa_indexing_config.yml`](hotpotqa_indexing_config.yml) | Indexing parameters (embedding model, metadata schema, collection name) |
| [`hotpotqa_rag_mipro_config.yml`](hotpotqa_rag_mipro_config.yml) | MIPROv2 optimizer parameters |
| [`hotpotqa_rag_copro_config.yml`](hotpotqa_rag_copro_config.yml) | COPRO optimizer parameters |
| [`hotpotqa_rag_bootstrap_few_shot_config.yml`](hotpotqa_rag_bootstrap_few_shot_config.yml) | BootstrapFewShot optimizer parameters |
| [`hotpotqa_rag_simba_config.yml`](hotpotqa_rag_simba_config.yml) | SIMBA optimizer parameters |
| [`hotpotqa_rag_gepa_config.yml`](hotpotqa_rag_gepa_config.yml) | GEPA optimizer parameters |
| [`hotpotqa_rag_evaluation_config.yml`](hotpotqa_rag_evaluation_config.yml) | Evaluation settings and DeepEval metric thresholds |

## Scripts

| Script | Description | Usage |
| :----- | :---------- | :---- |
| [`hotpotqa_indexing.py`](hotpotqa_indexing.py) | Load dataset from HuggingFace, extract metadata, embed, and store in Weaviate | `python hotpotqa_indexing.py` |
| [`hotpotqa_rag_module.py`](hotpotqa_rag_module.py) | Pipeline definition (`HotpotQARAG` class) | Imported by optimizer and evaluation scripts |
| [`hotpotqa_rag_mipro.py`](hotpotqa_rag_mipro.py) | Run MIPROv2 optimization | `python hotpotqa_rag_mipro.py` |
| [`hotpotqa_rag_copro.py`](hotpotqa_rag_copro.py) | Run COPRO optimization | `python hotpotqa_rag_copro.py` |
| [`hotpotqa_rag_bootstrap_few_shot.py`](hotpotqa_rag_bootstrap_few_shot.py) | Run BootstrapFewShot optimization | `python hotpotqa_rag_bootstrap_few_shot.py` |
| [`hotpotqa_rag_simba.py`](hotpotqa_rag_simba.py) | Run SIMBA optimization | `python hotpotqa_rag_simba.py` |
| [`hotpotqa_rag_gepa.py`](hotpotqa_rag_gepa.py) | Run GEPA optimization | `python hotpotqa_rag_gepa.py` |
| [`hotpotqa_rag_evaluation.py`](hotpotqa_rag_evaluation.py) | Evaluate optimized pipeline with DeepEval metrics | `python hotpotqa_rag_evaluation.py` |

All scripts must be run from the `hotpotqa/` directory for config file resolution:

```bash
cd src/dspy_opt/hotpotqa
python hotpotqa_rag_mipro.py
```
