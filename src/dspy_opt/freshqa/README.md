# FreshQA Pipeline

RAG pipeline for the [FreshQA (SealQA)](https://huggingface.co/datasets/vtllms/sealqa) dataset - a dynamic QA benchmark covering diverse question and answer types, including questions requiring world knowledge and questions with false premises that need to be debunked.

- **HuggingFace dataset:** [`vtllms/sealqa`](https://huggingface.co/datasets/vtllms/sealqa) (subset: `longseal`, split: `test`)
- **Weaviate collection:** `FreshQA`
- **Complexity type:** Single-hop, false-premise debunking

## Pipeline Class

**`FreshQARAG`** (defined in [`freshqa_rag_module.py`](freshqa_rag_module.py))

```python
class FreshQARAG(dspy.Module):
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
| [`freshqa_indexing_config.yml`](freshqa_indexing_config.yml) | Indexing parameters (embedding model, metadata schema, collection name) |
| [`freshqa_rag_mipro_config.yml`](freshqa_rag_mipro_config.yml) | MIPROv2 optimizer parameters (`max_bootstrapped_demos`, `max_labeled_demos`, `auto`) |
| [`freshqa_rag_copro_config.yml`](freshqa_rag_copro_config.yml) | COPRO optimizer parameters (`breadth`, `depth`, `init_temperature`) |
| [`freshqa_rag_bootstrap_few_shot_config.yml`](freshqa_rag_bootstrap_few_shot_config.yml) | BootstrapFewShot optimizer parameters (`max_bootstrapped_demos`, `max_rounds`) |
| [`freshqa_rag_simba_config.yml`](freshqa_rag_simba_config.yml) | SIMBA optimizer parameters (`bsize`, `num_candidates`, `max_steps`, `max_demos`) |
| [`freshqa_rag_gepa_config.yml`](freshqa_rag_gepa_config.yml) | GEPA optimizer parameters (`max_full_evals`, `reflection_minibatch_size`, `candidate_selection_strategy`) |
| [`freshqa_rag_evaluation_config.yml`](freshqa_rag_evaluation_config.yml) | Evaluation settings and DeepEval metric thresholds |

## Scripts

| Script | Description | Usage |
| :----- | :---------- | :---- |
| [`freshqa_indexing.py`](freshqa_indexing.py) | Load dataset from HuggingFace, extract metadata, embed, and store in Weaviate | `python freshqa_indexing.py` |
| [`freshqa_rag_module.py`](freshqa_rag_module.py) | Pipeline definition (`FreshQARAG` class) | Imported by optimizer and evaluation scripts |
| [`freshqa_rag_mipro.py`](freshqa_rag_mipro.py) | Run MIPROv2 optimization | `python freshqa_rag_mipro.py` |
| [`freshqa_rag_copro.py`](freshqa_rag_copro.py) | Run COPRO optimization | `python freshqa_rag_copro.py` |
| [`freshqa_rag_bootstrap_few_shot.py`](freshqa_rag_bootstrap_few_shot.py) | Run BootstrapFewShot optimization | `python freshqa_rag_bootstrap_few_shot.py` |
| [`freshqa_rag_simba.py`](freshqa_rag_simba.py) | Run SIMBA optimization | `python freshqa_rag_simba.py` |
| [`freshqa_rag_gepa.py`](freshqa_rag_gepa.py) | Run GEPA optimization | `python freshqa_rag_gepa.py` |
| [`freshqa_rag_evaluation.py`](freshqa_rag_evaluation.py) | Evaluate optimized pipeline with DeepEval metrics | `python freshqa_rag_evaluation.py` |

All scripts must be run from the `freshqa/` directory for config file resolution:

```bash
cd src/dspy_opt/freshqa
python freshqa_rag_mipro.py
```
