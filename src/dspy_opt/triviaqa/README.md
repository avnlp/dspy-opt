# TriviaQA Pipeline

RAG pipeline for the [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) dataset - a reading comprehension dataset containing question-answer-evidence triples authored by trivia enthusiasts with independently gathered evidence documents.

- **HuggingFace dataset:** [`mandarjoshi/trivia_qa`](https://huggingface.co/datasets/mandarjoshi/trivia_qa) (split: `test`)
- **Weaviate collection:** `TriviaQA`
- **Complexity type:** Trivia, factoid QA

## Pipeline Class

**`TriviaQARAG`** (defined in [`triviaqa_rag_module.py`](triviaqa_rag_module.py))

```python
class TriviaQARAG(dspy.Module):
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

TriviaQA uses typed metadata with an enum constraint on content type:

| Field | Type | Description |
| :---- | :--- | :---------- |
| `content_type` | string (enum: `review`, `lyrics`, `trivia`, `date_info`, `news`) | Category of the content |
| `primary_entity` | string | Main subject (airline, song, TV show, etc.) |
| `year` | number | Publication or relevant year |

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
| [`triviaqa_indexing_config.yml`](triviaqa_indexing_config.yml) | Indexing parameters (embedding model, metadata schema, collection name) |
| [`triviaqa_rag_mipro_config.yml`](triviaqa_rag_mipro_config.yml) | MIPROv2 optimizer parameters |
| [`triviaqa_rag_copro_config.yml`](triviaqa_rag_copro_config.yml) | COPRO optimizer parameters |
| [`triviaqa_rag_bootstrap_few_shot_config.yml`](triviaqa_rag_bootstrap_few_shot_config.yml) | BootstrapFewShot optimizer parameters |
| [`triviaqa_rag_simba_config.yml`](triviaqa_rag_simba_config.yml) | SIMBA optimizer parameters |
| [`triviaqa_rag_gepa_config.yml`](triviaqa_rag_gepa_config.yml) | GEPA optimizer parameters |
| [`triviaqa_rag_evaluation_config.yml`](triviaqa_rag_evaluation_config.yml) | Evaluation settings and DeepEval metric thresholds |

## Scripts

| Script | Description | Usage |
| :----- | :---------- | :---- |
| [`triviaqa_indexing.py`](triviaqa_indexing.py) | Load dataset from HuggingFace, extract metadata, embed, and store in Weaviate | `python triviaqa_indexing.py` |
| [`triviaqa_rag_module.py`](triviaqa_rag_module.py) | Pipeline definition (`TriviaQARAG` class) | Imported by optimizer and evaluation scripts |
| [`triviaqa_rag_mipro.py`](triviaqa_rag_mipro.py) | Run MIPROv2 optimization | `python triviaqa_rag_mipro.py` |
| [`triviaqa_rag_copro.py`](triviaqa_rag_copro.py) | Run COPRO optimization | `python triviaqa_rag_copro.py` |
| [`triviaqa_rag_bootstrap_few_shot.py`](triviaqa_rag_bootstrap_few_shot.py) | Run BootstrapFewShot optimization | `python triviaqa_rag_bootstrap_few_shot.py` |
| [`triviaqa_rag_simba.py`](triviaqa_rag_simba.py) | Run SIMBA optimization | `python triviaqa_rag_simba.py` |
| [`triviaqa_rag_gepa.py`](triviaqa_rag_gepa.py) | Run GEPA optimization | `python triviaqa_rag_gepa.py` |
| [`triviaqa_rag_evaluation.py`](triviaqa_rag_evaluation.py) | Evaluate optimized pipeline with DeepEval metrics | `python triviaqa_rag_evaluation.py` |

All scripts must be run from the `triviaqa/` directory for config file resolution:

```bash
cd src/dspy_opt/triviaqa
python triviaqa_rag_mipro.py
```
