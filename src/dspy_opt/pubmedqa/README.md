# PubMedQA Pipeline

RAG pipeline for the [PubMedQA](https://pubmedqa.github.io/) dataset - a biomedical question answering dataset based on PubMed abstracts, requiring domain-specific reasoning over scientific literature.

- **HuggingFace dataset:** [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA) (subset: `pqa_artificial`, split: `test`)
- **Weaviate collection:** `PubMedQA`
- **Complexity type:** Biomedical domain QA

## Pipeline Class

**`PubMedQARAG`** (defined in [`pubmedqa_rag_module.py`](pubmedqa_rag_module.py))

```python
class PubMedQARAG(dspy.Module):
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

PubMedQA uses a rich biomedical metadata schema for precise retrieval filtering:

| Field | Type | Description |
| :---- | :--- | :---------- |
| `diseases_conditions` | string | Diseases, disorders, or medical conditions mentioned |
| `biological_entities` | string | Genes, proteins, cells, molecules, or biological pathways studied |
| `species` | string | Species involved in the study (e.g., human, mouse, rat) |
| `study_type` | string | Type of research study design |
| `main_findings` | string | Key results or conclusions from the study |
| `effect_direction` | string | Direction of main effects reported |

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
| [`pubmedqa_indexing_config.yml`](pubmedqa_indexing_config.yml) | Indexing parameters (embedding model, metadata schema, collection name) |
| [`pubmedqa_rag_mipro_config.yml`](pubmedqa_rag_mipro_config.yml) | MIPROv2 optimizer parameters |
| [`pubmedqa_rag_copro_config.yml`](pubmedqa_rag_copro_config.yml) | COPRO optimizer parameters |
| [`pubmedqa_rag_bootstrap_few_shot_config.yml`](pubmedqa_rag_bootstrap_few_shot_config.yml) | BootstrapFewShot optimizer parameters |
| [`pubmedqa_rag_simba_config.yml`](pubmedqa_rag_simba_config.yml) | SIMBA optimizer parameters |
| [`pubmedqa_rag_gepa_config.yml`](pubmedqa_rag_gepa_config.yml) | GEPA optimizer parameters |
| [`pubmedqa_rag_evaluation_config.yml`](pubmedqa_rag_evaluation_config.yml) | Evaluation settings and DeepEval metric thresholds |

## Scripts

| Script | Description | Usage |
| :----- | :---------- | :---- |
| [`pubmedqa_indexing.py`](pubmedqa_indexing.py) | Load dataset from HuggingFace, extract metadata, embed, and store in Weaviate | `python pubmedqa_indexing.py` |
| [`pubmedqa_rag_module.py`](pubmedqa_rag_module.py) | Pipeline definition (`PubMedQARAG` class) | Imported by optimizer and evaluation scripts |
| [`pubmedqa_rag_mipro.py`](pubmedqa_rag_mipro.py) | Run MIPROv2 optimization | `python pubmedqa_rag_mipro.py` |
| [`pubmedqa_rag_copro.py`](pubmedqa_rag_copro.py) | Run COPRO optimization | `python pubmedqa_rag_copro.py` |
| [`pubmedqa_rag_bootstrap_few_shot.py`](pubmedqa_rag_bootstrap_few_shot.py) | Run BootstrapFewShot optimization | `python pubmedqa_rag_bootstrap_few_shot.py` |
| [`pubmedqa_rag_simba.py`](pubmedqa_rag_simba.py) | Run SIMBA optimization | `python pubmedqa_rag_simba.py` |
| [`pubmedqa_rag_gepa.py`](pubmedqa_rag_gepa.py) | Run GEPA optimization | `python pubmedqa_rag_gepa.py` |
| [`pubmedqa_rag_evaluation.py`](pubmedqa_rag_evaluation.py) | Evaluate optimized pipeline with DeepEval metrics | `python pubmedqa_rag_evaluation.py` |

All scripts must be run from the `pubmedqa/` directory for config file resolution:

```bash
cd src/dspy_opt/pubmedqa
python pubmedqa_rag_mipro.py
```
