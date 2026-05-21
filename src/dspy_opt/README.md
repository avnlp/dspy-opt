# dspy_opt

Root package for DSPy-based RAG pipeline optimization. Contains shared utility modules and per-dataset pipeline implementations.

## Package Structure

```text
dspy_opt/
├── utils/          # Shared reusable DSPy modules
├── freshqa/        # FreshQA (SealQA) dataset pipeline
├── hotpotqa/       # HotpotQA dataset pipeline
├── pubmedqa/       # PubMedQA dataset pipeline
├── triviaqa/       # TriviaQA dataset pipeline
└── wikipedia/      # Wikipedia dataset pipeline
```

## Submodules

| Module | Description | Documentation |
| :----- | :---------- | :------------ |
| [`utils/`](utils/) | Shared components: Query rewriting, Sub-query generation, Metadata extraction, Weaviate retrieval, DeepEval metrics | [utils/README.md](utils/README.md) |
| [`freshqa/`](freshqa/) | FreshQA/SealQA pipeline - Single-hop QA and False-premise debunking | [freshqa/README.md](freshqa/README.md) |
| [`hotpotqa/`](hotpotqa/) | HotpotQA pipeline - Multi-hop reasoning questions | [hotpotqa/README.md](hotpotqa/README.md) |
| [`pubmedqa/`](pubmedqa/) | PubMedQA pipeline - Biomedical QA with rich metadata filtering | [pubmedqa/README.md](pubmedqa/README.md) |
| [`triviaqa/`](triviaqa/) | TriviaQA pipeline - Trivia and factoid QA with typed metadata | [triviaqa/README.md](triviaqa/README.md) |
| [`wikipedia/`](wikipedia/) | Wikipedia pipeline - General knowledge QA | [wikipedia/README.md](wikipedia/README.md) |

## How Each Pipeline Works

Every dataset follows the same 5-stage architecture, composed from the shared `utils/` modules:

1. **QueryRewriter** - rewrites the user query for search optimization
2. **SubQueryGenerator** - decomposes complex queries into parallel sub-queries
3. **MetadataExtractor** - extracts structured metadata for filtering
4. **WeaviateRetriever** - hybrid search (vector + keyword) with metadata filtering
5. **dspy.ChainOfThought** - generates the final answer from aggregated passages

DSPy optimizers (MIPROv2, COPRO, BootstrapFewShot, SIMBA, GEPA) automatically tune prompts and few-shot examples, evaluated using DeepEval metrics.

## Adding a New Dataset

1. Create a new directory: `src/dspy_opt/<new_dataset>/`
2. Add an `__init__.py`
3. Create the following files following the existing pattern:
   - `<dataset>_indexing.py` + `<dataset>_indexing_config.yml` - indexing pipeline
   - `<dataset>_rag_module.py` - pipeline class (subclass `dspy.Module`)
   - `<dataset>_rag_<optimizer>.py` + `<dataset>_rag_<optimizer>_config.yml` - per-optimizer scripts
   - `<dataset>_rag_evaluation.py` + `<dataset>_rag_evaluation_config.yml` - evaluation script
4. Compose the shared utilities from `utils/` in your pipeline module
5. Add a `README.md` following the [freshqa/README.md](freshqa/README.md) template
