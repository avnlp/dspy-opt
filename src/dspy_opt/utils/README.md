# Shared Utilities

Reusable DSPy modules shared across all dataset pipelines. Each module is a `dspy.Module` subclass with a `forward()` method, composable in DSPy's optimization framework.

## Modules

| Module | Description |
| :----- | :---------- |
| [QueryRewriter](query_rewriter.py) | Rewrites user queries to optimize retrieval effectiveness |
| [SubQueryGenerator](sub_query_generator.py) | Decomposes complex queries into focused sub-queries |
| [MetadataExtractor](metadata_extractor.py) | Extracts structured metadata from text using an LLM and JSON schema |
| [WeaviateRetriever](weaviate_retriever.py) | Hybrid search (vector + keyword) against Weaviate with metadata filtering |
| [Metrics](metrics.py) | Wraps DeepEval metrics into DSPy-compatible metric functions |

## QueryRewriter

Rewrites search queries by expanding with synonyms, clarifying ambiguous terms, removing conversational noise, and preserving key entities.

**Class:** `QueryRewriter(use_chain_of_thought: bool = True)`

**Methods:**

```python
def forward(self, query: str) -> dspy.Prediction:
    """Returns Prediction with `rewritten_query` (and `rationale` if using CoT)."""

def batch_rewrite(self, queries: List[str]) -> List[str]:
    """Rewrites multiple queries. Returns list of optimized query strings."""
```

**Example:**

```python
from dspy_opt.utils.query_rewriter import QueryRewriter

rewriter = QueryRewriter()
result = rewriter("cheap flights to Paris next week")
print(result.rewritten_query)
# "affordable flights Paris France departure date next 7 days"
```

## SubQueryGenerator

Decomposes complex queries into 2-5 targeted sub-queries for parallel retrieval. Automatically determines query complexity using heuristics.

**Class:** `SubQueryGenerator(min_subqueries: int = 2, max_subqueries: int = 5)`

**Methods:**

```python
def forward(self, query: str, num_subqueries: Optional[int] = None) -> dspy.Prediction:
    """Returns Prediction with `sub_queries` (List[str]) and `rationale`."""

def batch_generate(self, queries: List[str]) -> List[List[str]]:
    """Generates sub-queries for multiple queries."""
```

**Example:**

```python
from dspy_opt.utils.sub_query_generator import SubQueryGenerator

generator = SubQueryGenerator()
result = generator("Compare renewable energy adoption in Germany vs France since 2020")
print(result.sub_queries)
# ["Germany renewable energy economic impact 2020-2024",
#  "France renewable energy economic impact 2020-2024",
#  "Germany France renewable energy comparison 2020-2024"]
```

## MetadataExtractor

Extracts structured metadata from text using an LLM and a user-provided JSON schema. Only successfully extracted (non-null) fields are included in the result.

**Class:** `MetadataExtractor(extractor_llm: dspy.LM)`

**Methods:**

```python
def forward(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts metadata fields from text according to the schema.
    Returns dict of validated, non-null metadata fields."""

def transform_documents(
    self, documents: List[dspy.Example], schema: Dict[str, Any]
) -> List[dspy.Example]:
    """Applies metadata extraction to a list of DSPy examples."""
```

**Schema format:**

```python
schema = {
    "properties": {
        "title": {"type": "string", "description": "The main title"},
        "category": {"type": "string", "description": "Content category"},
        "year": {"type": "number", "description": "Publication year"},
    }
}
```

Supported types: `string`, `number`, `boolean`. String fields also support `enum`.

## WeaviateRetriever

Connects to a Weaviate vector database for hybrid search (vector + keyword) with metadata filtering.

**Class:**

```python
WeaviateRetriever(
    weaviate_url: Optional[str] = None,
    weaviate_api_key: Optional[str] = None,
    collection_name: str = "TriviaQA",
    top_k: int = 3,
    metadata_schema: Optional[dict[str, dict[str, object]]] = None,
)
```

**Methods:**

```python
def forward(
    self,
    query: str,
    query_embedding: Optional[np.ndarray] = None,
    top_k: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> dspy.Prediction:
    """Returns Prediction with `passages` (List[str])."""
```

## Metrics

Wraps [DeepEval](https://deepeval.com/) evaluation metrics into DSPy-compatible metric functions.

**Supported metrics:** Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall, Contextual Relevancy.

**Functions:**

```python
def create_metrics_function(metrics: List[BaseMetric]) -> Callable[[Any, Any], float]:
    """Creates a metric function returning an averaged float score.
    For use with MIPROv2, COPRO, BootstrapFewShot, and SIMBA optimizers."""

def create_gepa_metrics_function(
    metrics: List[BaseMetric],
) -> Callable[..., dspy.Prediction]:
    """Creates a metric function returning a Prediction with `score` and `feedback`.
    For use with the GEPA optimizer (requires feedback for reflection)."""
```

Both functions expect `gold` with `.question` and `.answer` attributes, and `pred` with `.answer` and `.retrieved_context` attributes.
