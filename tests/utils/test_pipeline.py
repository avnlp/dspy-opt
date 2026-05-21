"""Tests for shared DSPy utility components."""

import contextlib
import json
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import pytest_mock

from dspy_opt.utils.metadata_extractor import MetadataExtractor
from dspy_opt.utils.metrics import create_gepa_metrics_function, create_metrics_function
from dspy_opt.utils.query_rewriter import QueryRewriter
from dspy_opt.utils.sub_query_generator import SubQueryGenerator
from dspy_opt.utils.weaviate_retriever import WeaviateRetriever
from tests.helpers import DummyEvaluationResult, DummyPrediction


class TestPipeline:
    """Utility pipeline tests."""

    def test_query_rewriter_batch_rewrite_uses_forward(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        """Ensure batch rewriting delegates to the forward method."""
        rewriter = QueryRewriter(use_chain_of_thought=False)
        forward_mock = mocker.Mock(
            return_value=DummyPrediction(rewritten_query="rewrite")
        )
        mocker.patch.object(rewriter, "forward", forward_mock)

        result = rewriter.batch_rewrite(["a", "b"])

        assert result == ["rewrite", "rewrite"]
        assert forward_mock.call_count == 2

    def test_sub_query_generator_parses_json(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        """Verify JSON sub-query payloads are parsed as lists."""
        generator = SubQueryGenerator(min_subqueries=2, max_subqueries=3)
        mocker.patch.object(
            generator,
            "generator",
            mocker.Mock(
                return_value=DummyPrediction(
                    sub_queries=json.dumps(["q1", "q2", "q3"]),
                    rationale="ok",
                )
            ),
        )

        result = generator.forward("compare apples and oranges", num_subqueries=3)

        assert result.sub_queries == ["q1", "q2", "q3"]
        assert result.rationale == "ok"

    def test_sub_query_generator_fallback_on_invalid_json(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        """Check that invalid JSON yields a fallback decomposition."""
        generator = SubQueryGenerator(min_subqueries=2, max_subqueries=5)
        mocker.patch.object(
            generator,
            "generator",
            mocker.Mock(return_value=DummyPrediction(sub_queries="not-json")),
        )

        result = generator.forward("what is quantum computing", num_subqueries=3)

        assert len(result.sub_queries) == 1
        assert "Decomposition error" in result.rationale

    def test_sub_query_generator_complexity_scale(self) -> None:
        """Confirm that more complex prompts scale sub-query counts."""
        generator = SubQueryGenerator(min_subqueries=2, max_subqueries=5)

        assert generator._determine_complexity("What is AI?") == 2
        assert (
            generator._determine_complexity(
                "Compare AI vs ML and list differences, pros, cons"
            )
            >= 3
        )

    def test_metadata_extractor_filters_schema_fields(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        """Ensure extractor returns only schema-defined metadata."""
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "year": {"type": "number"},
            },
        }
        extractor = MetadataExtractor(extractor_llm=mocker.Mock())
        mocker.patch(
            "dspy_opt.utils.metadata_extractor.dspy.context",
            return_value=contextlib.nullcontext(),
        )
        mocker.patch(
            "dspy_opt.utils.metadata_extractor.dspy.Predict",
            return_value=mocker.Mock(
                return_value=DummyPrediction(metadata=json.dumps({"category": "news"}))
            ),
        )

        result = extractor("text", schema)

        assert result == {"category": "news"}

    def test_metadata_extractor_invalid_schema_raises(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        """Assert that missing schema properties raise a ValueError."""
        extractor = MetadataExtractor(extractor_llm=mocker.Mock())
        with pytest.raises(ValueError):
            extractor("text", {"type": "object"})

    def test_weaviate_retriever_metadata_filter_skips_unknown(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        """Skip unknown metadata keys while building Weaviate filters."""
        wvc_module = mocker.Mock()
        wvc_module.query.Filter.and_ = mocker.Mock(return_value="combined")
        wvc_module.query.Filter.by_property = mocker.Mock(
            return_value=mocker.Mock(like=mocker.Mock(return_value="filter"))
        )
        wvc_module.init.Auth.api_key = mocker.Mock(return_value=mocker.Mock())
        mocker.patch("dspy_opt.utils.weaviate_retriever.wvc", wvc_module)
        client = mocker.Mock()
        client.is_ready.return_value = True
        client.collections.use.return_value = mocker.Mock()
        mocker.patch(
            "dspy_opt.utils.weaviate_retriever.weaviate.connect_to_weaviate_cloud",
            return_value=client,
        )
        retriever = WeaviateRetriever(
            weaviate_url="http://example",
            weaviate_api_key="key",
            collection_name="Collection",
            metadata_schema={"year": int},
        )

        result = retriever.metadata_to_weaviate_filter(
            {"unknown": "skip", "year": "2024"}
        )

        assert result == "combined"

    def test_weaviate_retriever_returns_empty_for_blank_query(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        """Avoid Weaviate calls when the query is empty or whitespace."""
        wvc_module = mocker.Mock()
        wvc_module.init.Auth.api_key = mocker.Mock(return_value=mocker.Mock())
        mocker.patch("dspy_opt.utils.weaviate_retriever.wvc", wvc_module)
        client = mocker.Mock()
        client.is_ready.return_value = True
        client.collections.use.return_value = mocker.Mock()
        mocker.patch(
            "dspy_opt.utils.weaviate_retriever.weaviate.connect_to_weaviate_cloud",
            return_value=client,
        )
        retriever = WeaviateRetriever(
            weaviate_url="http://example",
            weaviate_api_key="key",
            collection_name="Collection",
        )

        result = retriever.forward(" ")

        assert result.passages == []

    def test_weaviate_retriever_extracts_passages(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        """Confirm retrieved passages are filtered for non-empty text."""
        wvc_module = mocker.Mock()
        wvc_module.query.MetadataQuery = mocker.Mock(return_value="metadata")
        wvc_module.query.Filter.and_ = mocker.Mock(return_value=None)
        wvc_module.init.Auth.api_key = mocker.Mock(return_value=mocker.Mock())
        mocker.patch("dspy_opt.utils.weaviate_retriever.wvc", wvc_module)
        client = mocker.Mock()
        client.is_ready.return_value = True
        response = SimpleNamespace(
            objects=[
                SimpleNamespace(properties={"document_text": "doc"}),
                SimpleNamespace(properties={"document_text": ""}),
            ]
        )
        collection = mocker.Mock()
        collection.query.hybrid.return_value = response
        client.collections.use.return_value = collection
        mocker.patch(
            "dspy_opt.utils.weaviate_retriever.weaviate.connect_to_weaviate_cloud",
            return_value=client,
        )
        retriever = WeaviateRetriever(
            weaviate_url="http://example",
            weaviate_api_key="key",
            collection_name="Collection",
        )

        result = retriever.forward("query")

        assert result.passages == ["doc"]

    def test_metrics_function_averages_scores(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        """Check that metric scores are averaged into a scalar value."""
        mocker.patch(
            "dspy_opt.utils.metrics.evaluate",
            return_value=DummyEvaluationResult({"m1": 0.6, "m2": 0.4}),
        )
        metrics_fn = create_metrics_function([mocker.Mock(), mocker.Mock()])
        gold = SimpleNamespace(question="q", answer="a")
        pred = SimpleNamespace(answer="p", retrieved_context=["ctx"])

        score = metrics_fn(gold, pred)

        assert score == 0.5

    def test_gepa_metrics_function_returns_prediction(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        """Ensure GEPA metrics return a prediction wrapper with feedback."""
        mocker.patch(
            "dspy_opt.utils.metrics.evaluate",
            return_value=DummyEvaluationResult({"m1": 0.8}),
        )
        metrics_fn = create_gepa_metrics_function([mocker.Mock()])
        gold = SimpleNamespace(question="q", answer="a")
        pred = SimpleNamespace(answer="p", retrieved_context=["ctx"])

        result = metrics_fn(gold, pred)

        assert result.score == 0.8
        assert "m1" in result.feedback
