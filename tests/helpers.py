"""Shared test doubles for DSPy pipelines."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np


@dataclass
class DummyPrediction:
    """Lightweight prediction object for DSPy-like return values."""

    rewritten_query: str | None = None
    sub_queries: List[str] | None = None
    passages: List[str] | None = None
    answer: str | None = None
    reasoning: str | None = None
    rationale: str | None = None
    metadata: str | None = None
    score: float | None = None
    feedback: str | None = None


@dataclass
class DummyExample:
    """Minimal dspy.Example replacement for tests."""

    question: str
    answer: str

    def with_inputs(self, *_args: str) -> "DummyExample":
        """Return self to mirror dspy.Example chaining behavior."""
        return self


class DummyEmbeddingModel:
    """SentenceTransformer stand-in that returns deterministic vectors."""

    def encode(self, _text: Any, **_kwargs: Any) -> np.ndarray:
        """Return a stable embedding for repeatable test assertions."""
        return np.array([0.1, 0.2, 0.3], dtype=float)


class DummyOptimizer:
    """Optimizer stand-in that records compile inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.compiled_with = None

    def compile(self, pipeline: Any, trainset: Iterable[Any]) -> "DummyOptimizer":
        """Capture compile inputs while preserving fluent usage."""
        self.compiled_with = {
            "pipeline": pipeline,
            "trainset": list(trainset),
        }
        return self

    def save(self, *_args: str, **_kwargs: Any) -> None:
        """No-op save to satisfy optimizer API in tests."""
        return


class DummyEvaluate:
    """Evaluator stand-in that returns a fixed result and captures inputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, _pipeline: Any, metric: Any) -> Dict[str, Any]:
        """Return a fixed evaluation result for assertions."""
        return {"metric": metric, "status": "ok"}


class DummyDspyModule:
    """Simple callable module for pipeline dependencies."""

    def __init__(self, response: DummyPrediction) -> None:
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> DummyPrediction:
        """Record calls and return the configured prediction."""
        if args:
            self.calls.append({"args": args, "kwargs": kwargs})
        else:
            self.calls.append(kwargs)
        return self.response


class DummyRetriever:
    """Weaviate retriever stand-in that returns fixed passages."""

    def __init__(self, passages: List[str]) -> None:
        self.passages = passages
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> DummyPrediction:
        """Track retrieval calls while returning fixed passages."""
        self.calls.append(kwargs)
        return DummyPrediction(passages=self.passages)


class DummyDataset:
    """Dataset stand-in compatible with train/test split usage."""

    def __init__(self, questions: List[str], answers: List[str]) -> None:
        self._data = {
            "question": questions,
            "answer": answers,
            "long_answer": answers,
            "label": ["label" for _ in questions],
        }

    def __getitem__(self, item: str) -> List[str]:
        return self._data[item]

    def train_test_split(self, test_size: float) -> Dict[str, "DummyDataset"]:
        """Return deterministic train/test splits for pipeline tests."""
        _ = test_size
        return {
            "train": DummyDataset(self._data["question"], self._data["answer"]),
            "test": DummyDataset(self._data["question"], self._data["answer"]),
        }


class DummyMetricMeta:
    """Metric metadata container for DeepEval results."""

    def __init__(self, name: str, score: float) -> None:
        self.name = name
        self.score = score


class DummyTestResult:
    """Test result container for DeepEval results."""

    def __init__(self, metrics_data: List[DummyMetricMeta]) -> None:
        self.metrics_data = metrics_data


class DummyEvaluationResult:
    """Evaluation result container for DeepEval results."""

    def __init__(self, scores: Dict[str, float]) -> None:
        self.test_results = [
            DummyTestResult(
                [DummyMetricMeta(name, score) for name, score in scores.items()]
            )
        ]
