"""Shared test fixtures and helpers for DSPy pipelines."""

import sys
from types import ModuleType

import pytest

from tests.helpers import DummyDataset, DummyEmbeddingModel


def _install_sentence_transformers_stub() -> None:
    module = ModuleType("sentence_transformers")

    class SentenceTransformer:  # noqa: N801 - mirror external class name
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        def encode(self, _text: str):
            return DummyEmbeddingModel().encode(_text)

    module.SentenceTransformer = SentenceTransformer
    sys.modules.setdefault("sentence_transformers", module)


def _install_weaviate_stub() -> None:
    weaviate_module = ModuleType("weaviate")
    weaviate_module.connect_to_weaviate_cloud = lambda *args, **kwargs: None
    sys.modules.setdefault("weaviate", weaviate_module)

    wvc_module = ModuleType("weaviate.classes")

    class DummyAuth:
        @staticmethod
        def api_key(_key: str):
            return None

    class DummyFilter:
        @staticmethod
        def and_(*_filters):
            return None

        @staticmethod
        def by_property(_name: str):
            class DummyLike:
                @staticmethod
                def like(_value: object):
                    return None

            return DummyLike()

    class DummyMetadataQuery:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    wvc_module.init = ModuleType("init")
    wvc_module.init.Auth = DummyAuth
    wvc_module.query = ModuleType("query")
    wvc_module.query.Filter = DummyFilter
    wvc_module.query.MetadataQuery = DummyMetadataQuery
    sys.modules.setdefault("weaviate.classes", wvc_module)


def _install_deepeval_stub() -> None:
    deepeval_module = ModuleType("deepeval")
    deepeval_module.evaluate = lambda *args, **kwargs: None
    sys.modules.setdefault("deepeval", deepeval_module)

    metrics_module = ModuleType("deepeval.metrics")

    class DummyMetric:  # pragma: no cover - placeholder for imports
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    metrics_module.AnswerRelevancyMetric = DummyMetric
    metrics_module.ContextualPrecisionMetric = DummyMetric
    metrics_module.ContextualRecallMetric = DummyMetric
    metrics_module.ContextualRelevancyMetric = DummyMetric
    metrics_module.FaithfulnessMetric = DummyMetric
    metrics_module.BaseMetric = DummyMetric
    sys.modules.setdefault("deepeval.metrics", metrics_module)

    models_module = ModuleType("deepeval.models")

    class DummyLocalModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    models_module.LocalModel = DummyLocalModel
    sys.modules.setdefault("deepeval.models", models_module)

    evaluate_module = ModuleType("deepeval.evaluate")

    class DummyAsyncConfig:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    evaluate_module.AsyncConfig = DummyAsyncConfig
    sys.modules.setdefault("deepeval.evaluate", evaluate_module)

    test_case_module = ModuleType("deepeval.test_case")

    class DummyLLMTestCase:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    test_case_module.LLMTestCase = DummyLLMTestCase
    sys.modules.setdefault("deepeval.test_case", test_case_module)


def _install_datasets_stub() -> None:
    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: DummyDataset(["Q"], ["A"])
    sys.modules.setdefault("datasets", datasets_module)


def _install_dspy_stub() -> None:
    dspy_module = ModuleType("dspy")

    class DummyModule:
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

        def __call__(self, *args: object, **kwargs: object):
            forward = getattr(self, "forward", None)
            if callable(forward):
                return forward(*args, **kwargs)
            return None

    class DummyPrediction:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class DummySignature:
        pass

    class DummyField:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    class DummyChainOfThought:
        def __init__(self, _signature: object) -> None:
            self.signature = _signature

        def __call__(self, **kwargs: object) -> DummyPrediction:
            return DummyPrediction(**kwargs)

    class DummyPredict(DummyChainOfThought):
        pass

    class DummyEvaluate:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        def __call__(self, _pipeline: object, metric: object) -> dict:
            return {"metric": metric, "status": "ok"}

    class DummyOptimizer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        def compile(self, pipeline: object, trainset: object):
            return self

        def save(self, *args: object, **kwargs: object) -> None:
            return None

    class DummyExample:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs
            self.text = kwargs.get("text")
            self.metadata = kwargs.get("metadata", {})

        def with_inputs(self, *_args: str):
            return self

    def dummy_context(*args: object, **kwargs: object):
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, *exc: object) -> None:
                return None

        return _Ctx()

    dspy_module.Module = DummyModule
    dspy_module.Signature = DummySignature
    dspy_module.InputField = DummyField
    dspy_module.OutputField = DummyField
    dspy_module.Prediction = DummyPrediction
    dspy_module.ChainOfThought = DummyChainOfThought
    dspy_module.Predict = DummyPredict
    dspy_module.Evaluate = DummyEvaluate
    dspy_module.MIPROv2 = DummyOptimizer
    dspy_module.SIMBA = DummyOptimizer
    dspy_module.COPRO = DummyOptimizer
    dspy_module.GEPA = DummyOptimizer
    dspy_module.BootstrapFewShotWithRandomSearch = DummyOptimizer
    dspy_module.Example = DummyExample
    dspy_module.context = dummy_context
    dspy_module.LM = DummyModule
    dspy_module.configure = lambda *args, **kwargs: None
    sys.modules.setdefault("dspy", dspy_module)


def pytest_sessionstart() -> None:
    _install_sentence_transformers_stub()
    _install_weaviate_stub()
    _install_deepeval_stub()
    _install_datasets_stub()
    _install_dspy_stub()


@pytest.fixture
def dummy_embedding_model() -> DummyEmbeddingModel:
    return DummyEmbeddingModel()


@pytest.fixture
def dummy_dataset() -> DummyDataset:
    return DummyDataset(["Q1"], ["A1"])
