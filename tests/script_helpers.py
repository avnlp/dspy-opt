"""Shared test helpers for dataset optimizer/evaluation scripts."""

from typing import Any, Dict

import pytest_mock

from tests.helpers import (
    DummyDataset,
    DummyEmbeddingModel,
    DummyEvaluate,
    DummyExample,
    DummyOptimizer,
)


def base_config() -> Dict[str, Any]:
    return {
        "answer_llm": {"model": "m", "api_key_env": "ANSWER_KEY"},
        "extractor_llm": {"model": "m", "api_key_env": "EXTRACT_KEY"},
        "embedding": {"embedding_model": "e", "tokenizer_kwargs": {}},
        "weaviate": {"collection_name": "C", "top_k": 2},
        "metadata_schema": {},
        "rag_pipeline": {"top_k": 2},
        "evaluation": {
            "evaluator_llm": {
                "model": "m",
                "api_key_env": "EVAL_KEY",
                "base_url": "http://local",
            },
            "metrics": {
                "answer_relevancy": {},
                "contextual_precision": {},
                "contextual_recall": {},
                "contextual_relevancy": {},
                "faithfulness": {},
            },
            "settings": {
                "num_threads": 1,
                "display_progress": False,
                "display_table": False,
                "provide_traceback": False,
            },
        },
        "dataset": {
            "name": "name",
            "subset": "subset",
            "subset_name": "subset",
            "split": "train",
            "test_size": 0.1,
        },
        "optimizer": {
            "max_bootstrapped_demos": 1,
            "max_labeled_demos": 1,
            "max_rounds": 1,
            "auto": True,
            "bsize": 1,
            "num_candidates": 1,
            "max_steps": 1,
            "max_demos": 1,
            "breadth": 1,
            "depth": 1,
            "init_temperature": 0.1,
            "max_full_evals": 1,
            "reflection_minibatch_size": 1,
            "candidate_selection_strategy": "best",
            "use_merge": False,
            "num_threads": 1,
            "seed": 1,
        },
        "reflection_llm": {
            "model": "m",
            "api_key_env": "REFLECT_KEY",
            "temperature": 0.1,
            "max_tokens": 10,
        },
        "train_dataset": {"name": "name", "split": "train"},
        "test_dataset": {"name": "name", "split": "test"},
    }


def apply_common_patches(module: Any, mocker: pytest_mock.MockFixture) -> None:
    mocker.patch.object(module.yaml, "safe_load", return_value=base_config())
    mocker.patch("builtins.open", mocker.mock_open(read_data="{}"))
    mocker.patch.object(module.os, "getenv", return_value="value")
    mocker.patch.object(module, "load_dotenv")
    mocker.patch.object(
        module, "SentenceTransformer", return_value=DummyEmbeddingModel()
    )
    mocker.patch.object(module, "QueryRewriter", return_value=mocker.Mock())
    mocker.patch.object(module, "SubQueryGenerator", return_value=mocker.Mock())
    mocker.patch.object(module, "MetadataExtractor", return_value=mocker.Mock())
    mocker.patch.object(module, "WeaviateRetriever", return_value=mocker.Mock())
    mocker.patch.object(module.dspy, "Example", DummyExample)
    mocker.patch.object(module, "load_dataset", return_value=DummyDataset(["Q"], ["A"]))
    mocker.patch.object(module.dspy, "Evaluate", DummyEvaluate)
    mocker.patch.object(module, "LocalModel", return_value=mocker.Mock())
    mocker.patch.object(module, "AnswerRelevancyMetric", return_value=mocker.Mock())
    mocker.patch.object(module, "ContextualPrecisionMetric", return_value=mocker.Mock())
    mocker.patch.object(module, "ContextualRecallMetric", return_value=mocker.Mock())
    mocker.patch.object(module, "ContextualRelevancyMetric", return_value=mocker.Mock())
    mocker.patch.object(module, "FaithfulnessMetric", return_value=mocker.Mock())
    mocker.patch.object(module.dspy, "MIPROv2", return_value=DummyOptimizer())
    mocker.patch.object(module.dspy, "SIMBA", return_value=DummyOptimizer())
    mocker.patch.object(module.dspy, "COPRO", return_value=DummyOptimizer())
    mocker.patch.object(
        module.dspy, "BootstrapFewShotWithRandomSearch", return_value=DummyOptimizer()
    )
    mocker.patch.object(module.dspy, "GEPA", return_value=DummyOptimizer())
    mocker.patch.object(module, "LocalModel", return_value=mocker.Mock())


def patch_rag_pipeline(
    module: Any, mocker: pytest_mock.MockFixture, rag_class: str
) -> None:
    mocker.patch.object(module, rag_class, return_value=mocker.Mock())


def patch_prediction_modules(module: Any, mocker: pytest_mock.MockFixture) -> None:
    mocker.patch.object(module.dspy, "LM", return_value=mocker.Mock())
    mocker.patch.object(module.dspy, "configure")


def config_with_overrides(**overrides: Any) -> Dict[str, Any]:
    config = base_config()
    for key, value in overrides.items():
        config[key] = value
    return config
