"""Compute metrics for a RAG pipeline using DeepEval metrics."""

from typing import Any, Callable, List, Optional

import dspy
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


def create_metrics_function(metrics: List[BaseMetric]) -> Callable[[Any, Any], float]:
    """Factory function that creates a metrics function with required DSPy signature.

    The provided list of metrics is used by the inner function.
    The metrics function is expected to return a single float score.

    Args:
        metrics (List[BaseMetric]): A list of DeepEval metrics to use for evaluation.

    Returns:
        function: A metrics function for dspy.Evaluate.
    """

    def deepeval_metrics(gold: Any, pred: Any, trace: Optional[bool] = None) -> float:
        """Evaluate a RAG pipeline using DeepEval metrics.

        Expected attributes:
            - gold: must have .question, .answer
            - pred: must have .answer (the LLM's actual output)
        """
        # Extract fields from gold and pred
        question = getattr(gold, "question", "")
        gold_answer = getattr(gold, "answer", "")
        retrieved_context = getattr(pred, "retrieved_context", [])
        pred_answer = getattr(pred, "answer", "")

        # Build the test case
        test_case = LLMTestCase(
            input=question,
            expected_output=gold_answer,
            actual_output=pred_answer,
            retrieval_context=retrieved_context,
        )

        # Run evaluation using the specified metrics
        evaluation_result = evaluate(  # type: ignore[operator]
            test_cases=[test_case],
            metrics=metrics,
            async_config=AsyncConfig(
                run_async=False, throttle_value=60, max_concurrent=1
            ),
        )

        # Extract scores for aggregation
        scores = {}
        for test_result in evaluation_result.test_results:
            for metric_meta in test_result.metrics_data:
                scores[metric_meta.name] = metric_meta.score

        # Print the scores for each metric
        for metric_name, score in scores.items():
            print(f"{metric_name}: {score}")

        # Aggregate scores
        return round(sum(scores.values()) / len(scores), 2) if scores else 0.0

    return deepeval_metrics


def create_gepa_metrics_function(
    metrics: List[BaseMetric],
) -> Callable[
    [Any, Any, Optional[bool], Optional[str], Optional[str]], dspy.Prediction
]:
    """Factory function that creates a GEPA metrics function with feedback.

    Args:
        metrics (List[BaseMetric]): A list of DeepEval metrics to use for evaluation.

    Returns:
        function: A metrics function returning a Prediction with score and feedback.
    """

    def deepeval_metrics(
        gold: Any,
        pred: Any,
        trace: Optional[bool] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[str] = None,
    ) -> dspy.Prediction:
        """Evaluate a RAG pipeline using DeepEval metrics with feedback support."""
        _ = (trace, pred_name, pred_trace)

        question = getattr(gold, "question", "")
        gold_answer = getattr(gold, "answer", "")
        retrieved_context = getattr(pred, "retrieved_context", [])
        pred_answer = getattr(pred, "answer", "")

        test_case = LLMTestCase(
            input=question,
            expected_output=gold_answer,
            actual_output=pred_answer,
            retrieval_context=retrieved_context,
        )

        evaluation_result = evaluate(  # type: ignore[operator]
            test_cases=[test_case],
            metrics=metrics,
            async_config=AsyncConfig(
                run_async=False, throttle_value=60, max_concurrent=1
            ),
        )

        scores = {}
        feedback_items = []
        for test_result in evaluation_result.test_results:
            for metric_meta in test_result.metrics_data:
                scores[metric_meta.name] = metric_meta.score
                feedback_items.append(f"{metric_meta.name}: {metric_meta.score}")

        avg_score = round(sum(scores.values()) / len(scores), 2) if scores else 0.0
        feedback = ", ".join(feedback_items)

        return dspy.Prediction(score=avg_score, feedback=feedback)

    return deepeval_metrics
