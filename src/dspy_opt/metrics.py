from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase


def composite_deepeval_metric(gold, pred):
    """Combined faithfulness and answer relevancy metric."""
    test_case = LLMTestCase(
        input=gold.question, actual_output=pred.answer, expected_output=gold.answer, retrieval_context=pred.context
    )
    faithfulness = FaithfulnessMetric().measure(test_case)
    relevancy = AnswerRelevancyMetric().measure(test_case)
    return (faithfulness + relevancy) / 2
