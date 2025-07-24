from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase


def composite_metric(gold, pred):
    if not gold.answer or not pred.answer:
        return 0.0

    test_case = LLMTestCase(
        input=gold.question, actual_output=pred.answer, expected_output=gold.answer, retrieval_context=pred.context
    )

    faith = FaithfulnessMetric().measure(test_case)
    rel = AnswerRelevancyMetric().measure(test_case)
    prec = ContextualPrecisionMetric().measure(test_case)
    rec = ContextualRecallMetric().measure(test_case)

    return (faith.score + rel.score + prec.score + rec.score) / 4
