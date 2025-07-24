from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase


def trivia_metric(gold, pred):
    if not pred.answer:
        return 0.0

    predicted_answer = pred.answer.lower().strip()
    true_answer = gold.answer.lower().strip()

    if true_answer in predicted_answer:
        return 1.0
    if any(alias in predicted_answer for alias in true_answer.split()):
        return 0.8
    return 0.0


def composite_deepeval_metric(gold, pred):
    test_case = LLMTestCase(
        input=gold.question,
        actual_output=pred.answer,
        expected_output=gold.answer,
        retrieval_context=[],  # Not available in TriviaRAG output
    )
    faithfulness = FaithfulnessMetric().measure(test_case)
    relevancy = AnswerRelevancyMetric().measure(test_case)
    return (faithfulness.score + relevancy.score) / 2
