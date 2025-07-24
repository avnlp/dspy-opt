from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase


def pubmed_metric(gold, pred):
    if not pred.answer:
        return 0.0

    gold_answer = gold.answer.lower()
    pred_answer = pred.answer.lower()

    answer_map = {
        "yes": ["yes", "affirmative", "positive", "effective", "reduces", "beneficial"],
        "no": ["no", "negative", "not effective", "does not reduce", "ineffective", "harmful"],
        "maybe": ["maybe", "uncertain", "inconclusive", "mixed evidence", "possible", "potential"],
    }

    # Direct match
    if gold_answer == pred_answer:
        return 1.0

    # Alias matching
    for key, aliases in answer_map.items():
        if gold_answer == key:
            for alias in aliases:
                if alias in pred_answer:
                    return 1.0

    # Partial credit for related answers
    if "inconclusive" in pred_answer and "maybe" in gold_answer:
        return 0.8
    if "not" in pred_answer and "no" in gold_answer:
        return 0.9

    return 0.0


def composite_deepeval_metric(gold, pred):
    test_case = LLMTestCase(
        input=gold.question, actual_output=pred.answer, expected_output=gold.answer, retrieval_context=pred.context
    )

    faithfulness = FaithfulnessMetric().measure(test_case)
    relevancy = AnswerRelevancyMetric().measure(test_case)

    return (faithfulness.score + relevancy.score) / 2
