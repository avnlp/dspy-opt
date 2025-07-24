import dspy
from dspy.teleprompt import Evaluate


def freshqa_metric(gold, pred):
    """Custom metric for FreshQA evaluation with multiple answer handling."""
    if not gold.answer or not pred.answer:
        return 0.0

    gold_answer = gold.answer.lower()
    pred_answer = pred.answer.lower()

    # Handle multiple acceptable answers (separated by '|')
    if "|" in gold_answer:
        acceptable_answers = [ans.strip() for ans in gold_answer.split("|")]
        for ans in acceptable_answers:
            if ans in pred_answer:
                return 1.0
        return 0.0

    # Single answer evaluation
    if gold_answer == pred_answer:
        return 1.0
    if gold_answer in pred_answer:
        return 0.9
    return 0.0


def prepare_eval_data(samples):
    return [dspy.Example(question=s["question"], answer=s["answer"]).with_inputs("question") for s in samples]


def evaluate_pipeline(pipeline, eval_data, metric, num_threads=4):
    evaluator = Evaluate(
        devset=eval_data, metric=metric, num_threads=num_threads, display_progress=True, display_table=3
    )
    return evaluator(pipeline)
