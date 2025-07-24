import logging
import os
import time

import dspy
from datasets import load_dataset
from dspy.evaluate import Evaluate

from .evaluate_deepeval import composite_deepeval_metric, trivia_metric
from .prompt_optimization import create_ensemble, optimize_pipeline
from .retriever_and_rag import TriviaRAG, initialize_rag
from .weaviate_schema import create_schema, get_weaviate_client, upload_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trivia_rag")


def preprocess_triviaqa(data, num_samples=1000):
    return [
        {
            "question": sample["question"],
            "context": (
                sample["search_results"]["search_context"][0] if sample["search_results"]["search_context"] else ""
            ),
            "answer": sample["answer"]["aliases"][0] if sample["answer"]["aliases"] else sample["answer"]["value"],
        }
        for i, sample in enumerate(data)
        if i < num_samples
    ]


def prepare_eval_data(data):
    return [
        dspy.Example(question=sample["question"], answer=sample["answer"]).with_inputs("question") for sample in data
    ]


def main():
    # Load dataset
    dataset = load_dataset("trivia_qa", "unfiltered")
    train_data = dataset["train"]
    validation_data = dataset["validation"]

    # Preprocess data
    train_samples = preprocess_triviaqa(train_data, 2000)
    validation_samples = preprocess_triviaqa(validation_data, 500)

    # Setup Weaviate
    client = get_weaviate_client()
    class_name = create_schema(client)
    upload_documents(client, class_name, train_samples)

    # Initialize RAG
    trivia_rag = initialize_rag(class_name, client)

    # Prepare evaluation data
    eval_data = prepare_eval_data(validation_samples)

    # Evaluate baseline
    evaluate = Evaluate(devset=eval_data, metric=trivia_metric, num_threads=4)
    uncompiled_score = evaluate(trivia_rag)
    logger.info(f"Uncompiled Pipeline Accuracy: {uncompiled_score:.2%}")

    # Optimize pipeline
    train_set = eval_data[:350]
    val_set = eval_data[350:]
    optimized_rag = optimize_pipeline(trivia_rag, trivia_metric, train_set, val_set)

    # Create ensemble
    compiled_ensemble = create_ensemble(optimized_rag, train_set, val_set)

    # Evaluate optimized pipelines
    optimized_score = evaluate(optimized_rag)
    logger.info(f"Optimized Pipeline Accuracy: {optimized_score:.2%}")

    ensemble_score = evaluate(compiled_ensemble)
    logger.info(f"Ensemble Pipeline Accuracy: {ensemble_score:.2%}")

    # DeepEval evaluation
    deepeval_score = Evaluate(devset=eval_data[:50], metric=composite_deepeval_metric)(  # Use subset for efficiency
        optimized_rag
    )
    logger.info(f"DeepEval Composite Score: {deepeval_score * 100:.2f}%")


if __name__ == "__main__":
    main()
