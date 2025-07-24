import logging
import os
import time

import dspy
from datasets import load_dataset
from dspy.evaluate import Evaluate

from .evaluate_deepeval import composite_deepeval_metric, pubmed_metric
from .prompt_optimization import create_ensemble, optimize_pipeline
from .retriever_and_rag import PubMedRAG, initialize_rag
from .weaviate_schema import create_schema, get_weaviate_client, load_config, upload_documents

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pubmedqa_rag")


def preprocess_pubmedqa(data, config):
    processed_data = []
    unique_contexts = set()

    num_samples = config["preprocessing"]["num_samples"]
    max_chars = config["preprocessing"]["max_context_chars"]

    for sample in data.select(range(min(num_samples, len(data)))):
        question = sample["question"]
        context = sample["context"]["contexts"][0]  # Primary context
        context_chunk = context[:max_chars] + "..." if len(context) > max_chars else context

        processed_data.append(
            {
                "question": question,
                "context": context_chunk,
                "long_answer": sample["long_answer"],
                "final_decision": sample["final_decision"].lower(),
            }
        )
        unique_contexts.add(context_chunk)

    return processed_data, list(unique_contexts)


def prepare_eval_data(samples):
    return [
        dspy.Example(
            question=sample["question"],
            answer=sample["final_decision"],
        ).with_inputs("question")
        for sample in samples
    ]


def main():
    # Load configuration
    config = load_config()

    # Load dataset
    logger.info("Loading PubMedQA dataset...")
    dataset = load_dataset(config["dataset"]["name"], config["dataset"]["config_name"])
    train_data = dataset[config["dataset"]["train_split"]]
    validation_data = dataset[config["dataset"]["validation_split"]]

    # Preprocess data
    train_samples, train_contexts = preprocess_pubmedqa(train_data, config)
    validation_samples, _ = preprocess_pubmedqa(validation_data, num_samples=config["evaluation"]["validation_samples"])

    # Setup Weaviate
    client = get_weaviate_client(config)
    class_name = create_schema(client, config)
    upload_documents(client, class_name, train_contexts, config)

    # Initialize language model
    turbo = dspy.OpenAI(
        model=config["rag"]["lm_model"],
        max_tokens=config["rag"]["lm_max_tokens"],
        temperature=config["rag"].get("temperature", 0.7),
    )
    dspy.settings.configure(lm=turbo)

    # Initialize RAG
    pubmed_rag = initialize_rag(class_name, client, config)

    # Prepare evaluation data
    eval_data = prepare_eval_data(validation_samples)
    validation_samples = config["evaluation"]["validation_samples"]

    # Evaluate baseline
    evaluate = Evaluate(
        devset=eval_data[:validation_samples],
        metric=pubmed_metric,
        num_threads=config["evaluation"].get("num_threads", 4),
    )
    baseline_score = evaluate(pubmed_rag)
    logger.info(f"Baseline Accuracy: {baseline_score * 100:.2f}%")

    # Optimize pipeline
    train_samples = int(len(eval_data) * config["optimization"].get("train_ratio", 0.7))
    train_set = eval_data[:train_samples]
    val_set = eval_data[train_samples:]

    optimized_rag = optimize_pipeline(pubmed_rag, pubmed_metric, config, train_set, val_set)

    # Create ensemble
    compiled_ensemble = create_ensemble(optimized_rag, config, train_set, val_set)

    # Evaluate optimized pipelines
    optimized_score = evaluate(optimized_rag)
    logger.info(f"Optimized Accuracy: {optimized_score * 100:.2f}%")

    ensemble_score = evaluate(compiled_ensemble)
    logger.info(f"Ensemble Accuracy: {ensemble_score * 100:.2f}%")

    # DeepEval evaluation
    deepeval_score = Evaluate(devset=eval_data[: min(20, len(eval_data))], metric=composite_deepeval_metric)(
        optimized_rag
    )
    logger.info(f"DeepEval Composite Score: {deepeval_score * 100:.2f}%")


if __name__ == "__main__":
    main()
