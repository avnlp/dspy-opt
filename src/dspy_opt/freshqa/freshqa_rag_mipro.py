"""Optimized FreshQA RAG Pipeline using the MIPROv2 optimizer."""

import os

import dspy
import yaml
from datasets import load_dataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import LocalModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from dspy_opt.freshqa.freshqa_rag_module import FreshQARAG
from dspy_opt.utils.metadata_extractor import MetadataExtractor
from dspy_opt.utils.metrics import create_metrics_function
from dspy_opt.utils.query_rewriter import QueryRewriter
from dspy_opt.utils.sub_query_generator import SubQueryGenerator
from dspy_opt.utils.weaviate_retriever import WeaviateRetriever


def main() -> None:
    """Evaluation of the RAG pipeline on FreshQA dataset."""
    # Load configuration from YAML file
    with open("freshqa_rag_mipro_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Load environment variables
    load_dotenv()

    # Set the environment variables
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    # Configure LLM for generating answer
    answer_llm = dspy.LM(
        model=config["answer_llm"]["model"],
        api_key=os.getenv(config["answer_llm"]["api_key_env"]),
    )
    dspy.configure(lm=answer_llm)
    extractor_llm = dspy.LM(
        model=config["extractor_llm"]["model"],
        api_key=os.getenv(config["extractor_llm"]["api_key_env"]),
    )

    model = SentenceTransformer(
        config["embedding"]["embedding_model"],
        tokenizer_kwargs=config["embedding"]["tokenizer_kwargs"],
    )

    # Initialize components for RAG
    query_rewriter = QueryRewriter()
    sub_query_generator = SubQueryGenerator()
    metadata_extractor = MetadataExtractor(extractor_llm=extractor_llm)

    # Initialize Weaviate retriever
    weaviate_retriever = WeaviateRetriever(
        weaviate_url=weaviate_url,
        weaviate_api_key=weaviate_api_key,
        collection_name=config["collection_name"],
        top_k=config["top_k"],
        metadata_schema=config["metadata_schema"],
    )

    # Initialize the RAG pipeline
    rag_pipeline = FreshQARAG(
        query_rewriter=query_rewriter,
        sub_query_generator=sub_query_generator,
        metadata_extractor=metadata_extractor,
        metadata_schema=config["metadata_schema"],
        weaviate_retriever=weaviate_retriever,
        embedding_model=model,
        top_k=config["top_k"],
    )

    evaluator_llm = LocalModel(
        model=config["evaluator_llm"]["model"],
        api_key=os.getenv(config["evaluator_llm"]["api_key_env"]),
        base_url=config["evaluator_llm"]["base_url"],
    )

    # Initialize metrics
    metrics = [
        AnswerRelevancyMetric(
            model=evaluator_llm,
            **config["metrics"]["answer_relevancy"],
        ),
        ContextualPrecisionMetric(
            model=evaluator_llm,
            **config["metrics"]["contextual_precision"],
        ),
        ContextualRecallMetric(
            model=evaluator_llm,
            **config["metrics"]["contextual_recall"],
        ),
        ContextualRelevancyMetric(
            model=evaluator_llm,
            **config["metrics"]["contextual_relevancy"],
        ),
        FaithfulnessMetric(
            model=evaluator_llm,
            **config["metrics"]["faithfulness"],
        ),
    ]
    metrics_function = create_metrics_function(metrics)

    # Load dataset
    dataset = load_dataset(
        config["dataset"]["name"],
        config["dataset"]["subset"],
        split=config["dataset"]["split"],
    )
    dataset = dataset.train_test_split(test_size=config["dataset"]["test_size"])
    trainset = [
        dspy.Example(question=question, answer=answer).with_inputs("question")
        for question, answer in zip(
            dataset["train"]["question"], dataset["train"]["answer"]
        )
    ]
    testset = [
        dspy.Example(question=question, answer=answer).with_inputs("question")
        for question, answer in zip(
            dataset["test"]["question"], dataset["test"]["answer"]
        )
    ]

    # Optimize the RAG Pipeline
    optimizer = dspy.MIPROv2(
        metric=metrics_function,
        max_bootstrapped_demos=config["optimizer"]["max_bootstrapped_demos"],
        max_labeled_demos=config["optimizer"]["max_labeled_demos"],
        auto=config["optimizer"]["auto"],
    )
    optimized_rag = optimizer.compile(
        rag_pipeline,
        trainset=trainset,
    )

    # Save Optimized Pipeline
    optimized_rag.save("optimized_rag_mipro.json")

    # Evaluate the optimized RAG pipeline
    evaluate = dspy.Evaluate(
        devset=testset,
        num_threads=config["evaluation"]["settings"]["num_threads"],
        display_progress=config["evaluation"]["settings"]["display_progress"],
        display_table=config["evaluation"]["settings"]["display_table"],
        provide_traceback=config["evaluation"]["settings"]["provide_traceback"],
    )
    results = evaluate(optimized_rag, metric=metrics_function)
    print(results)


if __name__ == "__main__":
    main()
