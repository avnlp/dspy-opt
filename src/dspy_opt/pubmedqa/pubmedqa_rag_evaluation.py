"""PubMedQA RAG Pipeline using DSPy framework."""

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

from dspy_opt.pubmedqa.pubmedqa_rag_module import PubMedQARAG
from dspy_opt.utils.metadata_extractor import MetadataExtractor
from dspy_opt.utils.metrics import create_metrics_function
from dspy_opt.utils.query_rewriter import QueryRewriter
from dspy_opt.utils.sub_query_generator import SubQueryGenerator
from dspy_opt.utils.weaviate_retriever import WeaviateRetriever


def main() -> None:
    """Evaluation of the RAG pipeline on PubMedQA dataset."""
    # Load configuration from YAML file
    with open("pubmedqa_rag_evaluation_config.yml", "r") as f:
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
    rag_pipeline = PubMedQARAG(
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
    test_dataset = load_dataset(
        config["dataset"]["name"],
        subset=config["dataset"]["subset"],
        split=config["dataset"]["split"],
    )
    testset = [
        dspy.Example(question=question, answer=answer).with_inputs("question")
        for question, answer in zip(test_dataset["question"], test_dataset["answer"])
    ]

    # Evaluate the RAG pipeline
    evaluate = dspy.Evaluate(
        devset=testset,
        num_threads=config["evaluation"]["settings"]["num_threads"],
        display_progress=config["evaluation"]["settings"]["display_progress"],
        display_table=config["evaluation"]["settings"]["display_table"],
        provide_traceback=config["evaluation"]["settings"]["provide_traceback"],
    )
    results = evaluate(rag_pipeline, metric=metrics_function)
    print(results)


if __name__ == "__main__":
    main()
