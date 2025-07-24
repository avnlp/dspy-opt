import logging
import os
import re
import time

import dspy
from datasets import load_dataset
from dspy.evaluate import Evaluate

from .evaluate_deepeval import composite_metric
from .prompt_optimization import create_ensemble, optimize_pipeline
from .retriever_and_rag import LocalRetriever, WikipediaRAG, initialize_rag
from .weaviate_schema import create_schema, get_weaviate_client, upload_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wikipedia_rag")


def clean_wiki_text(text):
    t = re.sub(r"<[^>]+>", "", text)
    t = re.sub(r"\[\d+(?:-\d+)?\]", "", t)
    t = re.sub(r"==+.*?==+", "", t)
    return re.sub(r"\s+", " ", t).strip()


def preprocess_wiki(data):
    processed = []
    seen = set()
    for sample in data:
        title, text = sample["title"], clean_wiki_text(sample["text"])
        for chunk in text.split("\n"):
            if len(chunk) < 100:
                continue
            cid = f"{title}__{hash(chunk)}"
            if cid not in seen:
                processed.append({"title": title, "content": chunk, "url": sample.get("url", "")})
                seen.add(cid)
    return processed


def main():
    # Initialize language model
    GROQ_API_KEY = "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"
    lm = dspy.LM("groq/llama3-8b-8192", api_key=GROQ_API_KEY)
    dspy.configure(lm=lm)

    # Load dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:100]")
    val_ds = ds.select(range(min(len(ds), 15)))

    # Preprocess data
    train_samples = preprocess_wiki(ds)
    preprocess_wiki(val_ds)

    # Setup Weaviate
    client = get_weaviate_client()
    class_name = create_schema(client)
    upload_documents(client, class_name, train_samples)

    # Initialize RAG
    wiki_rag = initialize_rag(client, class_name)

    # Prepare evaluation data
    eval_questions = [
        ("What is the capital of France?", "Paris"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
        ("Largest planet in solar system?", "Jupiter"),
        ("First moon landing year?", "1969"),
        ("Main component of sun?", "Hydrogen"),
    ]
    eval_data = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in eval_questions]

    # Evaluate baseline
    evaluate = Evaluate(devset=eval_data, metric=composite_metric)
    baseline_score = evaluate(wiki_rag)
    logger.info(f"Baseline Composite Score: {baseline_score * 100:.2f}%")

    # Optimize pipeline
    optimized_rag = optimize_pipeline(wiki_rag, composite_metric, eval_data[:3], eval_data[3:])

    # Create ensemble
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    retriever = LocalRetriever(client, class_name, embedding_model)
    compiled_ensemble = create_ensemble(optimized_rag, retriever, eval_data[:3], eval_data[3:])

    # Evaluate optimized pipelines
    optimized_score = evaluate(optimized_rag)
    logger.info(f"Optimized Composite Score: {optimized_score * 100:.2f}%")

    ensemble_score = evaluate(compiled_ensemble)
    logger.info(f"Ensemble Composite Score: {ensemble_score * 100:.2f}%")


if __name__ == "__main__":
    main()
