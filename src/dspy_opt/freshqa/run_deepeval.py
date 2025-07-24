import dspy
from datasets import load_dataset
from evaluate_deepeval import evaluate_pipeline, freshqa_metric, prepare_eval_data
from indexing import create_schema, get_weaviate_client, upload_documents
from metadata_extractor import EnhancedRetriever, QueryMetadataEnhancer
from prompt_optimization import optimize_pipeline
from prompts import FreshQARAG
from sentence_transformers import SentenceTransformer


def preprocess_freshqa(data, num_samples=1000):
    processed_data = []
    unique_passages = set()

    for i, sample in enumerate(data):
        if i >= num_samples:
            break

        paragraphs = []
        for p in sample["context"]:
            clean_p = p.strip()
            if clean_p:
                paragraphs.append(clean_p)
                unique_passages.add(clean_p)

        processed_data.append({"question": sample["question"], "answer": sample["answer"], "paragraphs": paragraphs})

    return processed_data, list(unique_passages)


def create_retriever(client, class_name, embedding_model):
    class BaseRetriever(dspy.Retrieve):
        def __init__(self, k=5):
            super().__init__(k=k)
            self.client = client
            self.class_name = class_name
            self.embedding_model = embedding_model

        def forward(self, query):
            query_vector = self.embedding_model.encode(query).tolist()
            result = (
                self.client.query.get(self.class_name, ["content"])
                .with_near_vector({"vector": query_vector})
                .with_limit(self.k)
                .do()
            )

            if "errors" in result:
                print("Retrieval error:", result["errors"])
                return dspy.Prediction(passages=[])
            passages = [item["content"] for item in result["data"]["Get"][self.class_name]]
            return dspy.Prediction(passages=passages)

    base_retriever = BaseRetriever(k=5)
    metadata_extractor = QueryMetadataEnhancer()
    return EnhancedRetriever(base_retriever, metadata_extractor)


def main():
    # Configuration
    CLASS_NAME = "FreshQA"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    OPTIMIZATION_CONFIG = {
        "max_bootstrapped_demos": 8,
        "num_candidate_programs": 10,
        "num_threads": 4,
        "train_samples": 100,
    }

    # Initialize components
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Load dataset
    dataset = load_dataset("SamyaDey/FreshQA")
    train_data = dataset["train"]
    validation_data = dataset["validation"]

    # Preprocess data
    print("Preprocessing training data...")
    train_samples, train_paragraphs = preprocess_freshqa(train_data, num_samples=2000)
    print("Preprocessing validation data...")
    validation_samples, _ = preprocess_freshqa(validation_data, num_samples=500)
    print(f"Preprocessed {len(train_paragraphs)} unique passages")

    # Setup Weaviate
    client = get_weaviate_client()
    create_schema(client, CLASS_NAME)
    upload_documents(client, CLASS_NAME, train_paragraphs, embedding_model)

    # Initialize RAG pipeline
    retriever = create_retriever(client, CLASS_NAME, embedding_model)
    freshqa_rag = FreshQARAG(retriever=retriever)

    # Test pipeline
    print("\nTesting pipeline...")
    question = "What was the result of Ashleigh Barty's performance in Miami?"
    prediction = freshqa_rag(question)
    print("\nTest Question:", question)
    print("Retrieved Contexts:", prediction.context[:1])
    print("Generated Answer:", prediction.answer)

    # Prepare evaluation
    eval_data = prepare_eval_data(validation_samples[:100])

    # Evaluate uncompiled
    uncompiled_score = evaluate_pipeline(freshqa_rag, eval_data, freshqa_metric)
    print(f"Uncompiled Pipeline Accuracy: {uncompiled_score * 100:.2f}%")

    # Optimize pipeline
    print("Optimizing pipeline...")
    train_set = prepare_eval_data(train_samples[: OPTIMIZATION_CONFIG["train_samples"]])
    optimized_rag = optimize_pipeline(freshqa_rag, freshqa_metric, train_set, OPTIMIZATION_CONFIG)

    # Test optimized
    print("\nTesting optimized pipeline...")
    test_questions = [
        "Who was named the champion in the tennis tournament?",
        "Which company announced a new AI model recently?",
        "What was the outcome of the recent space mission?",
    ]
    for q in test_questions:
        pred = optimized_rag(q)
        print(f"\nQuestion: {q}")
        print(f"Answer: {pred.answer}")

    # Evaluate optimized
    optimized_score = evaluate_pipeline(optimized_rag, eval_data, freshqa_metric)
    print(f"Optimized Pipeline Accuracy: {optimized_score * 100:.2f}%")
    print(f"Improvement: {optimized_score - uncompiled_score:.4f} points")
    print("\nPipeline optimization complete!")


if __name__ == "__main__":
    main()
