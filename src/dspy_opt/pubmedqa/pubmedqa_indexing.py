"""Index PubMedQA dataset for RAG."""

import os

import dspy
import weaviate
import weaviate.classes as wvc
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from dspy_opt.utils.metadata_extractor import MetadataExtractor


if __name__ == "__main__":
    # Load configuration from YAML file
    with open("pubmedqa_indexing_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Load environment variables
    load_dotenv()

    # Set the environment variables
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

    dataset = load_dataset(
        config["dataset"]["name"],
        subset=config["dataset"]["subset"],
        split=config["dataset"]["split"],
    )
    doc_texts = [
        context
        for context_dict in dataset["context"]
        for context in context_dict["contexts"]
    ]
    doc_examples = [dspy.Example(text=doc_text, metadata={}) for doc_text in doc_texts]

    metadata_schema = config["metadata_schema"]

    extractor_llm = dspy.LM(
        model=config["extractor_llm"]["model"], api_key=os.getenv("GROQ_API_KEY")
    )

    metadata_extractor = MetadataExtractor(extractor_llm=extractor_llm)
    doc_examples = metadata_extractor.transform_documents(doc_examples, metadata_schema)

    model = SentenceTransformer(
        config["embedding"]["embedding_model"],
        tokenizer_kwargs=config["embedding"]["tokenizer_kwargs"],
    )

    # Connect to Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,  # type: ignore[arg-type]
        auth_credentials=wvc.init.Auth.api_key(WEAVIATE_API_KEY),  # type: ignore[arg-type]
    )

    # Check connection
    client.is_ready()

    collection_name = config["collection_name"]

    # Create the collection
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    weaviate_collection = client.collections.create(
        collection_name,
        vector_config=wvc.config.Configure.Vectors.self_provided(),
    )

    # Encode the documents
    embeddings = model.encode(
        doc_texts,
        batch_size=config["document_encoding"]["batch_size"],
        show_progress_bar=config["document_encoding"]["show_progress_bar"],
    )
    question_objs = [
        wvc.data.DataObject(
            properties={
                "document_text": doc_text,
                **doc_example.metadata,
            },
            vector=embedding,
        )
        for embedding, doc_text, doc_example in zip(embeddings, doc_texts, doc_examples)
    ]

    weaviate_collection = client.collections.use(collection_name)
    weaviate_collection.data.insert_many(question_objs)

    # Close connection
    client.close()
