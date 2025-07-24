import os
import time

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.auth import AuthApiKey


def get_weaviate_client():
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        msg = "Missing required environment variables: WEAVIATE_URL, WEAVIATE_API_KEY"
        raise ValueError(msg)

    return weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=AuthApiKey(WEAVIATE_API_KEY),
        additional_headers={"X-OpenAI-Api-Key": "sk-ignore"},
    )


def create_schema(client, class_name):
    class_obj = {
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "domain", "dataType": ["text"]},
            {"name": "time_period", "dataType": ["text"]},
            {"name": "entities", "dataType": ["text[]"]},
        ],
    }

    if client.schema.exists(class_name):
        print("Deleting existing class...")
        client.schema.delete_class(class_name)
        time.sleep(2)

    print("Creating new class schema...")
    client.schema.create_class(class_obj)
    return class_name


def upload_documents(client, class_name, passages, embedding_model):
    print("Generating embeddings... This may take a few minutes...")
    passage_embeddings = embedding_model.encode(passages, show_progress_bar=True).tolist()

    documents = [{"content": para, "vector": emb} for para, emb in zip(passages, passage_embeddings, strict=False)]

    print(f"Uploading {len(documents)} passages to Weaviate...")
    batch_size = 100
    with client.batch as batch:
        batch.batch_size = batch_size
        for i, doc in enumerate(documents):
            properties = {"content": doc["content"]}
            batch.add_data_object(data_object=properties, class_name=class_name, vector=doc["vector"])
            if i % 100 == 0:
                print(f"Uploaded {i} documents")
    print(f"Finished uploading {len(documents)} documents")
    return documents
