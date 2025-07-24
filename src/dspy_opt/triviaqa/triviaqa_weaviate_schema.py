import logging
import os
import time

import weaviate
from tqdm import tqdm
from weaviate.auth import AuthApiKey

logger = logging.getLogger("triviaqa.schema")


def get_weaviate_client():
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not all([WEAVIATE_URL, WEAVIATE_API_KEY, OPENAI_API_KEY]):
        msg = "Missing required environment variables"
        raise ValueError(msg)

    return weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=AuthApiKey(WEAVIATE_API_KEY),
        additional_headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
    )


# Define the Schema for the triviaqa dataset
def create_schema(client):
    class_name = "TriviaQA"
    class_schema = {
        "class": class_name,
        "vectorizer": "text2vec-openai",
        "moduleConfig": {"text2vec-openai": {"model": "text-embedding-ada-002", "modelVersion": "002", "type": "text"}},
        "properties": [{"name": "content", "dataType": ["text"]}],
    }

    if client.schema.exists(class_name):
        logger.info("Deleting existing class...")
        client.schema.delete_class(class_name)
        time.sleep(2)

    logger.info("Creating new schema...")
    client.schema.create_class(class_schema)
    return class_name


def upload_documents(client, class_name, train_samples):
    documents = [{"content": sample["context"]} for sample in train_samples if sample["context"]]

    logger.info(f"Uploading {len(documents)} documents to Weaviate...")
    with client.batch(batch_size=100) as batch:
        for _i, doc in enumerate(tqdm(documents, desc="Uploading")):
            batch.add_data_object(data_object={"content": doc["content"]}, class_name=class_name)
    return documents
