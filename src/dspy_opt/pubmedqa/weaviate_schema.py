import logging
import os
import time

import weaviate
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from weaviate.auth import AuthApiKey

logger = logging.getLogger("pubmedqa_rag.schema")

# Load environment variables
load_dotenv()


def load_config():
    with open("pubmedqa_config.yaml") as f:
        return yaml.safe_load(f)


def get_weaviate_client(config):
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", config["weaviate"].get("url", "http://localhost:8080"))
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

    auth_config = AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
    return weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth_config)


def create_schema(client, config):
    class_name = config["weaviate"]["class_name"]
    class_obj = {
        "class": class_name,
        "properties": config["weaviate"]["properties"],
        "vectorizer": config["weaviate"]["vectorizer"],
    }

    if client.schema.exists(class_name):
        logger.info("Deleting existing class...")
        client.schema.delete_class(class_name)
        time.sleep(2)

    logger.info("Creating new class schema...")
    client.schema.create_class(class_obj)
    logger.info("Schema created successfully")
    return class_name


def upload_documents(client, class_name, contexts, config):
    documents = [{"content": ctx} for ctx in contexts]
    batch_size = config["weaviate"].get("batch_size", 50)

    logger.info(f"Uploading {len(documents)} documents to Weaviate...")
    start_time = time.time()

    with client.batch(batch_size=batch_size) as batch:
        for i, doc in enumerate(documents):
            batch.add_data_object(doc, class_name)
            if i % 100 == 0 and i > 0:
                logger.info(f"Indexed {i} documents")

    logger.info(f"Completed indexing in {time.time() - start_time:.2f} seconds")
