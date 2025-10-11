import logging
import os
import time
import weaviate
from weaviate.classes.init import Auth
from weaviate.collections.classes.config import Configure, Property, DataType
from datasets import load_dataset
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pubmedqa_rag.ingest")

# Load .env
load_dotenv()

def get_weaviate_client():
    """Connect to Weaviate Cloud using v4 client with Cohere header."""
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_key = os.getenv("WEAVIATE_API_KEY")
    cohere_key = os.getenv("COHERE_APIKEY")

    if not weaviate_url or not weaviate_key or not cohere_key:
        raise ValueError(" Missing WEAVIATE_URL, WEAVIATE_API_KEY, or COHERE_APIKEY in .env")

    logger.info(f"Connecting to Weaviate Cloud: {weaviate_url}")
    headers = {"X-Cohere-Api-Key": cohere_key}

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_key),
        headers=headers,
    )

    logger.info(" Connected to Weaviate Cloud (v4).")
    return client

def create_schema(client, class_name: str):
    """Create a collection (class) with text2vec-cohere vectorizer."""
    if client.collections.exists(class_name):
        logger.info(f"Collection '{class_name}' exists. Deleting...")
        client.collections.delete(class_name)
        time.sleep(2)

    logger.info(f"Creating collection '{class_name}' with text2vec-cohere...")

    client.collections.create(
        name=class_name,
        vectorizer_config=Configure.Vectorizer.text2vec_cohere(
            model="embed-english-v3.0",
            truncate="RIGHT"
        ),
        properties=[
            Property(name="question", data_type=DataType.TEXT),
            Property(name="context", data_type=DataType.TEXT),
            Property(name="answer", data_type=DataType.TEXT),
            Property(name="pubid", data_type=DataType.TEXT),
        ]
    )
    logger.info(" Schema created successfully.")

def load_pubmedqa_train():
    """Load only the 'train' split of PubMedQA (pqa_labeled)."""
    logger.info("📥 Loading PubMedQA (pqa_labeled, train split)...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    train = dataset["train"]
    records = []
    for item in train:
        records.append({
            "question": item["question"],
            "context": " ".join(item["context"]),
            "answer": item["final_decision"],
            "pubid": str(item["pubid"]),
        })
    logger.info(f" Loaded {len(records)} records.")
    return records

def upload_documents(client, class_name: str, records: list):
    """Batch upload records to Weaviate collection."""
    collection = client.collections.get(class_name)
    batch_size = 100
    logger.info(f"📤 Uploading {len(records)} records...")

    start_time = time.time()
    with collection.batch.dynamic() as batch:
        for i, doc in enumerate(records):
            batch.add_object(properties=doc)
            if (i + 1) % 100 == 0:
                logger.info(f"Indexed {i + 1} / {len(records)}")

    logger.info(f" Upload completed in {time.time() - start_time:.2f} seconds.")

# Main
if __name__ == "__main__":
    CLASS_NAME = "PubMedQA"
    client = get_weaviate_client()
    create_schema(client, CLASS_NAME)
    records = load_pubmedqa_train()
    upload_documents(client, CLASS_NAME, records)
    client.close()
    logger.info("CloseOperation: Connection closed.")