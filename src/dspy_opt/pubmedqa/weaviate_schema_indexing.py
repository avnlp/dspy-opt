import logging
import os
import time
import weaviate
from weaviate.classes.init import Auth
from weaviate.collections.classes.config import Configure, Property, DataType
from weaviate.collections.classes.grpc import MetadataQuery
from datasets import load_dataset
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pubmedqa_rag")

load_dotenv()

CLASS_NAME = "PubMedQA"

def get_weaviate_client():
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_key = os.getenv("WEAVIATE_API_KEY")
    cohere_key = os.getenv("COHERE_APIKEY")

    if not all([weaviate_url, weaviate_key, cohere_key]):
        raise ValueError("❌ Missing env vars: WEAVIATE_URL, WEAVIATE_API_KEY, COHERE_APIKEY")

    headers = {"X-Cohere-Api-Key": cohere_key}
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_key),
        headers=headers,
        )
    logger.info(" Connected to Weaviate Cloud (v4).")
    return client

def create_schema(client):
    if client.collections.exists(CLASS_NAME):
        logger.info(f"🗑️ Deleting existing collection '{CLASS_NAME}'...")
        client.collections.delete(CLASS_NAME)
        time.sleep(2)

    client.collections.create(
        name=CLASS_NAME,
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
    logger.info(" Schema created.")

def load_pubmedqa_train():
    logger.info("📥 Loading PubMedQA train split...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    records = []
    for item in dataset["train"]:
        records.append({
            "question": item["question"],
            "context": " ".join(item["context"]),
            "answer": item["final_decision"],
            "pubid": str(item["pubid"]),
        })
    logger.info(f" Loaded {len(records)} records.")
    return records

def upload_documents(client, records):
    collection = client.collections.get(CLASS_NAME)
    logger.info(f" Uploading {len(records)} records...")
    start = time.time()
    with collection.batch.dynamic() as batch:
        for i, doc in enumerate(records):
            batch.add_object(properties=doc)
            if (i + 1) % 100 == 0:
                logger.info(f"Indexed {i + 1} / {len(records)}")
    logger.info(f" Upload done in {time.time() - start:.2f}s")

#  NEW: Retrieve one document
def retrieve_one_document(client):
    """Fetch one document using a simple query (no semantic search)."""
    collection = client.collections.get(CLASS_NAME)
    
    # Method 1: Get one object by limiting results (no vector search)
    response = collection.query.fetch_objects(limit=1)
    
    if response.objects:
        obj = response.objects[0]
        logger.info(" Retrieved one document:")
        print(f"Question: {obj.properties['question']}")
        print(f"Answer: {obj.properties['answer']}")
        print(f"PubID: {obj.properties['pubid']}")
        print(f"Context (truncated): {obj.properties['context'][:200]}...")
        return obj
    else:
        logger.warning(" No documents found in collection.")
        return None

#  NEW: Semantic search example
def semantic_search(client, query_text: str, limit: int = 1):
    """Perform vector search using Cohere embeddings."""
    collection = client.collections.get(CLASS_NAME)
    response = collection.query.near_text(
        query=query_text,
        limit=limit,
        return_metadata=MetadataQuery(distance=True)
    )
    
    if response.objects:
        obj = response.objects[0]
        logger.info(f" Top result for query: '{query_text}'")
        print(f"Distance: {obj.metadata.distance:.4f}")
        print(f"Question: {obj.properties['question']}")
        print(f"Answer: {obj.properties['answer']}")
        print(f"PubID: {obj.properties['pubid']}")
        return obj
    else:
        logger.warning(" No results for semantic search.")
        return None

# Main execution
if __name__ == "__main__":
    client = get_weaviate_client()
    
    # (Optional) Re-ingest data — comment out if already loaded
    # create_schema(client)
    # records = load_pubmedqa_train()
    # upload_documents(client, records)
    
    #  Retrieve one document
    logger.info("\n--- Fetching one random document ---")
    retrieve_one_document(client)
    
    #  Semantic search example
    logger.info("\n--- Semantic search example ---")
    semantic_search(client, "What are the best ways to prevent common cold?")
    
    client.close()
    logger.info("CloseOperation: Connection closed.")