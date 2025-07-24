import logging
import os
import time

import weaviate
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from weaviate.auth import AuthApiKey

logger = logging.getLogger("wikipedia.schema")


def get_weaviate_client():
    WEAVIATE_URL = "https://gflesu4vr83zm2kkpeybq.c0.asia-southeast1.gcp.weaviate.cloud"
    WEAVIATE_API_KEY = "UG1nNW5lTUFCYTNDSGZzRV9lbjRNUlZiaGtBZkxDQ1g2SDhUczRPSlh4Y29TbGhQSEl3eEVCdnBmbXR3PV92MjAw"
    return weaviate.Client(url=WEAVIATE_URL, auth_client_secret=AuthApiKey(WEAVIATE_API_KEY))


def create_schema(client):
    class_name = "WikipediaChunks"
    if client.collections.exists(class_name):
        client.collections.delete(class_name)
        time.sleep(2)

    client.collections.create(
        name=class_name,
        properties=[
            weaviate.classes.config.Property(name="title", data_type=weaviate.classes.config.DataType.TEXT),
            weaviate.classes.config.Property(name="content", data_type=weaviate.classes.config.DataType.TEXT),
            weaviate.classes.config.Property(name="url", data_type=weaviate.classes.config.DataType.TEXT),
        ],
        vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
    )
    return class_name


def upload_documents(client, class_name, train_samples):
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    contents = [s["content"] for s in train_samples]
    embs = embedding_model.encode(contents, show_progress_bar=True, batch_size=16, convert_to_numpy=True)
    docs = [{"title": s["title"], "content": s["content"], "url": s.get("url", "")} for s in train_samples]

    collection = client.collections.get(class_name)
    with collection.batch(dynamic=True) as batch:
        for s, e in tqdm(zip(docs, embs, strict=False), desc="Uploading"):
            batch.add_object(data_object=s, vector=e.tolist())
    return docs
