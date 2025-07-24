import weaviate
import yaml
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.init import Auth

# Load config from YAML
with open("indexing_config.yaml") as f:
    config = yaml.safe_load(f)

# Load HotpotQA Dataset
hotpotqa_data = load_dataset(config["DATASET_NAME"], split=config["DATASET_SPLIT"])

contexts = hotpotqa_data["context"]
document_texts = []
for context_list in contexts:
    for context in context_list:
        document_texts.append(context)

# Create and split Langchain Documents
documents = [Document(page_content=text) for text in document_texts]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["TEXT_SPLITTER_CHUNK_SIZE"], chunk_overlap=config["TEXT_SPLITTER_CHUNK_OVERLAP"]
)
documents = text_splitter.split_documents(documents)

# Load Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name=config["EMBEDDING_MODEL_NAME"],
    model_kwargs=config["EMBEDDING_MODEL_MODEL_KWARGS"],
    encode_kwargs=config["EMBEDDING_MODEL_ENCODE_KWARGS"],
)

# Connect to Weaviate
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=config["WEAVIATE_URL"],
    auth_credentials=Auth.api_key(config["WEAVIATE_API_KEY"]),
)

# Index documents to Weaviate
db = WeaviateVectorStore.from_documents(documents, embeddings, client=weaviate_client)
