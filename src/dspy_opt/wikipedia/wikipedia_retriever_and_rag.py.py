import dspy
from sentence_transformers import SentenceTransformer


class LocalRetriever(dspy.Retrieve):
    def __init__(self, client, collection_name, model, k=3):
        super().__init__(k=k)
        self.client = client
        self.cn = collection_name
        self.model = model

    def forward(self, query):
        vec = self.model.encode(query).tolist()
        res = self.client.collections.get(self.cn).query.near_vector({"vector": vec}).with_limit(self.k).do()
        hits = res["data"]["Get"].get(self.cn, [])
        return dspy.Prediction(passages=[f"Title: {h['title']}\nContent: {h['content']}" for h in hits])


class GenerateWikiAnswer(dspy.Signature):
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()


class WikipediaRAG(dspy.Module):
    def __init__(self, retriever, num_passages=3):
        super().__init__()
        self.retrieve = retriever
        self.retrieve.k = num_passages
        self.generate = dspy.ChainOfThought(GenerateWikiAnswer)

    def forward(self, question):
        ctx = self.retrieve(question).passages
        ans = self.generate(context=ctx, question=question).answer
        return dspy.Prediction(context=ctx, answer=ans)


def initialize_rag(client, class_name):
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    retriever = LocalRetriever(client, class_name, embedding_model)
    return WikipediaRAG(retriever)
