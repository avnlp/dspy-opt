import dspy
from dspy.retrieve.weaviate_rm import WeaviateRM


class GenerateAnswer(dspy.Signature):
    """Answer questions based on the context."""

    context = dspy.InputField(desc="Context containing relevant facts")
    question = dspy.InputField(desc="Question to answer")
    answer = dspy.OutputField(desc="Generated answer")


class TriviaRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(answer=prediction.answer)


def initialize_rag(class_name, weaviate_client):
    WeaviateRM(class_name, weaviate_client=weaviate_client)
    return TriviaRAG()
