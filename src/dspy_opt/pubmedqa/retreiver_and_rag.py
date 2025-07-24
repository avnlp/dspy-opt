import dspy
from dspy.retrieve.weaviate_rm import WeaviateRM


class GenerateBiomedAnswer(dspy.Signature):
    """Answer biomedical questions based on research context."""

    context = dspy.InputField(desc="Relevant biomedical research context")
    question = dspy.InputField(desc="Clinical or research question")
    rationale = dspy.OutputField(desc="Step-by-step reasoning before final answer")
    answer = dspy.OutputField(desc="Final concise answer (yes/no/maybe)")


class PubMedRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateBiomedAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, rationale=prediction.rationale, answer=prediction.answer)


def initialize_rag(class_name, weaviate_client, config):
    WeaviateRM(class_name, weaviate_client=weaviate_client)
    return PubMedRAG(num_passages=config["rag"]["num_passages"])
