import dspy


class GenerateAnswer(dspy.Signature):
    """Answer questions with reasoning based on relevant contexts."""

    context = dspy.InputField(desc="Relevant contexts for reasoning")
    question = dspy.InputField(desc="Question to answer")
    answer = dspy.OutputField(desc="Generated answer with justification")


class FreshQARAG(dspy.Module):
    def __init__(self, retriever, num_passages=5):
        super().__init__()
        self.retrieve = retriever
        self.retrieve.k = num_passages
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


class QueryMetadataExtractor(dspy.Signature):
    """Extract key metadata from user queries for enhanced retrieval."""

    question = dspy.InputField(desc="Original user question")
    entities = dspy.OutputField(desc="Comma-separated list of key entities")
    time_period = dspy.OutputField(desc="Relevant time period if mentioned")
    domain = dspy.OutputField(desc="Question domain/topic category")
