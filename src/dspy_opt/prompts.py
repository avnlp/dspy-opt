from dspy import InputField, OutputField, Signature


class GenerateAnswer(Signature):
    """Answer questions with using the context provided."""

    context = InputField(desc="contains relevant facts about the question")
    question = InputField(desc="question to be answered")
    answer = OutputField(desc="precise answer")
