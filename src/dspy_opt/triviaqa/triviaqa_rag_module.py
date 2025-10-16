"""TriviaQA RAG Pipeline using DSPy framework."""

from typing import Any, Dict

import dspy
from sentence_transformers import SentenceTransformer

from dspy_opt.utils.metadata_extractor import MetadataExtractor
from dspy_opt.utils.query_rewriter import QueryRewriter
from dspy_opt.utils.sub_query_generator import SubQueryGenerator
from dspy_opt.utils.weaviate_retriever import WeaviateRetriever


# First define all signatures
class TriviaQAAnswerSignature(dspy.Signature):
    """Signature for generating answers to TriviaQA questions."""

    context = dspy.InputField(desc="List of relevant passages from the knowledge base")
    question = dspy.InputField(desc="The original question to be answered")
    rewritten_query = dspy.OutputField(desc="Rewritten query to optimize for retrieval")
    sub_queries = dspy.OutputField(
        desc="List of sub-queries generated for complex questions"
    )
    answer = dspy.OutputField(desc="Concise and accurate answer to the question")
    reasoning = dspy.OutputField(desc="Brief explanation of how the answer was derived")


class TriviaQARAG(dspy.Module):
    """Complete TriviaQA RAG pipeline using DSPy framework."""

    def __init__(
        self,
        query_rewriter: QueryRewriter,
        sub_query_generator: SubQueryGenerator,
        metadata_extractor: MetadataExtractor,
        metadata_schema: Dict[str, Any],
        weaviate_retriever: WeaviateRetriever,
        embedding_model: SentenceTransformer,
        top_k: int = 3,
    ):
        """Initialize the TriviaQA RAG pipeline."""
        super().__init__()

        # Initialize components
        self.query_rewriter = query_rewriter
        self.sub_query_generator = sub_query_generator
        self.metadata_extractor = metadata_extractor
        self.metadata_schema = metadata_schema
        self.retriever = weaviate_retriever
        self.embedding_model = embedding_model
        self.top_k = top_k

        # Define the answer generation signature
        self.generate_answer = dspy.ChainOfThought(TriviaQAAnswerSignature)

    def forward(self, question: str) -> dspy.Prediction:
        """Execute the complete RAG pipeline."""
        try:
            # Rewrite the query to optimize for retrieval
            rewritten_query_result = self.query_rewriter(question)
            rewritten_query = rewritten_query_result.rewritten_query

            # Generate sub-queries if the question is complex
            sub_queries_result = self.sub_query_generator(rewritten_query)
            sub_queries = sub_queries_result.sub_queries

            # Extract metadata from all the queries
            rewritten_query_metadata = self.metadata_extractor(
                rewritten_query, self.metadata_schema
            )
            sub_queries_metadata = [
                self.metadata_extractor(sub_query, self.metadata_schema)
                for sub_query in sub_queries
            ]

            # Retrieve documents for each query
            all_passages = []

            # Retrieve for the main rewritten query
            main_retrieval = self.retriever(
                query=rewritten_query,
                query_embedding=self.embedding_model.encode(rewritten_query),
                top_k=self.top_k,
                metadata=rewritten_query_metadata,
            )
            all_passages.extend(main_retrieval.passages)

            # Retrieve for each sub-query if any exist
            for sub_query, sub_query_metadata in zip(sub_queries, sub_queries_metadata):
                sub_retrieval = self.retriever(
                    query=sub_query,
                    query_embedding=self.embedding_model.encode(sub_query),
                    top_k=self.top_k,
                    metadata=sub_query_metadata,
                )
                all_passages.extend(sub_retrieval.passages)

            # Remove duplicates while preserving order
            unique_passages = list(dict.fromkeys(all_passages))

            # If no context was retrieved, use a fallback message
            if not unique_passages:
                unique_passages = ["No relevant context found in the knowledge base."]

            answer_result = self.generate_answer(
                context=unique_passages, question=question
            )

            return dspy.Prediction(
                question=question,
                rewritten_query=rewritten_query,
                sub_queries=sub_queries,
                retrieved_context=unique_passages,
                answer=answer_result.answer,
                reasoning=answer_result.reasoning,
            )
        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            fallback_answer = self.generate_answer(
                context=["Limited context available"], question=question
            )
            return dspy.Prediction(
                question=question,
                rewritten_query=question,
                sub_queries=[question],
                retrieved_context=["Limited context available"],
                answer=fallback_answer.answer,
                reasoning=f"Error recovery: {str(e)}",
            )
