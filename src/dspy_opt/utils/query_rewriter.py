"""Query Rewriter: Rewrite user queries to optimize for information retrieval."""

from typing import List

import dspy


class QueryRewriteSignature(dspy.Signature):
    """Rewrite user queries to optimize for information retrieval."""

    original_query = dspy.InputField(
        desc="User's original search query. Must be rewritten to improve search effectiveness "
        "without altering core intent. Focus on: "
        "- Expanding with relevant synonyms/concepts "
        "- Clarifying ambiguous terms "
        "- Removing noise words "
        "- Maintaining conciseness "
        "- Preserving key entities and constraints"
    )
    rewritten_query = dspy.OutputField(
        desc="Optimized query string ready for search engine. Must: "
        "- Be 5-15 words long "
        "- Contain only essential search terms "
        "- Exclude explanatory phrases like 'I want' or 'looking for' "
        "- Include expanded concepts where helpful "
        "- Preserve numerical constraints and key entities "
        "- Output ONLY the rewritten query string with no additional text"
    )


class QueryRewriter(dspy.Module):
    """Rewrite search queries to improve retrieval effectiveness.

    This component:
    1. Expands queries with relevant semantic concepts
    2. Clarifies ambiguous terms using context
    3. Removes conversational noise
    4. Preserves critical constraints (dates, numbers, entities)
    5. Optimizes for search engine tokenization

    Example transformations:
        "How do I fix my leaking faucet?" → "faucet repair leak plumbing tools"
        "Best camera under $500 for travel" → "best compact camera $500 travel"
    """

    def __init__(self, use_chain_of_thought: bool = True):
        """Initializes QueryRewriter.

        Args:
            use_chain_of_thought: Whether to use reasoning steps for better rewrites
                (default: True for higher quality results)
        """
        super().__init__()
        if use_chain_of_thought:
            self.rewriter = dspy.ChainOfThought(QueryRewriteSignature)
        else:
            self.rewriter = dspy.Predict(QueryRewriteSignature)

    def forward(self, query: str) -> dspy.Prediction:
        """Rewrites a single query for optimal search performance.

        Args:
            query: Original user search query

        Returns:
            dspy.Prediction containing:
                - rewritten_query: Optimized search string
                - rationale (if using ChainOfThought): Reasoning steps

        Example:
            >>> rewriter = QueryRewriter()
            >>> result = rewriter("cheap flights to Paris next week")
            >>> result.rewritten_query
            'affordable flights Paris France departure date next 7 days'
        """
        return self.rewriter(original_query=query)

    def batch_rewrite(self, queries: List[str]) -> List[str]:
        """Rewrites multiple queries efficiently.

        Args:
            queries: List of original search queries

        Returns:
            List of optimized search queries

        Example:
            >>> rewriter.batch_rewrite(
                [
                    "iPhone battery life",
                    "gluten free restaurants",
                ]
            )
            [
                'iPhone 15 battery endurance specifications',
                'gluten-free restaurants nearby dietary restriction'
            ]
        """
        return [self.forward(q).rewritten_query for q in queries]
