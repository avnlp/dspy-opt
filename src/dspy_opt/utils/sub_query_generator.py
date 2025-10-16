"""Sub-Query Generator: Generate sub-queries for complex queries."""

import json
from typing import List, Optional

import dspy


class SubQuerySignature(dspy.Signature):
    """Decompose complex queries into focused sub-queries for optimal retrieval."""

    original_query = dspy.InputField(
        desc="User's original complex search query requiring decomposition. "
        "Must identify distinct aspects, entities, and constraints that warrant separate searches."
    )
    num_subqueries = dspy.InputField(
        desc="Target number of sub-queries to generate (typically 2-5). "
        "Adjust based on query complexity: "
        "- Simple queries: 2 sub-queries "
        "- Medium complexity: 3 sub-queries "
        "- Highly complex: 4-5 sub-queries"
    )
    sub_queries = dspy.OutputField(
        desc="JSON array of optimized sub-queries. Each must: "
        "- Address a distinct aspect of the original query "
        "- Be self-contained for search execution "
        "- Preserve all critical constraints (dates, numbers, entities) "
        "- Exclude explanatory phrases "
        "- Maintain 5-12 word length "
        "- Output ONLY valid JSON array with no additional text"
    )


class SubQueryGenerator(dspy.Module):
    """Decompose complex search queries into targeted sub-queries for retrieval.

    This component:
    1. Identifies distinct aspects of complex queries
    2. Generates focused sub-queries for each aspect
    3. Preserves critical constraints across sub-queries
    4. Optimizes each sub-query for search engine effectiveness
    5. Balances coverage and specificity

    Example decomposition:
        Original: 'Compare the economic impact of renewable energy adoption in Germany
        vs France since 2020, focusing on job creation and GDP growth'
        Sub-queries:
        - 'Germany renewable energy economic impact job creation statistics 2020-2024'
        - 'France renewable energy economic impact job creation statistics 2020-2024'
        - 'Germany France renewable energy GDP growth comparison 2020-2024'
        - 'European Union renewable energy policy effects employment'
    """

    def __init__(self, min_subqueries: int = 2, max_subqueries: int = 5):
        """Initialize SubQueryGenerator.

        Args:
            min_subqueries: Minimum number of sub-queries to generate (default: 2)
            max_subqueries: Maximum number of sub-queries to generate (default: 5)
        """
        super().__init__()
        self.min_subqueries = min(2, min_subqueries)
        self.max_subqueries = max(5, max_subqueries)
        self.generator = dspy.ChainOfThought(SubQuerySignature)

    def _determine_complexity(self, query: str) -> int:
        """Determines optimal number of sub-queries based on query complexity."""
        # Simple heuristics for complexity assessment
        # Base complexity
        complexity = 1

        # Count key elements that increase complexity
        if any(
            word in query.lower() for word in ["compare", "versus", "vs", "difference"]
        ):
            complexity += 1
        if any(word in query.lower() for word in ["and", "&", "also"]):
            complexity += 1
        if len(query.split()) > 10:
            complexity += 1
        if any(char in query for char in [":", ";", ","]):
            complexity += 1

        # Map to target sub-query count
        return min(self.max_subqueries, max(self.min_subqueries, complexity))

    def forward(
        self, query: str, num_subqueries: Optional[int] = None
    ) -> dspy.Prediction:
        """Decomposes a query into targeted sub-queries.

        Args:
            query: Original complex search query
            num_subqueries: Optional override for number of sub-queries

        Returns:
            dspy.Prediction containing:
                - sub_queries: List of optimized sub-query strings
                - rationale: Reasoning steps for decomposition

        Raises:
            ValueError: If query decomposition fails
        """
        # Determine optimal sub-query count if not specified
        target_count = num_subqueries or self._determine_complexity(query)
        target_count = max(self.min_subqueries, min(self.max_subqueries, target_count))

        try:
            # Generate sub-queries
            result = self.generator(
                original_query=query, num_subqueries=str(target_count)
            )

            # Parse and validate JSON output
            sub_queries = json.loads(result.sub_queries)

            # Validate output structure
            if not isinstance(sub_queries, list) or not all(
                isinstance(q, str) for q in sub_queries
            ):
                raise ValueError("Invalid sub-queries format - must be list of strings")

            # Ensure we have reasonable number of sub-queries
            if len(sub_queries) < self.min_subqueries:
                # Fallback to single expanded query if decomposition failed
                return dspy.Prediction(
                    sub_queries=[self._fallback_rewrite(query)],
                    rationale="Decomposition failed - used single expanded query",
                )
            # Truncate to max if needed
            return dspy.Prediction(
                sub_queries=sub_queries[: self.max_subqueries],
                rationale=result.rationale,
            )

        except Exception as e:
            # Fallback to single query rewrite on error
            return dspy.Prediction(
                sub_queries=[self._fallback_rewrite(query)],
                rationale=f"Decomposition error: {str(e)}. Used fallback rewrite.",
            )

    def _fallback_rewrite(self, query: str) -> str:
        """Creates a single optimized query when decomposition fails."""
        # Simple rewrite removing conversational elements
        return " ".join(
            word
            for word in query.split()
            if word.lower() not in ["how", "what", "why", "i", "me", "my"]
        )

    def batch_generate(self, queries: List[str]) -> List[List[str]]:
        """Generates sub-queries for multiple complex queries.

        Args:
            queries: List of original complex search queries

        Returns:
            List of sub-query lists corresponding to each input query

        Example:
            >>> generator.batch_generate(
                [
                    "Compare iPhone 15 and Samsung S24 specs",
                    "Symptoms and treatments for seasonal allergies",
                ]
            )
            [
                [
                    "iPhone 15 specifications camera battery processor",
                    "Samsung Galaxy S24 specifications camera battery processor",
                ],
                [
                    "seasonal allergies symptoms nasal congestion sneezing",
                    "seasonal allergies treatment antihistamines nasal sprays",
                ],
            ]
        """
        return [self.forward(q).sub_queries for q in queries]
