"""Weaviate Retriever: Retrieves relevant passages from Weaviate using Hybrid Search."""

from typing import Any, Dict, Optional

import dspy
import numpy as np
import weaviate
import weaviate.classes as wvc


class WeaviateRetriever(dspy.Module):
    """Custom Weaviate retriever for DSPy."""

    def __init__(
        self,
        weaviate_url: Optional[str] = None,
        weaviate_api_key: Optional[str] = None,
        collection_name: str = "Triviaqa",
        top_k: int = 3,
        metadata_schema: Optional[dict[str, dict[str, object]]] = None,
    ):
        """Initialize Weaviate retriever.

        Args:
            weaviate_url: URL to Weaviate cluster (e.g., WCS endpoint)
            weaviate_api_key: API key for authentication
            collection_name: Name of the Weaviate collection
            top_k: Default number of passages to retrieve
            metadata_schema: Optional schema dict mapping metadata keys to types
                            (e.g., {"content_type": str, "year": int})
        """
        super().__init__()
        self.collection_name = collection_name
        self.top_k = top_k
        self.weaviate_url = weaviate_url
        self.weaviate_api_key = weaviate_api_key
        self.metadata_schema = metadata_schema or {}

        if not weaviate_url or not weaviate_api_key:
            raise ValueError("Weaviate URL and API key are required")

        try:
            # Initialize Weaviate client
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.weaviate_url,  # type: ignore[arg-type]
                auth_credentials=wvc.init.Auth.api_key(self.weaviate_api_key),  # type: ignore[arg-type]
            )
        except Exception as e:
            raise ConnectionError(f"Could not connect to Weaviate: {str(e)}") from e

        if not self.client.is_ready():
            raise ConnectionError(
                "Connection to Weaviate failed: client not ready"
            ) from None

        try:
            self.collection = self.client.collections.use(collection_name)
        except Exception as e:
            raise ValueError(
                f"Collection '{collection_name}' does not exist: {str(e)}"
            ) from e

    def metadata_to_weaviate_filter(
        self, metadata: Dict[str, Any]
    ) -> Optional[wvc.query.Filter]:
        """Convert metadata dictionary to Weaviate filter.

        Only includes fields present in self.metadata_schema.
        Supports exact matching for now.

        Args:
            metadata: Dictionary of extracted metadata

        Returns:
            Weaviate Filter object or None
        """
        filters = []
        for key, val in metadata.items():
            # Skip unknown metadata fields
            if key not in self.metadata_schema:
                continue

            expected_type = self.metadata_schema[key]
            if not isinstance(val, expected_type):  # type: ignore[arg-type]
                try:
                    # Attempt coercion
                    val = expected_type(val)  # type: ignore[operator] # noqa: PLW2901
                except (ValueError, TypeError):
                    # Skip invalid type
                    continue

            prop_filter = wvc.query.Filter.by_property(key).like(val)
            filters.append(prop_filter)

        return wvc.query.Filter.and_(*filters) if filters else None  # type: ignore[attr-defined]

    def forward(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dspy.Prediction:
        """Retrieve relevant passages from Weaviate.

        Args:
            query: The user query string
            query_embedding: Optional precomputed embedding vector
            top_k: Number of results to return (overrides default if provided)
            metadata: Optional metadata to filter results by

        Returns:
            dspy.Prediction with list of retrieved passages
        """
        try:
            if not query or not query.strip():
                return dspy.Prediction(passages=[])

            # Use provided metadata or default to empty dict
            query_metadata = metadata or {}

            # Build filter from metadata
            weaviate_filter = self.metadata_to_weaviate_filter(query_metadata)

            # Perform hybrid search
            limit = top_k or self.top_k
            response = self.collection.query.hybrid(
                query=query,
                vector=query_embedding,  # type: ignore[arg-type]
                limit=limit,
                filters=weaviate_filter,  # type: ignore[arg-type]
                return_metadata=wvc.query.MetadataQuery(certainty=True),
            )

            # Extract non-empty document texts
            passages = [
                obj.properties["document_text"]
                for obj in response.objects
                if obj.properties.get("document_text", "")
            ]

            return dspy.Prediction(passages=passages)

        except Exception as e:
            print(f"Error retrieving from Weaviate: {str(e)}")
            return dspy.Prediction(passages=[])
