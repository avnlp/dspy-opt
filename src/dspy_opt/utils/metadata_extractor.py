"""Metadata Extractor: Extracts structured metadata from text based on a JSON schema."""

import json
from typing import Any, Dict, List

import dspy


class ExtractMetadataSignature(dspy.Signature):
    """Extract structured metadata from text based on a JSON schema."""

    text = dspy.InputField(
        desc="The input text to extract metadata from. For any field not"
        " explicitly mentioned in the text, omit that field. Do NOT use "
        "placeholders like 'Unknown', 'N/A', or make up values. Only use "
        "facts directly stated in the input."
    )
    metadata_schema = dspy.InputField(
        desc="JSON schema defining the expected metadata structure"
    )
    metadata = dspy.OutputField(
        desc="JSON object containing only successfully extracted fields (non-null values)"
    )


class MetadataExtractor(dspy.Module):
    """Extracts structured metadata from text using a language model and a JSON schema.

    This class leverages a function-calling or structured-output-capable LLM to extract
    metadata fields defined by a user-provided JSON schema. The schema is dynamically
    converted into a Pydantic model, which is used to enforce type safety and
    validation. Only fields that are successfully extracted (i.e., not null) are
    included in the result.

    This metadata can be used for filtering search results, or in the case of queries,
    it can be used to filter out documents that are not relevant to the query.
    """

    def __init__(self, extractor_llm: dspy.LM):
        """Initializes the MetadataExtractor with a language model."""
        super().__init__()
        self.extractor_llm = extractor_llm

    def validate_schema(self, schema: Dict[str, Any]) -> None:
        """Validates the schema structure and types."""
        if "properties" not in schema:
            raise ValueError("Schema must contain a 'properties' key.")

        allowed_types = {"string", "number", "boolean"}
        properties = schema["properties"]

        for name, spec in properties.items():
            json_type = spec.get("type", "string")
            if json_type not in allowed_types:
                raise ValueError(
                    f"Unsupported type '{json_type}' for field '{name}'. "
                    f"Only {allowed_types} are allowed."
                )
            if "enum" in spec and json_type != "string":
                raise ValueError(
                    f"Enum is only supported for string fields. "
                    f"Field '{name}' has type '{json_type}'."
                )

    def forward(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts metadata from text according to the provided schema."""
        self.validate_schema(schema)

        # Convert schema to JSON string for LLM consumption
        schema_str = json.dumps(schema)

        try:
            # Get structured output from LLM
            with dspy.context(lm=self.extractor_llm):
                extractor = dspy.Predict(ExtractMetadataSignature)
                result = extractor(text=text, metadata_schema=schema_str)
                metadata = json.loads(result.metadata)

            # Filter out null values and invalid fields
            validated = {}
            for field in schema["properties"]:
                if field in metadata and metadata[field] is not None:
                    validated[field] = metadata[field]
            return validated
        except Exception:
            return {}

    def transform_documents(
        self, documents: List[dspy.Example], schema: Dict[str, Any]
    ) -> List[dspy.Example]:
        """Applies metadata extraction to a list of DSPy examples/documents."""
        transformed_documents = []

        for doc in documents:
            extracted = self(doc.text, schema)
            new_metadata = {**doc.metadata, **extracted}
            transformed_documents.append(
                dspy.Example(text=doc.text, metadata=new_metadata).with_inputs("text")
            )

        return transformed_documents
