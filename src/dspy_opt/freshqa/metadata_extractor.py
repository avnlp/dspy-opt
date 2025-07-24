import dspy

from .prompts import QueryMetadataExtractor


class QueryMetadataEnhancer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_metadata = dspy.Predict(QueryMetadataExtractor)

    def forward(self, question):
        metadata = self.extract_metadata(question=question)
        return dspy.Prediction(entities=metadata.entities, time_period=metadata.time_period, domain=metadata.domain)


class EnhancedRetriever(dspy.Retrieve):
    def __init__(self, base_retriever, metadata_extractor):
        super().__init__()
        self.base_retriever = base_retriever
        self.metadata_extractor = metadata_extractor

    def forward(self, question):
        # Extract metadata
        metadata = self.metadata_extractor(question)

        # Enhance query with metadata
        enhanced_query = f"{question}"
        if metadata.domain:
            enhanced_query += f" [Domain: {metadata.domain}]"
        if metadata.time_period:
            enhanced_query += f" [Time: {metadata.time_period}]"
        if metadata.entities:
            enhanced_query += f" [Entities: {', '.join(metadata.entities)}]"

        # Retrieve with enhanced query
        return self.base_retriever(enhanced_query)
