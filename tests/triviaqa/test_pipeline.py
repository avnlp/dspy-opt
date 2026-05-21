"""Tests for the TriviaQA pipeline."""

import importlib
import runpy
from typing import Any, Dict

import pytest
import pytest_mock

from tests.helpers import (
    DummyDspyModule,
    DummyEmbeddingModel,
    DummyPrediction,
    DummyRetriever,
)
from tests.script_helpers import (
    apply_common_patches,
    config_with_overrides,
    patch_prediction_modules,
    patch_rag_pipeline,
)


class TestPipeline:
    """TriviaQA pipeline coverage."""

    def test_rag_pipeline_success_path(self, mocker: pytest_mock.MockFixture) -> None:
        """Validate the happy path through the TriviaQA RAG pipeline."""
        module = importlib.import_module("dspy_opt.triviaqa.triviaqa_rag_module")
        rewriter = DummyDspyModule(DummyPrediction(rewritten_query="rewrite"))
        sub_query = DummyDspyModule(DummyPrediction(sub_queries=["sub", "sub"]))
        extractor = DummyDspyModule(DummyPrediction())
        retriever = DummyRetriever(["passage", "passage", "extra"])
        answer = DummyDspyModule(DummyPrediction(answer="final", reasoning="why"))
        mocker.patch.object(module.dspy, "ChainOfThought", return_value=answer)
        pipeline = module.TriviaQARAG(
            query_rewriter=rewriter,
            sub_query_generator=sub_query,
            metadata_extractor=extractor,
            metadata_schema={},
            weaviate_retriever=retriever,
            embedding_model=DummyEmbeddingModel(),
            top_k=2,
        )

        result = pipeline.forward("question?")

        assert result.rewritten_query == "rewrite"
        assert result.sub_queries == ["sub", "sub"]
        assert result.answer == "final"
        assert result.retrieved_context == ["passage", "extra"]
        assert retriever.calls[0]["query"] == "rewrite"
        assert retriever.calls[1]["query"] == "sub"

    def test_rag_pipeline_fallback_on_exception(
        self, mocker: pytest_mock.MockFixture
    ) -> None:
        """Ensure fallback logic returns a recovery prediction on failure."""
        module = importlib.import_module("dspy_opt.triviaqa.triviaqa_rag_module")
        rewriter = DummyDspyModule(DummyPrediction(rewritten_query="rewrite"))
        sub_query = DummyDspyModule(DummyPrediction(sub_queries=["sub"]))
        extractor = DummyDspyModule(DummyPrediction())
        retriever = DummyRetriever(["passage"])
        answer = DummyDspyModule(DummyPrediction(answer="fallback", reasoning="why"))
        mocker.patch.object(module.dspy, "ChainOfThought", return_value=answer)
        prediction_mock = mocker.Mock(
            side_effect=[
                RuntimeError("boom"),
                DummyPrediction(answer="fallback", reasoning="Error recovery: boom"),
            ]
        )
        mocker.patch.object(module.dspy, "Prediction", prediction_mock)
        pipeline = module.TriviaQARAG(
            query_rewriter=rewriter,
            sub_query_generator=sub_query,
            metadata_extractor=extractor,
            metadata_schema={},
            weaviate_retriever=retriever,
            embedding_model=DummyEmbeddingModel(),
        )

        result = pipeline.forward("question?")

        assert result.answer == "fallback"
        assert "Error recovery" in result.reasoning

    @pytest.mark.parametrize(
        "module_path, rag_class, overrides",
        [
            (
                "dspy_opt.triviaqa.triviaqa_rag_mipro",
                "TriviaQARAG",
                {
                    "train_dataset": {"name": "name", "split": "train"},
                    "test_dataset": {"name": "name", "split": "test"},
                },
            ),
            (
                "dspy_opt.triviaqa.triviaqa_rag_copro",
                "TriviaQARAG",
                {
                    "train_dataset": {"name": "name", "split": "train"},
                    "test_dataset": {"name": "name", "split": "test"},
                },
            ),
            (
                "dspy_opt.triviaqa.triviaqa_rag_simba",
                "TriviaQARAG",
                {
                    "train_dataset": {"name": "name", "split": "train"},
                    "test_dataset": {"name": "name", "split": "test"},
                },
            ),
            (
                "dspy_opt.triviaqa.triviaqa_rag_bootstrap_few_shot",
                "TriviaQARAG",
                {
                    "train_dataset": {"name": "name", "split": "train"},
                    "test_dataset": {"name": "name", "split": "test"},
                    "optimizer": {
                        "max_bootstrapped_demos": 1,
                        "max_labeled_demos": 1,
                        "max_rounds": 1,
                    },
                },
            ),
            (
                "dspy_opt.triviaqa.triviaqa_rag_gepa",
                "TriviaQARAG",
                {
                    "train_dataset": {"name": "name", "split": "train"},
                    "test_dataset": {"name": "name", "split": "test"},
                },
            ),
            (
                "dspy_opt.triviaqa.triviaqa_rag_evaluation",
                "TriviaQARAG",
                {
                    "test_dataset": {"name": "name", "split": "test"},
                    "dataset": {
                        "name": "name",
                        "subset": "subset",
                        "split": "train",
                        "test_size": 0.1,
                    },
                },
            ),
        ],
    )
    def test_script_main_runs(
        self,
        mocker: pytest_mock.MockFixture,
        module_path: str,
        rag_class: str,
        overrides: Dict[str, Any],
    ) -> None:
        """Run main entry points to confirm scripts stay importable."""
        module = importlib.import_module(module_path)
        apply_common_patches(module, mocker)
        patch_prediction_modules(module, mocker)
        patch_rag_pipeline(module, mocker, rag_class)
        if overrides:
            mocker.patch.object(
                module.yaml,
                "safe_load",
                return_value=config_with_overrides(**overrides),
            )

        module.main()

    def test_indexing_script_runs(self, mocker: pytest_mock.MockFixture) -> None:
        """Execute the indexing script with patched external dependencies."""
        weaviate_module = mocker.Mock()
        weaviate_module.connect_to_weaviate_cloud.return_value = mocker.Mock(
            is_ready=mocker.Mock(return_value=True),
            collections=mocker.Mock(
                exists=mocker.Mock(return_value=False),
                create=mocker.Mock(return_value=mocker.Mock()),
                use=mocker.Mock(
                    return_value=mocker.Mock(
                        data=mocker.Mock(insert_many=mocker.Mock())
                    )
                ),
            ),
            close=mocker.Mock(),
        )
        wvc_module = mocker.Mock()
        wvc_module.init.Auth.api_key.return_value = mocker.Mock()
        wvc_module.config.Configure.Vectors.self_provided.return_value = "vectors"
        wvc_module.data.DataObject.side_effect = lambda properties, vector: {
            "properties": properties,
            "vector": vector,
        }
        mocker.patch.dict(
            "sys.modules",
            {
                "weaviate": weaviate_module,
                "weaviate.classes": wvc_module,
            },
        )
        dummy_dataset = {"context": ["doc1", "doc2"]}
        mocker.patch.dict(
            "sys.modules",
            {
                "datasets": mocker.Mock(
                    load_dataset=mocker.Mock(return_value=dummy_dataset)
                )
            },
        )
        sentence_transformers_module = mocker.Mock()
        sentence_transformers_module.SentenceTransformer.return_value = mocker.Mock(
            encode=mocker.Mock(return_value=[[0.1, 0.2, 0.3]])
        )
        mocker.patch.dict(
            "sys.modules",
            {"sentence_transformers": sentence_transformers_module},
        )
        mocker.patch.dict(
            "sys.modules",
            {
                "dspy_opt.utils.metadata_extractor": mocker.Mock(
                    MetadataExtractor=mocker.Mock(
                        return_value=mocker.Mock(
                            transform_documents=lambda docs, _schema: docs
                        )
                    )
                )
            },
        )
        mocker.patch.dict(
            "sys.modules",
            {"dotenv": mocker.Mock(load_dotenv=mocker.Mock())},
        )
        mocker.patch("builtins.open", mocker.mock_open(read_data="{}"))
        mocker.patch("os.getenv", return_value="value")
        mocker.patch(
            "yaml.safe_load",
            return_value={
                "dataset": {"name": "name", "split": "train"},
                "metadata_schema": {},
                "extractor_llm": {"model": "m"},
                "embedding": {"embedding_model": "m", "tokenizer_kwargs": {}},
                "weaviate": {
                    "collection_name": "Collection",
                },
                "document_encoding": {"batch_size": 1, "show_progress_bar": False},
            },
        )

        runpy.run_module("dspy_opt.triviaqa.triviaqa_indexing", run_name="__main__")
