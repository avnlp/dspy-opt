import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, Ensemble

from .retreiver_and_rag import PubMedRAG


def optimize_pipeline(rag_module, metric, config, train_set, val_set):
    teleprompter = BootstrapFewShotWithRandomSearch(
        metric=metric,
        num_candidate_programs=config["optimization"].get("num_candidate_programs", 8),
        max_bootstrapped_demos=config["optimization"].get("max_bootstrapped_demos", 5),
        num_threads=config["optimization"].get("num_threads", 4),
        teacher_settings={"lm": dspy.OpenAI(model=config["optimization"]["teacher_model"])},
    )
    return teleprompter.compile(rag_module, trainset=train_set, valset=val_set)


def create_ensemble(teleprompter, config, train_set, val_set):
    models = []
    for k in [3, 5, 7]:  # Different numbers of passages
        model = PubMedRAG(num_passages=k)
        models.append(model)

    ensemble = Ensemble(models=models, size=3)
    return teleprompter.compile(ensemble, trainset=train_set, valset=val_set)
