from dspy.teleprompt import BootstrapFewShotWithRandomSearch, Ensemble


def optimize_pipeline(rag_module, metric, train_set, val_set):
    teleprompter = BootstrapFewShotWithRandomSearch(
        metric=metric, num_candidate_programs=8, max_bootstrapped_demos=6, num_threads=4
    )
    return teleprompter.compile(rag_module, trainset=train_set, valset=val_set)


def create_ensemble(teleprompter, train_set, val_set):
    models = [TriviaRAG(num_passages=k) for k in [3, 5, 7]]
    ensemble = Ensemble(models=models, size=3)
    return teleprompter.compile(ensemble, trainset=train_set, valset=val_set)
