from dspy.teleprompt import BootstrapFewShotWithRandomSearch, Ensemble


def optimize_pipeline(rag_module, metric, train_set, val_set):
    teleprompter = BootstrapFewShotWithRandomSearch(
        metric=metric, num_candidate_programs=6, max_bootstrapped_demos=3, num_threads=2
    )
    return teleprompter.compile(rag_module, trainset=train_set, valset=val_set)


def create_ensemble(teleprompter, retriever, train_set, val_set):
    models = [WikipediaRAG(retriever, num_passages=k) for k in [2, 3, 4]]
    ensemble = Ensemble(models=models, size=3)
    return teleprompter.compile(ensemble, trainset=train_set, valset=val_set)
