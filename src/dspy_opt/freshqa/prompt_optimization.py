import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch


def optimize_pipeline(pipeline, metric, trainset, config):
    teleprompter = BootstrapFewShotWithRandomSearch(
        metric=metric,
        max_bootstrapped_demos=config["max_bootstrapped_demos"],
        num_candidate_programs=config["num_candidate_programs"],
        num_threads=config["num_threads"],
        teacher_settings={"lm": dspy.settings.lm},
    )
    return teleprompter.compile(pipeline, trainset=trainset)
