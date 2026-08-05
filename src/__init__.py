"""
Quora Question Pairs - Source code package.
"""

from .preprocessing import (
    tokenize,
    preprocess_text,
    word_match_share,
    jaccard_similarity,
    common_word_count,
    load_data,
    engineer_features,
    get_question_frequency_features,
    get_text_stats,
    get_semantic_features,
    get_overlap_stats,
)

from .models import (
    QuestionPairModel,
    LogisticRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    SVMModel,
)

from .evaluation import (
    evaluate_model,
    print_evaluation_report,
)

__all__ = [
    # preprocessing
    "tokenize",
    "preprocess_text",
    "word_match_share",
    "jaccard_similarity",
    "common_word_count",
    "load_data",
    "engineer_features",
    "get_question_frequency_features",
    "get_text_stats",
    "get_semantic_features",
    "get_overlap_stats",
    # models
    "QuestionPairModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "SVMModel",
    # evaluation
    "evaluate_model",
    "print_evaluation_report",
]
