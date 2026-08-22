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
    ensure_wordnet_resource,
    get_duplicate_ratio_percent,
    print_duplicate_ratio,
)

from .models import (
    QuestionPairModel,
    LogisticRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    SVMModel,
    ENSEMBLE_MODEL_ORDER,
    build_ensemble_weights,
    save_ensemble_components,
)

try:
    from .evaluation import (
        evaluate_model,
        print_evaluation_report,
        compute_f1_and_logloss,
        build_model_comparison_dataframe,
        display_model_comparison,
    )
except ImportError:
    evaluate_model = None
    print_evaluation_report = None
    compute_f1_and_logloss = None
    build_model_comparison_dataframe = None
    display_model_comparison = None

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
    "ensure_wordnet_resource",
    "get_duplicate_ratio_percent",
    "print_duplicate_ratio",
    # models
    "QuestionPairModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "SVMModel",
    "ENSEMBLE_MODEL_ORDER",
    "build_ensemble_weights",
    "save_ensemble_components",
    # evaluation
]
