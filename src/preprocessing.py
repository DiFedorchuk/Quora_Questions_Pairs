"""
Data preprocessing and feature engineering for Quora Question Pairs.
"""

import re
import warnings
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def _load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        import nltk

        downloaded = nltk.download("stopwords", quiet=True)
        if downloaded:
            try:
                return set(stopwords.words("english"))
            except LookupError:
                pass

        warnings.warn(
            "NLTK stopwords resource is unavailable. Falling back to sklearn English stopwords.",
            RuntimeWarning,
            stacklevel=2,
        )
        return set(ENGLISH_STOP_WORDS)


def _load_lemmatizer():
    import nltk

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    return WordNetLemmatizer()


# Initialize stopwords and lemmatizer
stops = _load_stopwords()
lemmatizer = _load_lemmatizer()


def tokenize(text):
    """
    Tokenize text by converting to lowercase, removing special characters,
    and filtering out stopwords.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w and w not in stops]


def preprocess_text(text):
    """
    Advanced text preprocessing with lemmatization.
    Steps:
    1. Convert to lowercase
    2. Remove punctuation and special characters
    3. Tokenize
    4. Remove stopwords
    5. Apply lemmatization
    
    Args:
        text: Raw text string
        
    Returns:
        str: Preprocessed and lemmatized text
    """
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Tokenization
    tokens = [w for w in tokens if w not in stops]  # Remove stop words
    tokens = [lemmatizer.lemmatize(w) for w in tokens]  # Lemmatization
    return " ".join(tokens)


def word_match_share(row):
    """
    Calculate the share of words that appear in both questions.
    
    Args:
        row: DataFrame row with 'question1' and 'question2' columns
        
    Returns:
        float: Ratio of shared words to total words (excluding stopwords)
    """
    q1words = {}
    q2words = {}

    for word in str(row["question1"]).lower().split():
        if word not in stops:
            q1words[word] = 1

    for word in str(row["question2"]).lower().split():
        if word not in stops:
            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:
        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    return (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))


def jaccard_similarity(row):
    """
    Calculate Jaccard similarity between two questions.
    
    Args:
        row: DataFrame row with 'question1' and 'question2' columns
        
    Returns:
        float: Jaccard similarity (intersection / union of tokenized words)
    """
    q1 = set(tokenize(row["question1"]))
    q2 = set(tokenize(row["question2"]))
    if not q1 or not q2:
        return 0.0
    return len(q1 & q2) / len(q1 | q2)


def common_word_count(row):
    """
    Count the number of common words between two questions.
    
    Args:
        row: DataFrame row with 'question1' and 'question2' columns
        
    Returns:
        int: Number of common words after tokenization
    """
    q1 = set(tokenize(row["question1"]))
    q2 = set(tokenize(row["question2"]))
    return len(q1 & q2)


def load_data(train_path, test_path):
    """
    Load training and test datasets from CSV files.
    
    Args:
        train_path: Path to training data CSV file
        test_path: Path to test data CSV file
        
    Returns:
        tuple: (df_train, df_test) DataFrames
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def engineer_features(df):
    """
    Engineer features for question pair similarity.
    
    Args:
        df: DataFrame with 'question1' and 'question2' columns
        
    Returns:
        DataFrame: Input DataFrame with new feature columns added:
                   - q1_len, q2_len: Word counts for each question
                   - len_diff: Absolute difference in word counts
                   - common_words: Number of shared words
                   - jaccard_sim: Jaccard similarity
                   - word_match_share: Proportion of shared words
    """
    df = df.copy()
    
    # Word length features
    df["q1_len"] = df["question1"].astype(str).apply(lambda x: len(x.split()))
    df["q2_len"] = df["question2"].astype(str).apply(lambda x: len(x.split()))
    df["len_diff"] = (df["q1_len"] - df["q2_len"]).abs()
    
    # Similarity features
    df["common_words"] = df.apply(common_word_count, axis=1)
    df["jaccard_sim"] = df.apply(jaccard_similarity, axis=1)
    df["word_match_share"] = df.apply(word_match_share, axis=1)
    
    return df


def get_question_frequency_features(df):
    """
    Calculate question frequency-based features.
    
    Args:
        df: DataFrame with 'question1' and 'question2' columns
        
    Returns:
        DataFrame: Input DataFrame with frequency features added:
                   - max_q_freq: Maximum frequency of the two questions
                   - min_q_freq: Minimum frequency of the two questions
    """
    df = df.copy()
    
    # Calculate question frequencies
    all_questions = pd.Series(pd.concat([df["question1"], df["question2"]]).astype(str))
    question_freq = all_questions.value_counts()
    
    q1_freq = df["question1"].map(question_freq)
    q2_freq = df["question2"].map(question_freq)
    
    df["max_q_freq"] = pd.concat([q1_freq, q2_freq], axis=1).max(axis=1)
    df["min_q_freq"] = pd.concat([q1_freq, q2_freq], axis=1).min(axis=1)
    
    return df


def get_text_stats(questions):
    """
    Get text statistics for a series of questions.
    
    Args:
        questions: pandas Series of question texts
        
    Returns:
        dict: Dictionary with character and word count statistics
    """
    qs = pd.Series(questions).astype(str)
    
    char_len = qs.apply(len)
    word_len = qs.apply(lambda x: len(x.split()))
    
    return {
        "char_mean": char_len.mean(),
        "char_std": char_len.std(),
        "char_max": char_len.max(),
        "word_mean": word_len.mean(),
        "word_std": word_len.std(),
        "word_max": word_len.max(),
    }


def get_semantic_features(questions):
    """
    Analyze semantic features of questions (question marks, math tags, etc).
    
    Args:
        questions: pandas Series of question texts
        
    Returns:
        dict: Dictionary with semantic feature percentages
    """
    qs = pd.Series(questions).astype(str)
    
    return {
        "qmarks": np.mean(qs.apply(lambda x: "?" in x)) * 100,
        "math_tags": np.mean(qs.apply(lambda x: "[math]" in x)) * 100,
        "fullstop": np.mean(qs.apply(lambda x: "." in x)) * 100,
        "capital_first": np.mean(qs.apply(lambda x: x[0].isupper())) * 100,
        "capitals": np.mean(qs.apply(lambda x: any(y.isupper() for y in x))) * 100,
        "numbers": np.mean(qs.apply(lambda x: any(y.isdigit() for y in x))) * 100,
    }


def get_overlap_stats(df_train, df_test):
    """
    Calculate overlap statistics between train and test datasets.
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        
    Returns:
        dict: Dictionary with overlap statistics
    """
    train_questions = pd.Index(pd.unique(pd.concat([df_train["question1"], df_train["question2"]]).astype(str)))
    test_questions = pd.Index(pd.unique(pd.concat([df_test["question1"], df_test["question2"]]).astype(str)))
    overlap = train_questions.intersection(test_questions)
    
    return {
        "train_unique_questions": len(train_questions),
        "test_unique_questions": len(test_questions),
        "overlap_count": len(overlap),
        "overlap_ratio_train": len(overlap) / len(train_questions),
        "overlap_ratio_test": len(overlap) / len(test_questions),
    }


def ensure_wordnet_resource():
    """Explicitly download the NLTK wordnet resource when needed."""
    import nltk

    nltk.download("wordnet")


def get_duplicate_ratio_percent(df, label_column="is_duplicate"):
    """
    Compute duplicate-pair percentage for a labeled dataset.

    Args:
        df: DataFrame with duplicate labels
        label_column: Name of binary duplicate label column

    Returns:
        float: Duplicate percentage in [0, 100]
    """
    return round(df[label_column].mean() * 100, 2)


def print_duplicate_ratio(df, label_column="is_duplicate"):
    """
    Print duplicate-pair percentage in a consistent format.

    Args:
        df: DataFrame with duplicate labels
        label_column: Name of binary duplicate label column
    """
    ratio = get_duplicate_ratio_percent(df, label_column=label_column)
    print('Duplicate pairs(is_duplicate = 1): {}%'.format(ratio))
