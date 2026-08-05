"""Lightweight project launcher for Quora Question Pairs.

This script does not execute the full notebook workflow. It only provides:
- a quick run-order summary
- an optional small smoke test on synthetic data
- an optional dataset presence check
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_DATA = DATA_DIR / "quora_question_pairs_train.csv.zip"
TEST_DATA = DATA_DIR / "quora_question_pairs_test.csv.zip"


def print_run_order() -> None:
    print("Recommended project order:")
    print("1. notebooks/01_eda.ipynb")
    print("2. notebooks/02_baseline.ipynb")
    print("3. notebooks/03_models.ipynb")
    print()
    print("This helper intentionally does not run the full project.")
    print("Use --smoke-test for a tiny feature-engineering sanity check.")


def check_data() -> int:
    missing = [path for path in (TRAIN_DATA, TEST_DATA) if not path.exists()]
    if missing:
        print("Missing data files:")
        for path in missing:
            print(f"- {path}")
        return 1

    print("All expected data files are present.")
    return 0


def smoke_test() -> int:
    import pandas as pd

    from src import engineer_features, get_question_frequency_features

    sample = pd.DataFrame(
        {
            "question1": ["What is AI?", "How are you?"],
            "question2": ["What is artificial intelligence?", "How do you do?"],
            "is_duplicate": [1, 0],
        }
    )

    engineered = engineer_features(sample)
    engineered = get_question_frequency_features(engineered)

    expected_columns = {
        "question1",
        "question2",
        "is_duplicate",
        "q1_len",
        "q2_len",
        "len_diff",
        "common_words",
        "jaccard_sim",
        "word_match_share",
        "max_q_freq",
        "min_q_freq",
    }

    missing_columns = expected_columns.difference(engineered.columns)
    if missing_columns:
        print("Smoke test failed. Missing columns:")
        for column in sorted(missing_columns):
            print(f"- {column}")
        return 1

    print("Smoke test passed.")
    print("Engineered columns:")
    print(", ".join(engineered.columns))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight launcher for the Quora project")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny feature-engineering sanity check on synthetic data.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check that the expected dataset files exist.",
    )
    args = parser.parse_args()

    if not args.smoke_test and not args.check_data:
        print_run_order()
        return 0

    if args.check_data:
        status = check_data()
        if status != 0:
            return status

    if args.smoke_test:
        return smoke_test()

    return 0


if __name__ == "__main__":
    sys.exit(main())
