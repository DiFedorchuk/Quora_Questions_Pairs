"""Project launcher for Quora Question Pairs.

Runs notebooks in order:
1. notebooks/01_eda.ipynb
2. notebooks/02_baseline.ipynb
3. notebooks/03_models.ipynb
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
NOTEBOOKS_DIR = ROOT / "notebooks"
REPORTS_DIR = ROOT / "reports" / "executed_notebooks"
TRAIN_DATA = DATA_DIR / "quora_question_pairs_train.csv.zip"
TEST_DATA = DATA_DIR / "quora_question_pairs_test.csv.zip"
NOTEBOOK_ORDER = [
    NOTEBOOKS_DIR / "01_eda.ipynb",
    NOTEBOOKS_DIR / "02_baseline.ipynb",
    NOTEBOOKS_DIR / "03_models.ipynb",
]


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
    return 0


def execute_notebook(notebook_path: Path, timeout: int) -> int:
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError as exc:
        print("Notebook execution dependencies are missing. Install requirements first:")
        print("pip install -r requirements.txt")
        print(f"Details: {exc}")
        return 1

    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}")
        return 1

    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    preprocessor = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    print(f"Running {notebook_path.relative_to(ROOT)} ...")

    try:
        preprocessor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    except Exception as exc:
        print(f"Failed while executing {notebook_path.name}: {exc}")
        return 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / notebook_path.name
    with output_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)

    print(f"Completed {notebook_path.name} (saved to {output_path.relative_to(ROOT)})")
    return 0


def run_notebooks(timeout: int, stop_on_error: bool) -> int:
    for notebook_path in NOTEBOOK_ORDER:
        status = execute_notebook(notebook_path, timeout)
        if status != 0 and stop_on_error:
            return status
    return 0


def clean_executed_reports() -> int:
    if not REPORTS_DIR.exists():
        print("No executed notebook reports to clean.")
        return 0
    shutil.rmtree(REPORTS_DIR)
    print(f"Removed {REPORTS_DIR}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Quora project pipeline")
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check that expected dataset files exist.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny feature-engineering sanity check.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not execute notebooks (only run selected checks).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-notebook execution timeout in seconds (default: 1800).",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with next notebooks even if one fails.",
    )
    parser.add_argument(
        "--clean-reports",
        action="store_true",
        help="Remove reports/executed_notebooks and exit.",
    )
    args = parser.parse_args()

    if args.clean_reports:
        return clean_executed_reports()

    if args.check_data:
        status = check_data()
        if status != 0:
            return status

    if args.smoke_test:
        status = smoke_test()
        if status != 0:
            return status

    if args.no_run:
        print("Notebook execution skipped (--no-run).")
        return 0

    if not args.check_data:
        status = check_data()
        if status != 0:
            return status

    return run_notebooks(timeout=args.timeout, stop_on_error=not args.keep_going)


if __name__ == "__main__":
    sys.exit(main())
