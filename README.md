# Quora Question Pairs - Duplicate Detection

This project aims to identify duplicate question pairs in the Quora dataset. The dataset contains question pairs and labels indicating whether they are duplicates (same intent and meaning).

## Project Overview

The Quora Question Pairs dataset consists of:
- **Training set**: 323,432 question pairs
- **Test set**: 80,858 question pairs  
- **Target variable**: `is_duplicate` (binary classification)
- **Class distribution**: Imbalanced (about 37% duplicates in training set)

### Key Insights from EDA

1. **Data Overlap**: 53% of unique test questions appear in the training set
2. **Text Characteristics**:
   - Average question length: ~11 words, ~60 characters
   - Range: 1-100+ words
3. **Question Frequency**: Most questions appear only once; some high-frequency questions (e.g., "What is...") appear hundreds of times
4. **Semantic Features**: 
   - ~95% contain question marks
   - ~50% have capital letters
   - ~20% contain numbers
5. **Similarity Patterns**: Duplicate question pairs show significantly higher:
   - Word match share (~0.65 vs ~0.35 for non-duplicates)
   - Jaccard similarity (~0.6 vs ~0.25 for non-duplicates)
   - Common word count

## Project Structure

```
quora-question-pairs/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── data/                          # Dataset files
│   ├── quora_question_pairs_train.csv.zip
│   └── quora_question_pairs_test.csv.zip
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_baseline.ipynb         # Baseline models
│   └── 03_models.ipynb           # Advanced models
├── src/                           # Python source code
│   ├── __init__.py
│   ├── preprocessing.py           # Data loading and feature engineering
│   ├── models.py                  # Machine learning models
│   └── evaluation.py              # Evaluation metrics and utilities
├── models/                        # Saved trained models

## Dependencies

See `requirements.txt` for the complete list. Key dependencies:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning algorithms and metrics
- **matplotlib/seaborn**: Data visualization
- **nltk**: Natural language processing (tokenization, stopwords)
- **xgboost/lightgbm**: Gradient boosting models
- **torch/transformers**: Deep learning and BERT-based models

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd quora-question-pairs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
```

4. Run the project pipeline:
```bash
python run_project.py
```

## Usage

### Pipeline runner
```bash
python run_project.py
python run_project.py --check-data
python run_project.py --smoke-test
python run_project.py --no-run --check-data --smoke-test
python run_project.py --timeout 3600
```
By default, `run_project.py` executes notebooks in this order:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_baseline.ipynb`
3. `notebooks/03_models.ipynb`


## Feature Engineering

The project includes several engineered features:

### Word-based Features
- **q1_len, q2_len**: Word count in each question
- **len_diff**: Absolute difference in word lengths
- **common_words**: Count of shared words between questions
- **word_match_share**: Proportion of words shared between questions

### Similarity Features
- **jaccard_sim**: Jaccard similarity between tokenized words
- **word_match_share**: Word overlap ratio

### Frequency Features
- **max_q_freq**: Maximum frequency of the two questions
- **min_q_freq**: Minimum frequency of the two questions

## Models

The project includes several baseline and advanced models:

- **Logistic Regression**
- **XGBoost**
- **LightGBM**
- **LSTM**
- **Fine-tuned BERT**

## Notebooks

- **01_eda.ipynb**: Comprehensive exploratory data analysis
  - Data loading and exploration
  - Duplicate distribution analysis
  - Text length and semantic analysis
  - Feature engineering and visualization
  - Question frequency and overlap analysis

- **02_baseline.ipynb**: Baseline models and evaluation
  - Simple similarity-based approaches
  - Logistic Regression baseline
  - Performance comparison

- **03_models.ipynb**: Advanced models
  - Baseline Models: Logistic Regression
  - Feature Engineering: Bag of Words (BOW), TF-IDF, BERT Embedding
  - Tree-based Models: XGBoost, LightGBM
  - Deep Learning: LSTM, Fine-tuned BERT
  - Ensemble: Weighted averaging of top models

## Results

### Model Metrics

| Model | Log Loss | F1-score | Model parameters |
| --- | --- | --- | --- |
| Baseline Logistic Regression | 0.42 | 0.69 | `LogisticRegression(random_state=42, solver='liblinear')`; features: `len_diff, common_words, jaccard_sim, word_match_share, max_q_freq, min_q_freq` |
| Bag of Words + Logistic Regression | 0.60 | 0.61 | `CountVectorizer(max_features=10000, stop_words=list(stops))` + `StandardScaler(with_mean=False)` + `LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)` |
| TF-IDF + Logistic Regression | 0.62 | 0.61 | `TfidfVectorizer(max_features=10000, stop_words=list(stops))` + `StandardScaler(with_mean=False)` + `LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)` |
| BERT Embeddings + Logistic Regression | 0.57 | 0.55 | Embeddings from `distilbert-base-uncased` (`max_length=128`, embedding `batch_size=64`, chunk size `500`, sampled rows `50000`) + `LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)` |
| LSTM | 0.50 | 0.69 | Siamese LSTM with `VOCAB_SIZE=20000`, `MAX_SEQUENCE_LENGTH=30`, `EMBEDDING_DIM=100`, `HIDDEN_DIM=128`, `NUM_LAYERS=2`, `DROPOUT=0.5`, `BATCH_SIZE=64`, `N_EPOCHS=5`, optimizer `Adam` |
| XGBoost (tuned) | 0.33 | 0.76 | `XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42, learning_rate=0.1, max_depth=5, n_estimators=300)` |
| LightGBM (tuned) | 0.33 | 0.75 | `LGBMClassifier(objective='binary', metric='logloss', random_state=42, learning_rate=0.1, max_depth=10, n_estimators=100, num_leaves=31)` |
| Fine-tuned BERT | 0.42 | 0.81 | `AutoModelForSequenceClassification('distilbert-base-uncased', num_labels=2)` with `MAX_LEN=128`, `TRAIN_BATCH_SIZE=16`, `EVAL_BATCH_SIZE=32`, `EPOCHS=3`, `LEARNING_RATE=2e-5`, `EPS=1e-8`, fine-tuning sample `50000` |
| Weighted Ensemble | 0.28 | 0.85 | Weighted average of probabilities from fine-tuned BERT + tuned XGBoost + tuned LightGBM with weights `BERT=0.4`, `XGBoost=0.3`, `LightGBM=0.3`; decision threshold `0.5` |

Key findings from analysis:
- ~63% of question pairs are duplicates (37% non-duplicates)
- Strong correlation between lexical similarity and duplicate status
- Question frequency is a significant differentiator
- Train-test overlap suggests potential data leakage considerations
- The weighted ensemble is the best overall model in the notebook

## Future Improvements

- Deep learning approaches (LSTM, BERT embeddings)
- External similarity metrics (e.g., semantic similarity, word embeddings)
- Ensemble methods combining multiple model types
- Cross-validation and more robust evaluation
- Handling class imbalance (SMOTE, class weights)

## References

- [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs)
