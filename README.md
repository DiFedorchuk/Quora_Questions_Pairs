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
└── reports/                       # Analysis reports and visualizations
```

## Dependencies

See `requirements.txt` for the complete list. Key dependencies:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning algorithms and metrics
- **matplotlib/seaborn**: Data visualization
- **nltk**: Natural language processing (tokenization, stopwords)

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

## Usage

### Load Data and Engineer Features
```python
from src import load_data, engineer_features

# Load datasets
df_train, df_test = load_data('data/quora_question_pairs_train.csv.zip', 
                               'data/quora_question_pairs_test.csv.zip')

# Engineer features
df_train = engineer_features(df_train)
```

### Train a Model
```python
from src import RandomForestModel
from sklearn.model_selection import train_test_split

# Prepare features and target
X = df_train[['q1_len', 'q2_len', 'len_diff', 'common_words', 'jaccard_sim', 'word_match_share']]
y = df_train['is_duplicate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestModel(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
model.save('models/random_forest_model.pkl')
```

### Evaluate Model
```python
from src import evaluate_model, print_evaluation_report

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

metrics = evaluate_model(y_test, y_pred, y_pred_proba)
print_evaluation_report(metrics)
```

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

- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Advanced ensemble method
- **Support Vector Machine**: Kernel-based classifier

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
  - Random Forest and Gradient Boosting
  - Hyperparameter tuning
  - Final model selection and ensemble methods

## Results

Key findings from analysis:
- ~63% of question pairs are duplicates (37% non-duplicates)
- Strong correlation between lexical similarity and duplicate status
- Question frequency is a significant differentiator
- Train-test overlap suggests potential data leakage considerations

## Future Improvements

- Deep learning approaches (LSTM, BERT embeddings)
- External similarity metrics (e.g., semantic similarity, word embeddings)
- Ensemble methods combining multiple model types
- Cross-validation and more robust evaluation
- Handling class imbalance (SMOTE, class weights)

## References

- [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs)
- Dataset contains intellectual property from Quora

## License

[Specify your license here]

## Contact

For questions or feedback, please contact the project maintainer.
