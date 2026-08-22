# 🔍 Quora Question Pairs - Duplicate Question Detection

> End-to-end NLP & Machine Learning project for identifying semantically duplicate questions using the Quora Question Pairs dataset.

## ✨ Highlights

- 🚀 Built a complete NLP pipeline for duplicate question detection
- 📊 Performed exploratory data analysis (EDA) on 320K+ question pairs
- 🔧 Engineered lexical, similarity, and frequency-based features
- 🧠 Trained and compared multiple ML, Deep Learning, and Transformer-based models
- 🤖 Fine-tuned DistilBERT on 50,000 question pairs
- 🌲 Tuned XGBoost and LightGBM models
- 🏆 Achieved **F1-score of 0.85** using a weighted ensemble approach

---

## 📖 Project Overview

The goal of this project is to determine whether two questions from Quora express the same intent and meaning.

Duplicate question detection is a classic Natural Language Processing (NLP) problem that combines text understanding, semantic similarity analysis, feature engineering, and machine learning.

The solution combines:

- 🔤 Classical NLP techniques
- 📈 Feature Engineering
- 🌲 Tree-based Machine Learning Models
- 🧠 Deep Learning Architectures
- 🤖 Transformer-based Models (BERT)
- 🎯 Ensemble Learning

The target variable is:

- `1` → Duplicate Question
- `0` → Non-Duplicate Question

---

## 💼 Business Value & Real-World Applications

Duplicate question detection has practical applications across many industries and products.

### 🌐 Question & Answer Platforms

- Detect duplicate questions before submission
- Reduce content redundancy
- Improve search quality
- Improve user experience by directing users to existing answers

Examples:

- Quora
- Stack Overflow
- Reddit Communities
- Knowledge Sharing Platforms

### 🤖 Customer Support Automation

- Match incoming tickets with previously resolved cases
- Suggest relevant solutions automatically
- Reduce support workload
- Improve response times

Examples:

- IT Service Desk
- SaaS Customer Support
- Help Centers

### 🔍 Enterprise Knowledge Management

- Identify duplicate FAQs and documentation
- Improve internal knowledge bases
- Enhance document retrieval systems
- Consolidate similar knowledge articles

Examples:

- SharePoint
- Confluence
- Corporate Knowledge Portals

### 🛒 E-commerce Platforms

- Detect duplicate product-related questions
- Consolidate FAQs
- Improve customer self-service experience

### 🧠 AI-Powered Systems

The techniques used in this project are foundational components of:

- Semantic Search
- Retrieval-Augmented Generation (RAG)
- Chatbots
- Virtual Assistants
- Recommendation Systems
- Question Answering Systems

---

## 📊 Dataset

### Quora Question Pairs Dataset

| Metric | Value |
|----------|----------|
| Training Samples | 323,432 |
| Test Samples | 80,858 |
| Problem Type | Binary Classification |
| Target Variable | `is_duplicate` |
| Duplicate Rate | ~37% |
| Dataset Source | Kaggle |

🔗 https://www.kaggle.com/c/quora-question-pairs

---

## 🔬 Exploratory Data Analysis

### 📌 Key Findings

#### Data Overlap

- 53% of unique test questions also appear in the training set
- Potential train-test overlap should be considered during model evaluation

#### Text Characteristics

- Average question length: ~11 words
- Average character count: ~60 characters
- Most questions appear only once
- Some generic questions occur hundreds of times

#### Semantic Characteristics

- ❓ ~95% of questions contain question marks
- 🔠 ~50% contain capital letters
- 🔢 ~20% contain numbers

#### Similarity Patterns

Duplicate questions exhibit significantly higher similarity scores.

| Metric | Duplicate | Non-Duplicate |
|----------|----------|----------|
| Word Match Share | ~0.65 | ~0.35 |
| Jaccard Similarity | ~0.60 | ~0.25 |

This confirms that lexical and semantic similarity are strong indicators of duplicate intent.

---

## ⚙️ Feature Engineering

Several handcrafted NLP features were created to improve model performance.

### 📏 Length Features

- `q1_len`
- `q2_len`
- `len_diff`

### 📝 Word-Based Features

- `common_words`
- `word_match_share`

### 🔗 Similarity Features

- `jaccard_sim`
- `word_match_share`

### 📈 Frequency Features

- `max_q_freq`
- `min_q_freq`

These features capture structural, lexical, and behavioral patterns between question pairs.

---

## 🧠 Models Implemented

### Traditional Machine Learning

✅ Logistic Regression

✅ Bag of Words + Logistic Regression

✅ TF-IDF + Logistic Regression

✅ XGBoost

✅ LightGBM

### Deep Learning

✅ Siamese LSTM

✅ BERT Embeddings

✅ Fine-tuned DistilBERT

### Ensemble Learning

✅ Weighted Ensemble

---

## 🚀 Results

### Model Performance

| Model | Log Loss | F1 Score |
|---------|---------|---------|
| Logistic Regression | 0.42 | 0.69 |
| Bag of Words + Logistic Regression | 0.60 | 0.61 |
| TF-IDF + Logistic Regression | 0.62 | 0.61 |
| BERT Embeddings + Logistic Regression | 0.57 | 0.55 |
| LSTM | 0.50 | 0.69 |
| XGBoost (Tuned) | 0.33 | 0.76 |
| LightGBM (Tuned) | 0.33 | 0.75 |
| Fine-tuned BERT | 0.42 | 0.81 |
| 🏆 Weighted Ensemble | **0.28** | **0.85** |

### 🥇 Best Model

The best-performing solution was a weighted ensemble combining:

- 🤖 Fine-tuned BERT (40%)
- 🌲 XGBoost (30%)
- 🌲 LightGBM (30%)

**Final Performance:**

- F1 Score: **0.85**
- Log Loss: **0.28**

This ensemble benefited from combining deep semantic representations with powerful gradient boosting models.

---

## 📈 Skills Demonstrated

This project demonstrates practical experience in:

- Natural Language Processing (NLP)
- Text Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning
- Deep Learning
- Transformer Models (BERT)
- Ensemble Learning
- Model Evaluation
- Python Data Science Ecosystem

---

## 📂 Project Structure

```text
quora-question-pairs/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── quora_question_pairs_train.csv.zip
│   └── quora_question_pairs_test.csv.zip
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_models.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
│
└── models/
```

---

## FastAPI Inference API

This repository now includes a deployable FastAPI app at [`app/main.py`](/Users/dima/Desktop/depl/Quora_Questions_Pairs/app/main.py).

### Endpoints

- `GET /health` - health check
- `POST /predict` - duplicate-question prediction

Request example:

```json
{
  "question1": "How can I learn Python quickly?",
  "question2": "What is the fastest way to learn Python?"
}
```

Response example:

```json
{
  "is_duplicate": true,
  "duplicate_probability": 0.812345,
  "threshold": 0.5,
  "model": "tuned-xgboost-lightgbm-weighted-ensemble"
}
```

---

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Test:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"question1":"How to lose weight fast?","question2":"What are quick ways to lose weight?"}'
```

---

## Deploy with Docker

The project includes [`Dockerfile`](/Users/dima/Desktop/depl/Quora_Questions_Pairs/Dockerfile) and [`.dockerignore`](/Users/dima/Desktop/depl/Quora_Questions_Pairs/.dockerignore).

Build and run:

```bash
docker build -t quora-fastapi .
docker run -p 8000:8000 quora-fastapi
```

The Docker image installs `libgomp1`, which is required for the saved XGBoost and LightGBM models to load on Debian-based images.

Then open:

- `http://localhost:8000/docs` (Swagger UI)
- `http://localhost:8000/health`

---

## Deploy to Render/Railway/Heroku-style platforms

Use the existing [`Procfile`](/Users/dima/Desktop/depl/Quora_Questions_Pairs/Procfile):

```text
web: uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```
