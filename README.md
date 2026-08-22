# 🔍 Quora Question Pairs - Duplicate Question Detection
End-to-end NLP & Machine Learning project for identifying semantically duplicate questions using the Quora Question Pairs dataset.
The project combines classical NLP techniques, advanced feature engineering, Machine Learning, Deep Learning, Transformer-based architectures, and production deployment to predict whether two questions express the same intent.

---

# 🔗 Quick Links

- 🌐 Live API: https://deployment-quora-questions-pairs.onrender.com
- 📖 Swagger Docs: https://deployment-quora-questions-pairs.onrender.com/docs
- ❤️ Health Check: https://deployment-quora-questions-pairs.onrender.com/health
- 📊 Kaggle Dataset: https://www.kaggle.com/c/quora-question-pairs

---

# 🚀 Live Demo

## 🌐 Deployed FastAPI Application

### Interactive Swagger Documentation

👉 https://deployment-quora-questions-pairs.onrender.com/docs

### Health Check

👉 https://deployment-quora-questions-pairs.onrender.com/health

The project is fully deployed on Render and serves real-time duplicate-question predictions through a REST API built with FastAPI.

---

# ✨ Highlights

- 🚀 Built a complete NLP pipeline for duplicate question detection
- 📊 Performed exploratory data analysis (EDA) on 320K+ question pairs
- 🔧 Engineered lexical, similarity, and frequency-based features
- 🧠 Trained and compared multiple ML, Deep Learning, and Transformer-based models
- 🤖 Fine-tuned DistilBERT on 50,000 question pairs
- 🌲 Tuned XGBoost and LightGBM models
- 🏆 Achieved **F1-Score of 0.85** using a weighted ensemble approach
- 🌐 Deployed a production-ready FastAPI inference service on Render
- 🐳 Containerized the application using Docker

---

# 📖 Project Overview

The goal of this project is to determine whether two questions from Quora express the same meaning and user intent.

Duplicate Question Detection is a classic Natural Language Processing (NLP) problem that combines:

- 🔤 Text Processing
- 📈 Feature Engineering
- 🌲 Machine Learning
- 🧠 Deep Learning
- 🤖 Transformer Models
- 🎯 Ensemble Learning

The final solution combines:

- Classical NLP Techniques
- Handcrafted Features
- Gradient Boosting Models
- Transformer-Based Models
- Weighted Ensemble Learning

### Target Variable

| Value | Meaning |
|---------|---------|
| 1 | Duplicate Question |
| 0 | Non-Duplicate Question |

---

# 💼 Business Value & Real-World Applications

## 🌐 Question & Answer Platforms

- Detect duplicate questions before submission
- Reduce content redundancy
- Improve search quality
- Improve user experience

Examples:

- Quora
- Stack Overflow
- Reddit Communities
- Knowledge Sharing Platforms

---

## 🤖 Customer Support Automation

- Match tickets with previously resolved cases
- Recommend relevant solutions
- Reduce support workload
- Improve response times

Examples:

- Help Desk Systems
- SaaS Customer Support
- IT Service Management

---

## 🔍 Enterprise Knowledge Management

- Detect duplicate knowledge articles
- Improve FAQ management
- Consolidate internal documentation
- Improve retrieval quality

Examples:

- SharePoint
- Confluence
- Internal Knowledge Bases

---

## 🛒 E-Commerce Platforms

- Consolidate customer questions
- Improve FAQ systems
- Improve self-service support

---

## 🧠 AI-Powered Systems

The techniques used in this project are foundational components of:

- Semantic Search
- Retrieval-Augmented Generation (RAG)
- Chatbots
- Virtual Assistants
- Recommendation Systems
- Question Answering Systems

---

# 📊 Dataset

## Quora Question Pairs Dataset

| Metric | Value |
|----------|----------|
| Training Samples | 323,432 |
| Test Samples | 80,858 |
| Problem Type | Binary Classification |
| Target Variable | is_duplicate |
| Duplicate Rate | ~37% |
| Dataset Source | Kaggle |

Dataset:

https://www.kaggle.com/c/quora-question-pairs

---

# 🔬 Exploratory Data Analysis (EDA)

## 📌 Key Findings

### Data Overlap

- 53% of unique test questions also appear in the training set
- Potential train-test overlap should be considered during evaluation

### Text Characteristics

- Average question length: ~11 words
- Average character count: ~60 characters
- Most questions appear only once
- Some generic questions occur hundreds of times

### Semantic Characteristics

- ❓ ~95% contain question marks
- 🔠 ~50% contain capital letters
- 🔢 ~20% contain numbers

### Similarity Patterns

| Metric | Duplicate | Non-Duplicate |
|----------|----------|----------|
| Word Match Share | 0.65 | 0.35 |
| Jaccard Similarity | 0.60 | 0.25 |

These results indicate that lexical and semantic similarity are strong indicators of duplicate intent.

---

# ⚙️ Feature Engineering

Several handcrafted NLP features were created.

## 📏 Length Features

- `q1_len`
- `q2_len`
- `len_diff`

## 📝 Word-Based Features

- `common_words`
- `word_match_share`

## 🔗 Similarity Features

- `jaccard_sim`
- `word_match_share`

## 📈 Frequency Features

- `max_q_freq`
- `min_q_freq`

These features help capture structural, lexical, and behavioral relationships between question pairs.

---

# 🧠 Models Implemented

## Traditional Machine Learning

✅ Logistic Regression

✅ Bag of Words + Logistic Regression

✅ TF-IDF + Logistic Regression

✅ XGBoost

✅ LightGBM

---

## Deep Learning

✅ Siamese LSTM

✅ BERT Embeddings

✅ Fine-tuned DistilBERT

---

## Ensemble Learning

✅ Weighted Ensemble

---

# 🚀 Results

## Model Performance

| Model | Log Loss | F1 Score |
|---------|---------|---------|
| Logistic Regression | 0.42 | 0.69 |
| BoW + Logistic Regression | 0.60 | 0.61 |
| TF-IDF + Logistic Regression | 0.62 | 0.61 |
| BERT Embeddings + Logistic Regression | 0.57 | 0.55 |
| LSTM | 0.50 | 0.69 |
| XGBoost (Tuned) | 0.33 | 0.76 |
| LightGBM (Tuned) | 0.33 | 0.75 |
| Fine-tuned BERT | 0.42 | 0.81 |
| 🏆 Weighted Ensemble | **0.28** | **0.85** |

---

# 🥇 Best Model

The best-performing solution was a weighted ensemble consisting of:

- 🤖 Fine-tuned DistilBERT (40%)
- 🌲 XGBoost (30%)
- 🌲 LightGBM (30%)

### Final Performance

- ✅ F1 Score: **0.85**
- ✅ Log Loss: **0.28**

The ensemble benefited from combining deep semantic representations with powerful gradient boosting models.

---

# 🚀 FastAPI Inference API

The repository includes a production-ready FastAPI application for real-time duplicate question detection.

## Live Endpoints

### Swagger UI

https://deployment-quora-questions-pairs.onrender.com/docs

### Health Check

https://deployment-quora-questions-pairs.onrender.com/health

---

## GET /health

Returns service health status.

Example:

```json
{
  "status": "healthy"
}
```

---

## POST /predict

Predict whether two questions are semantically equivalent.

Request:

```json
{
  "question1": "How can I learn Python quickly?",
  "question2": "What is the fastest way to learn Python?"
}
```

Response:

```json
{
  "is_duplicate": true,
  "duplicate_probability": 0.812345,
  "threshold": 0.5,
  "model": "tuned-xgboost-lightgbm-weighted-ensemble"
}
```

---

# 🏗 Production Deployment

The machine learning model is deployed as a REST API using:

- FastAPI
- Uvicorn
- Docker
- Render
- XGBoost
- LightGBM
- Scikit-Learn
