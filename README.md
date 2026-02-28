# SentiSpark — Restaurant Review Sentiment Analysis at Scale

A Big Data pipeline that classifies sentiment (positive / neutral / negative) from millions of Yelp restaurant reviews using **Apache Spark MLlib**, **Hadoop HDFS**, and a **React dashboard** powered by **Supabase**.

---

## Overview

Yelp generates millions of reviews daily — analyzing them manually is impossible. **SentiSpark** automates this process by building a distributed NLP pipeline capable of processing ~6.9 million reviews (~8.65 GB) and exposing actionable insights through an interactive web dashboard.

This project applies the full **CRISP-DM methodology** to a real-world Big Data problem.

---

## Objectives

- Automatically classify reviews as **positive / neutral / negative**
- Predict star rating (1–5) based solely on review text
- Identify keywords associated with each sentiment category
- Process 1 million reviews in under 2 hours

### Success Criteria

| Metric | Target |
|---|---|
| Accuracy | > 80% |
| F1-Score (macro) | > 0.75 |
| Inference time | < 30 min / 1M reviews |

---

## Dataset

| Attribute | Details |
|---|---|
| Name | Yelp Open Dataset |
| Source | https://www.yelp.com/dataset |
| Size | 6,990,280 reviews |
| Format | JSON (one object per line) |
| Coverage | 11 metropolitan areas (USA & Canada) |
| Main file | `yelp_academic_dataset_review.json` |

Each review object contains: `review_id`, `user_id`, `business_id`, `stars`, `date`, `text`, `useful`, `funny`, `cool`.

---

## Architecture

```
Yelp Dataset (JSON)
       │
       ▼
Hadoop HDFS (Raw Storage)
       │
       ▼
Apache Spark Pipeline
  ├── RegexTokenizer
  ├── StopWordsRemover
  ├── CountVectorizer
  ├── IDF (TF-IDF)
  └── Logistic Regression (MLlib)
       │
       ▼
Processed Results (HDFS + Parquet)
       │
       ▼
Supabase (PostgreSQL Cloud)
       │
       ▼
React Dashboard
```

### Infrastructure Components

**Hadoop HDFS**
- 1 NameNode (master) + 3 DataNodes (workers)
- Replication factor: 3
- Effective capacity: ~100 GB
- Stores: raw data, processed results, trained model

**Apache Spark**
- 1 Driver Node + 2 Worker Nodes
- Spark SQL for data transformations
- Spark MLlib for model training

**Supabase (PostgreSQL)**
- Stores only aggregated results (~50–100 MB)
- Auto-generated REST API consumed by the frontend
- Tables: `daily_stats`, `predictions`, `top_words`, `model_metrics`

**Frontend (React)**
- Interactive dashboard with sentiment distribution, temporal trends, top keywords, and model metrics
- Charting via Chart.js

---

## ML Pipeline

### Preprocessing
- Lowercase conversion
- URL and special character removal
- Minimum 10-word filter
- Duplicate removal by `text` field
- Null filtering on `text` and `stars`

### Feature Engineering
- **Tokenization**: `RegexTokenizer`
- **Stopword Removal**: Standard English list
- **Vectorization**: TF-IDF with vocabulary of 10,000 words

### Label Encoding

| Stars | Sentiment | Label |
|---|---|---|
| 1–2 | Negative | 0 |
| 3 | Neutral | 1 |
| 4–5 | Positive | 2 |

### Model: Logistic Regression (Multiclass)
- Regularization: L2 (`regParam`: 0.01, 0.1, 1.0)
- Optimizer: L-BFGS
- Max iterations: 100
- Strategy: 3-fold cross-validation + class weighting

### Data Split
| Set | Proportion | Approx. Size |
|---|---|---|
| Training | 70% | ~4.7M reviews |
| Validation | 15% | ~1M reviews |
| Test | 15% | ~1M reviews |

---

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Apache Spark 3.x
- Python 3.8+
- Node.js 18+

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/sentispark.git
cd sentispark

# 2. Start the Hadoop cluster with Docker
git clone https://github.com/big-data-europe/docker-hadoop.git
cd docker-hadoop
docker-compose up -d
cd ..

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download the Yelp dataset
# Visit https://www.yelp.com/dataset and place the file at:
# data/yelp_academic_dataset_review.json

# 5. Load data into HDFS
hdfs dfs -mkdir -p /data/yelp/raw
hdfs dfs -put data/yelp_academic_dataset_review.json /data/yelp/raw/

# 6. Run the Spark pipeline
spark-submit --master yarn src/pipeline/train.py

# 7. Sync results to Supabase
python src/sync/upload_results.py

# 8. Start the dashboard
cd frontend
npm install
npm run dev
```

---

## Project Structure

```
sentispark/
├── src/
│   ├── pipeline/
│   │   ├── preprocess.py       # Data cleaning & feature engineering
│   │   ├── train.py            # Model training with Spark MLlib
│   │   └── evaluate.py         # Metrics & confusion matrix
│   ├── sync/
│   │   └── upload_results.py   # Push results to Supabase
│   └── utils/
│       └── config.py           # Config & constants
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   └── components/             # Dashboard UI
├── notebooks/                  # EDA & exploratory analysis
├── docs/                       # Technical documentation
├── requirements.txt
└── README.md
```

---

## Expected Results

- **Accuracy**: ~82%
- **F1-Score (macro)**: > 0.75
- **Training time**: ~1 hour (distributed)
- **Full pipeline**: ~2 hours for 6.9M reviews

---

## Tech Stack

![Hadoop](https://img.shields.io/badge/Hadoop-3.x-yellow)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.x-orange)
![PySpark](https://img.shields.io/badge/PySpark-MLlib-red)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-green)
![React](https://img.shields.io/badge/React-Vite-blue)

- **Cluster**: Docker + Docker Compose (big-data-europe/docker-hadoop)
- **Storage**: Hadoop HDFS
- **Processing**: Apache Spark, Spark SQL
- **ML**: Spark MLlib (Logistic Regression, TF-IDF)
- **Language**: Python / PySpark
- **Database**: Supabase (PostgreSQL)
- **Frontend**: React (Vite), Chart.js

---

## Author

**Amir Sebastián Flores Cardona**
Software Development — Universidad Tecmilenio

---

## License

This project is for academic purposes. The Yelp dataset is subject to [Yelp's Dataset Terms of Use](https://www.yelp.com/dataset/terms).
