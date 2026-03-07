# restaurant-review-sentiment — Restaurant Review Sentiment Analysis at Scale

A Big Data pipeline that classifies sentiment (positive / neutral / negative) from millions of Yelp restaurant reviews using **Apache Spark MLlib**, **Hadoop HDFS**, and a **React dashboard** powered by **Supabase**.

---

## Overview

Yelp generates millions of reviews daily — analyzing them manually is impossible. **SentiSpark** automates this process by building a distributed NLP pipeline capable of processing ~6.9 million reviews (~8.65 GB) and exposing actionable insights through an interactive web dashboard.

This project applies the full **CRISP-DM methodology** to a real-world Big Data problem.

---

## Objectives

- Automatically classify reviews as **positive / neutral / negative**
- Derive sentiment labels from review stars (1–2 negative, 3 neutral, 4–5 positive)
- Identify keywords associated with each sentiment category
- Process 1 million reviews in under 2 hours (end-to-end on distributed setup)

### Success Criteria

| Metric           | Target                |
| ---------------- | --------------------- |
| Accuracy         | > 80%                 |
| F1-Score (macro) | > 0.75                |
| Scoring time     | < 30 min / 1M reviews |

---

## Dataset

| Attribute | Details                              |
| --------- | ------------------------------------ |
| Name      | Yelp Open Dataset                    |
| Source    | https://www.yelp.com/dataset         |
| Size      | 6,990,280 reviews                    |
| Format    | JSON (one object per line)           |
| Coverage  | 11 metropolitan areas (USA & Canada) |
| Main file | `yelp_academic_dataset_review.json`  |

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
- Spark MLlib for model training and scoring

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
| ----- | --------- | ----- |
| 1–2   | Negative  | 0     |
| 3     | Neutral   | 1     |
| 4–5   | Positive  | 2     |

### Model: Logistic Regression (Multiclass)

- Regularization: L2 (`regParam`: 0.01, 0.1, 1.0)
- Optimizer: L-BFGS
- Max iterations: 100
- Strategy: 3-fold cross-validation + class weighting

### Data Split

| Set        | Proportion | Approx. Size  |
| ---------- | ---------- | ------------- |
| Training   | 70%        | ~4.7M reviews |
| Validation | 15%        | ~1M reviews   |
| Test       | 15%        | ~1M reviews   |

---

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.11 (recommended for PySpark 3.5.x)
- Node.js 18+

> For local MVP development, Spark can run in local mode (`local[*]`).
> For distributed execution (`--master yarn`), Spark must be installed in the same cluster environment as Hadoop/YARN.

> PySpark 3.5.x is not reliable on Python 3.13 due `distutils` removal. Use Python 3.11 for local runs.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/amirmx2905/restaurant-review-sentiment.git
cd restaurant-review-sentiment

# 2. Start Hadoop with Docker (HDFS + YARN)
git clone https://github.com/big-data-europe/docker-hadoop.git
cd docker-hadoop
docker-compose up -d
cd ..

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download the Yelp dataset
# Visit https://www.yelp.com/dataset and place the file at:
# data/yelp_academic_dataset_review.json

# 5. Create HDFS folder and upload data
# Run these commands inside the NameNode container (replace <namenode-container>)
docker exec -it <namenode-container> hdfs dfs -mkdir -p /data/yelp/raw
docker exec -it <namenode-container> hdfs dfs -put -f /path/in/container/yelp_academic_dataset_review.json /data/yelp/raw/

# 6A. Run Spark pipeline in local mode (MVP / single machine)
python -m src.pipeline.train --master local[*]

# 6B. Run Spark pipeline on YARN (distributed)
# Requires Spark to be installed and configured with the same Hadoop/YARN cluster
spark-submit --master yarn src/pipeline/train.py

# 7. Sync results to Supabase
python src/sync/upload_results.py

# 8. Start the dashboard
cd frontend
npm install
npm run dev
```

### Core Pipeline Validation (Before Supabase)

Run this checklist first to ensure the core path is complete:

1. Ingest Yelp reviews JSON into raw parquet
2. Train model and generate metrics
3. Score data and produce processed outputs

```bash
# 0. Optional: create Python 3.11 virtual environment
/opt/homebrew/bin/python3.11 -m venv .venv311
.venv311/bin/python -m pip install -r requirements.txt

# 1. Quick smoke test (50k sample)
head -n 50000 yepl_dataset/yelp_academic_dataset_review.json > yepl_dataset/review_sample_50000.json
.venv311/bin/python -m src.data.ingest --config config.quick.yaml
.venv311/bin/python -m src.jobs.train --config config.quick.yaml
.venv311/bin/python -m src.jobs.score --config config.quick.yaml

# 2. Full local run (entire review file)
.venv311/bin/python -m src.data.ingest --config config.yaml
.venv311/bin/python -m src.jobs.train --config config.yaml
.venv311/bin/python -m src.jobs.score --config config.yaml
```

Expected outputs (local paths from config):

- Raw parquet: `artifacts/raw_reviews`
- Curated parquet: `artifacts/curated_reviews`
- Model: `artifacts/model`
- Metrics JSON: `artifacts/model_metrics`
- Predictions: `artifacts/predictions`
- Aggregates: `artifacts/daily_stats`, `artifacts/class_distribution`

---

## Planned Project Structure

The structure below represents the intended project layout.

```
restaurant-review-sentiment/
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
- **Training time**: ~1 hour (distributed, hardware-dependent)
- **Scoring**: < 30 min per 1M reviews
- **Full pipeline**: target < 2 hours per 1M reviews in distributed mode

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
- **Processing**: Apache Spark, Spark SQL (local mode or YARN mode)
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
