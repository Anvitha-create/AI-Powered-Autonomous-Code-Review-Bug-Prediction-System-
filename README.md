# AI-Powered Autonomous Code Review & Bug Prediction System

An advanced AI-driven code review platform that combines Machine Learning, Deep Learning, and Static Code Analysis techniques to automatically detect bugs, identify security vulnerabilities, evaluate code quality, and generate actionable improvement suggestions. The system integrates XGBoost, CodeBERT, Graph Neural Networks (GNN), LSTM-based sequence analysis, AST parsing, and GitHub API integration to provide comprehensive code intelligence and automated review capabilities.

## Features

* Automated code quality assessment and review
* Bug prediction using XGBoost machine learning models
* Semantic code understanding using Microsoft CodeBERT
* Abstract Syntax Tree (AST) parsing and static analysis
* Dependency graph construction and Graph Neural Network analysis
* Sequential anomaly detection using LSTM networks
* GitHub Pull Request review automation
* FastAPI-powered REST API backend
* Interactive web dashboard for code upload and analysis
* Real-time quality, complexity, maintainability, and security scoring
* Security vulnerability identification and reporting
* Multi-module AI scoring and consensus-based evaluation

## Tech Stack

### Programming Languages

* Python

### Machine Learning & AI

* XGBoost
* PyTorch
* Hugging Face Transformers
* CodeBERT
* Graph Neural Networks (GNN)
* LSTM Neural Networks

### Backend & APIs

* FastAPI
* Uvicorn
* REST APIs
* GitHub API

### Data Processing & Analysis

* NumPy
* Pandas
* Scikit-learn
* AST Parsing

### Development Tools

* Git
* GitHub
* VS Code

## System Architecture

```text
Code Input
    │
    ▼
AST Parser
    │
    ├── XGBoost Bug Predictor
    ├── CodeBERT Semantic Analyzer
    ├── GNN Structure Analyzer
    └── LSTM Sequence Analyzer
    │
    ▼
Score Aggregation Engine
    │
    ▼
Review Report & Recommendations
```

## Model Evaluation

The bug prediction component was evaluated using the CodeXGLUE Defect Detection Benchmark Dataset.

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 49.50% |
| Precision | 57.58% |
| Recall    | 49.14% |
| F1 Score  | 53.02% |

These results demonstrate the system's ability to identify software defects and provide automated code quality insights across diverse code samples.

## API Endpoints

### Dashboard

```http
GET /
```

### Analyze Code

```http
POST /api/analyze
```

### Health Check

```http
GET /api/health
```

### GitHub Pull Request Review

```http
POST /api/github-review
```

## Running the Project

### Clone Repository

```bash
git clone <repository-url>
cd code_review_system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
```

Update the required environment variables.

### Start Server

```bash
python api_server.py
```

Open the application in your browser:

```text
http://localhost:8000
```

## Project Structure

```text
codebert_module/
gnn_module/
lstm_module/
xgboost_module/
utils/
tests/
models/
api_server.py
main.py
requirements.txt
README.md
```



Computer Science Engineering Student | Full-Stack Development | AI/ML | Cybersecurity
