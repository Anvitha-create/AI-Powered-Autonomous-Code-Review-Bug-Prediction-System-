# AI-Powered Autonomous Code Review & Bug Prediction System

An intelligent code review platform that combines XGBoost, CodeBERT, Graph Neural Networks (GNN), LSTM-based sequence analysis, AST parsing, and GitHub API integration to automate bug detection, vulnerability analysis, and code quality assessment.

## Features

* Automated code quality analysis
* Bug prediction using XGBoost
* Semantic code understanding with CodeBERT
* Code structure analysis using AST Parsing
* Dependency graph analysis using Graph Neural Networks (GNN)
* Sequential anomaly detection using LSTM
* GitHub Pull Request review integration
* FastAPI-based REST API
* Interactive web dashboard for code upload and analysis
* Real-time quality, complexity, and maintainability scoring

## Tech Stack

* Python
* FastAPI
* XGBoost
* PyTorch
* Hugging Face Transformers (CodeBERT)
* Graph Neural Networks (GNN)
* LSTM Networks
* AST Parsing
* GitHub API
* NumPy
* Scikit-learn

## System Architecture

Code Input → AST Parser → XGBoost Bug Predictor → CodeBERT Semantic Analyzer → GNN Structure Analyzer → LSTM Sequence Analyzer → Score Aggregation Engine → Review Report

## Performance

Evaluated on the CodeXGLUE Defect Detection Benchmark Dataset.

* Accuracy: 49.50%
* Precision: 57.58%
* Recall: 49.14%
* F1 Score: 53.02%

## API Endpoints

### Dashboard

GET /

### Analyze Code

POST /api/analyze

### Health Check

GET /api/health

## Running the Project

```bash
pip install -r requirements.txt
python api_server.py
```

Open:

http://localhost:8000



Anvitha Shetty
