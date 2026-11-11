# Assignment 03 — Building a Production-Ready MLOps AI Agent

**Course:** MLOps & Production AI Engineering | Fall 2026  
**Project:** `assignment-03-mlops-agent`  
**Core Stack:** Python 3.11, FastAPI, DVC, MLflow, Apache Airflow 3, Kubernetes, Docker, GitHub Actions

## 1. Overview

This repository implements a production-ready **MLOps AI Agent** capable of interpreting natural-language engineering requests, orchestrating structured tools across **DVC**, **MLflow**, **Apache Airflow 3**, and **Kubernetes**, collecting verifiable operational evidence, and producing grounded technical diagnoses without fabricating metrics or executing unauthorized infrastructure mutations.

### Repository Structure

```text
assignment-03-mlops-agent/
├── agent/
│   ├── __init__.py
│   ├── main.py                  # CLI entrypoint for the MLOps AI Agent
│   ├── orchestrator.py          # Multi-tool planning, correlation, and safety guardrails
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── dvc_tools.py         # Task 1: DVC dataset provenance & reproducibility (5 tools)
│   │   ├── mlflow_tools.py      # Task 2: MLflow experiment intelligence & registry (5 tools)
│   │   ├── airflow_tools.py     # Task 3: Apache Airflow 3 REST API workflow control (5 tools)
│   │   └── kubernetes_tools.py  # Task 4: Kubernetes Pod/Service diagnostics & safe mutations (6 tools)
│   └── schemas.py               # Structured I/O models, secret redaction, and JSON audit logger
├── api/
│   ├── __init__.py
│   └── main.py                  # FastAPI service (/healthz, /readyz, /version, /predict, /agent/query)
├── dags/
│   ├── __init__.py
│   └── training_pipeline.py     # 5-stage Airflow DAG (validate -> preprocess -> train -> evaluate -> gate)
├── data/
│   ├── churn_dataset.csv        # Versioned customer churn dataset
│   ├── churn_dataset.csv.dvc    # DVC pointer tracking MD5 checksum
│   └── provenance_registry.json # Recorded DVC dataset-to-Git-commit lineage
├── training/
│   ├── __init__.py
│   └── train.py                 # Scikit-learn GradientBoosting training & quality gate stages
├── kubernetes/
│   ├── deployment.yaml          # 2-replica Deployment with liveness/readiness probes & resource limits
│   ├── service.yaml             # ClusterIP Service exposing port 80 -> 8000
│   └── configmap.yaml           # Non-sensitive environment configuration
├── tests/
│   ├── __init__.py
│   └── test_agent_and_tools.py  # 12 automated pytest/unittest reliability & safety tests
├── screenshots/                 # Exported evidence diagrams for DVC, MLflow, Airflow, and Kubernetes
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions linting & automated test workflow
├── Dockerfile                   # Container image definition for FastAPI + MLOps Agent
├── dvc.yaml                     # Reproducible DVC pipeline definition
├── pyproject.toml               # Reproducible Python package & test configuration
├── README.md                    # Setup, configuration, service launch, and usage guide
└── architecture.md              # Full Technical Report, tool permissions, and failure investigations
```

## 2. Installation & Environment Configuration

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Docker & Kubernetes (`kubectl` / Minikube / Kind) for cluster deployment

### Setup Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### Environment Variables (Never Commit Secrets)

All credentials and endpoints are injected via environment variables (or Kubernetes `Secret` / `ConfigMap` objects). Copy the template below into your shell session:

```bash
export APP_VERSION="v2.1.0"
export MODEL_NAME="churn-classifier"
export MODEL_VERSION="2"
export MLFLOW_RUN_ID="run_25_lgbm_candidate"
export DVC_DATASET_VERSION="v2.1"
export GIT_COMMIT="a4c92e1"
export MLFLOW_TRACKING_URI="http://localhost:5000"
export AIRFLOW_API_URL="http://localhost:8080/api/v2"
export AIRFLOW_API_TOKEN="replace-with-runtime-jwt"
```

## 3. Launching Services

### A. Launch the FastAPI Service Locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify health, readiness, and version endpoints:

```bash
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/readyz
curl -s http://localhost:8000/version
```

### B. Run the 5-Stage Training Pipeline & DVC Stages

```bash
python3 -c "from dags.training_pipeline import run_pipeline_synchronously; print(run_pipeline_synchronously('data/churn_dataset.csv'))"
```

### C. Deploy to Kubernetes (`mlops-prod` namespace)

```bash
kubectl create namespace mlops-prod --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl get pods,svc,deploy -n mlops-prod
```

## 4. Interacting with the MLOps AI Agent

### Via CLI (`agent/main.py`)

```bash
# Mandatory multi-tool scenario
python3 -m agent.main --query "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."

# Dataset provenance query
python3 -m agent.main --query "Which dataset version was used for experiment 25?"

# Deployment failure diagnosis
python3 -m agent.main --query "Why is my prediction API unavailable?"
```

### Via FastAPI REST Endpoint (`POST /agent/query`)

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
    "namespace": "mlops-prod",
    "confirm_cluster_change": false
  }'
```

## 5. Executing Automated Tests & Linting

Run the 12 automated tests (covering mocked tool responses, tool timeouts, invalid requests, non-fabrication when tools are unavailable, unauthorized Kubernetes mutation rejection, secret redaction in structured logs, and two injected failure investigations):

```bash
pytest tests/ -v
# Or using standard library unittest runner:
python3 -m unittest discover -s tests -v
```
