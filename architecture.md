# Architecture & Technical Report — Production-Ready MLOps AI Agent

## 1. Complete System Architecture

The MLOps AI Agent follows a layered, evidence-first architecture that separates conversational request validation (`api/main.py`), tool planning and safety policy enforcement (`agent/orchestrator.py`), structured audit logging (`agent/schemas.py`), and the four external MLOps integration adapters (`agent/tools/`).

```text
+-----------------------------------------------------------------------+
|                    User / ML Engineer Request                         |
|   "Find the best recent model, identify dataset version, check K8s..."|
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|              FastAPI Interface (api/main.py)                          |
|  - Pydantic validation (/agent/query, /predict, /healthz, /readyz)    |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|         MLOps AI Agent Orchestrator (agent/orchestrator.py)           |
|  - Multi-tool intent planner & structured tool-calling registry       |
|  - Safety controls: No auto-retrain, K8s namespace & confirm guards   |
|  - StructuredAuditLogger: request_id, duration_ms, secret scrubbing   |
+--------+------------------+-------------------+------------------+----+
         |                  |                   |                  |
         v                  v                   v                  v
+----------------+ +-----------------+ +-----------------+ +------------------+
|   DVC Module   | |  MLflow Module  | | Airflow 3 Module| | Kubernetes Module|
|  dvc_tools.py  | | mlflow_tools.py | | airflow_tools.py| |kubernetes_tools.py|
| - Provenance   | | - MlflowClient  | | - /api/v2 REST  | | - Pods/Deploys   |
| - MD5 drift    | | - Run metrics   | | - Trigger guard | | - Probe/OOM logs |
| - Subprocess   | | - SLO quality   | | - Task retry &  | | - ClusterIP Svc  |
|   allowlist    | |   gate filter   | |   diagnostics   | | - Mutation guard |
+--------+-------+ +--------+--------+ +--------+--------+ +---------+--------+
         |                  |                   |                    |
         +------------------+---------+---------+--------------------+
                                      |
                                      v
+-----------------------------------------------------------------------+
|                  Evidence-Based AgentResponse                         |
|  - known_facts (grounded strictly in tool outputs)                    |
|  - uncertainties (explicitly reported when tools fail or lack data)   |
|  - recommended_actions (defensible engineering next steps)            |
+-----------------------------------------------------------------------+
```

### Architectural Principles
1. **Zero Hallucination on Tool Failure**: Every operational statement in `known_facts` is derived directly from a `ToolResult` with `status == "success"`. If a tool raises a `ConnectionError` or `TimeoutError`, `execute_with_audit` records `status="unavailable"` or `"timeout"` and populates `uncertainties` without inventing metrics, commit hashes, or Pod statuses.
2. **Defensible Model Selection**: `MLflowTools.find_best_run` evaluates both the primary metric (`f1_score`) and production SLO constraints (`quality_gate == "PASSED"` and `inference_latency_ms <= 50.0`). It explicitly explains why `run_26_deep_ensemble_overfit` (raw F1 `0.896`, latency `142.0ms`, `FAILED_LATENCY_SLO`) is disqualified in favor of `run_27_lgbm_v2_2_calibrated` (F1 `0.894`, latency `11.8ms`, `PASSED`).
3. **Defense-in-Depth Safety Controls**:
   - **DVC Subprocess Allowlist**: `DVCTools._run_approved_cmd` only permits read/checkout commands (`dvc status`, `dvc diff`, `dvc checkout`, `dvc pull`, `git rev-parse`, `git log`, `git status`) and raises `PermissionError` on any other binary or subcommand.
   - **Airflow Trigger Allowlist & No Auto-Retraining**: Only `churn_training_pipeline` and `data_validation_daily` can be triggered, and advisory queries asking *whether* retraining is necessary never invoke `trigger_training_dag`.
   - **Kubernetes Namespace & Explicit Confirmation Guard**: `modify_cluster_deployment` rejects any state change unless `namespace in {"mlops-prod", "mlops-staging"}` AND `confirm_cluster_change=True`.
   - **Secret Redaction**: `StructuredAuditLogger` scrubs `Bearer` tokens, `sk-...` keys, passwords, and token fields before emitting JSON audit logs.

## 2. Tool Definitions and Permissions Matrix

| Module | Callable Tool Name | Purpose | Permission / Safety Policy |
| :--- | :--- | :--- | :--- |
| **DVC (`dvc_tools.py`)** | `list_dvc_datasets` | List tracked `.dvc` files, `dvc.yaml`, and remote storage | Read-only |
| **DVC (`dvc_tools.py`)** | `get_dataset_version` | Retrieve dataset version, MD5, and Git commit for current state or experiment | Read-only; allowlisted `git rev-parse` |
| **DVC (`dvc_tools.py`)** | `compare_dataset_versions` | Compare row delta, feature delta, and hashes across two dataset versions | Read-only |
| **DVC (`dvc_tools.py`)** | `check_dvc_data_status` | Compute workspace MD5 against `.dvc` pointer to detect uncommitted drift | Read-only |
| **DVC (`dvc_tools.py`)** | `reproduce_dataset_version` | Generate or execute reproducible `git checkout` + `dvc checkout` sequence | Allowlisted `dvc checkout` only |
| **MLflow (`mlflow_tools.py`)** | `list_experiments_and_runs` | List recent training runs, metrics, and dataset tags via `MlflowClient` | Read-only |
| **MLflow (`mlflow_tools.py`)** | `get_run_details` | Inspect parameters, metrics, tags, and artifact paths for a run | Read-only |
| **MLflow (`mlflow_tools.py`)** | `find_best_run` | Identify best production-ready run balancing metric and SLO quality gate | Read-only |
| **MLflow (`mlflow_tools.py`)** | `compare_runs` | Compare multiple runs across F1, ROC-AUC, latency, and calibration | Read-only |
| **MLflow (`mlflow_tools.py`)** | `get_model_registry_info` | Inspect registered model versions across `Archived`, `Staging`, `Production` | Read-only |
| **Airflow (`airflow_tools.py`)** | `list_dags` | Query `/api/v2/dags` for active workflows and stage topology | Authenticated read (`Bearer` redacted) |
| **Airflow (`airflow_tools.py`)** | `get_dag_run_status` | Retrieve run state and per-task execution metrics for a DAG run | Authenticated read |
| **Airflow (`airflow_tools.py`)** | `trigger_training_dag` | Trigger a new DAG run via `POST /api/v2/dags/{dag_id}/dagRuns` | Restricted to `APPROVED_DAGS`; never auto-triggered on advisory queries |
| **Airflow (`airflow_tools.py`)** | `get_failed_tasks` | List failed task instances and try counts in a DAG run | Authenticated read |
| **Airflow (`airflow_tools.py`)** | `get_task_diagnostics` | Extract task failure logs, root cause, and remediation advice | Authenticated read |
| **Kubernetes (`kubernetes_tools.py`)** | `list_pods_and_deployments` | Inspect Pod phases, replica counts, and probe states | Restricted to `APPROVED_NAMESPACES` |
| **Kubernetes (`kubernetes_tools.py`)** | `get_pod_status_and_logs` | Fetch container exit codes, restart counts, and tail logs | Restricted to `APPROVED_NAMESPACES` |
| **Kubernetes (`kubernetes_tools.py`)** | `inspect_services` | Inspect `ClusterIP` service ports, selectors, and ready endpoints | Restricted to `APPROVED_NAMESPACES` |
| **Kubernetes (`kubernetes_tools.py`)** | `diagnose_unhealthy_pods` | Correlate Kubernetes warning events, probes, and OOM terminations | Restricted to `APPROVED_NAMESPACES` |
| **Kubernetes (`kubernetes_tools.py`)** | `get_deployed_version` | Report live container image, MLflow model version, and DVC lineage | Restricted to `APPROVED_NAMESPACES` |
| **Kubernetes (`kubernetes_tools.py`)** | `modify_cluster_deployment` | Scale, restart, or roll back a Deployment | Requires `confirm_cluster_change=True` AND `namespace in {"mlops-prod", "mlops-staging"}` |

## 3. Five Tested Agent Conversations with Actual Tool Evidence

### Scenario 1 — Dataset Provenance
- **User Prompt:** `"Which dataset version was used for experiment 25?"`
- **Tools Invoked:** `["get_dataset_version"]`
- **Recorded Tool Evidence:**
  `{"experiment_id": "exp-25", "dataset_version": "v2.1", "git_commit": "a4c92e1", "git_tag": "data-v2.1", "md5": "9c2e81b4a0f123e77c92a4b610f55d12", "rows": 11800, "features": 18, "dvc_remote": "s3://mlops-course-dvc-store/churn-datasets"}`
- **Agent Response:**
  `"Experiment exp-25 used DVC dataset version v2.1, associated with Git commit a4c92e1 (MD5: 9c2e81b4a0f123e77c92a4b610f55d12, rows: 11800), retrieved from remote 's3://mlops-course-dvc-store/churn-datasets'."`

### Scenario 2 — Model Selection
- **User Prompt:** `"Compare the last five training runs and recommend the best candidate based on F1 score."`
- **Tools Invoked:** `["compare_runs"]`
- **Recorded Tool Evidence:**
  Compares `run_23_xgb_baseline` (F1 `0.842`), `run_24_rf_tuned` (F1 `0.868`), `run_25_lgbm_candidate` (F1 `0.891`), `run_26_deep_ensemble_overfit` (F1 `0.896`, latency `142.0ms`, `FAILED_LATENCY_SLO`), and `run_27_lgbm_v2_2_calibrated` (F1 `0.894`, ROC-AUC `0.944`, latency `11.8ms`, `PASSED`).
- **Agent Response:**
  `"Compared 5 MLflow runs on 'f1_score'. Recommended candidate is 'run_27_lgbm_v2_2_calibrated' (F1=0.894, ROC-AUC=0.944, latency=11.8ms, dataset=v2.2). Run 'run_26_deep_ensemble_overfit' achieved the highest raw f1_score (0.896), but is NOT production-ready because quality_gate=FAILED_LATENCY_SLO (inference_latency_ms=142.0ms > 50.0ms SLO, calibration_ece=0.089)."`

### Scenario 3 — Pipeline Execution
- **User Prompt:** `"Trigger the model training pipeline and report whether execution succeeded."`
- **Tools Invoked:** `["trigger_training_dag", "get_dag_run_status"]`
- **Recorded Tool Evidence:**
  `POST http://airflow-api.mlops-prod.svc:8080/api/v2/dags/churn_training_pipeline/dagRuns` queues the 5-stage pipeline (`validate_dataset -> preprocess_data -> train_model -> evaluate_and_log_mlflow -> apply_model_quality_gate`), and `get_dag_run_status` reports the latest completed run `manual__2026-11-10T09:15:00Z` with `state="success"` across all 5 tasks.
- **Agent Response:**
  `"Triggered Airflow DAG 'churn_training_pipeline' via POST http://airflow-api.mlops-prod.svc:8080/api/v2/dags/churn_training_pipeline/dagRuns (state='queued'). Latest Airflow DAG run 'manual__2026-11-10T09:15:00Z' for 'churn_training_pipeline' completed with state='success' on dataset v2.2."`

### Scenario 4 — Deployment Diagnosis
- **User Prompt:** `"Why is my prediction API unavailable?"`
- **Tools Invoked:** `["list_pods_and_deployments", "diagnose_unhealthy_pods", "get_pod_status_and_logs", "inspect_services"]`
- **Recorded Tool Evidence:**
  Pod `churn-inference-api-7f9a2c1b0-z2q8w` is in `CrashLoopBackOff` (`restarts=6`, `last_exit_code=137`, `OOMKilled`), readiness probe `/readyz` and liveness probe `/healthz` are failing, and container logs record `MemoryError: Allocation of 612MB exceeds container memory limit (512Mi)`.
- **Agent Response:**
  `"Kubernetes deployment 'churn-inference-api' in 'mlops-prod' has 1/2 ready replicas (cluster_healthy=False). Root cause: Pod 'churn-inference-api-7f9a2c1b0-z2q8w' (image v2.2.0-rc1) is in CrashLoopBackOff with 6 restarts due to OOMKilled (Exit Code 137). Container logs show loading the unapproved 612MB StackingEnsemble artifact exceeded the 512Mi memory limit, causing readiness (/readyz) and liveness (/healthz) probes to fail."`

### Scenario 5 — Multi-Tool Reasoning (Mandatory Integration Scenario)
- **User Prompt:** `"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."`
- **Tools Invoked:** `["find_best_run", "get_dataset_version", "get_deployed_version", "list_pods_and_deployments", "get_dag_run_status"]`
- **Agent Response Summary:**
  - **Known Facts:**
    1. Best production-ready MLflow run is `run_27_lgbm_v2_2_calibrated` (`exp-27`), trained on DVC dataset `v2.2` (Git commit `7f3b91c`, MD5 `d8e8fca2dc0f896fd7cb4cb0031ba249`) with `F1=0.894`, `ROC-AUC=0.944`, `latency=11.8ms`, and `quality_gate=PASSED`.
    2. Kubernetes `mlops-prod` currently runs 2/2 healthy replicas of `churn-inference-api:v2.1.0` serving MLflow model `churn-classifier` v2 (`run_25_lgbm_candidate`, trained on older dataset `v2.1` / commit `a4c92e1`).
    3. Airflow DAG `churn_training_pipeline` already succeeded on dataset `v2.2` in run `manual__2026-11-10T09:15:00Z`.
  - **Uncertainties:** Real-time production feature-drift telemetry and ground-truth churn labels from the past 24 hours are not yet ingested into MLflow.
  - **Recommended Actions:** Retraining is **not** necessary (and was not auto-triggered) because `run_27_lgbm_v2_2_calibrated` is already trained and validated on the newest dataset `v2.2`. Instead, promote registered model v3 (`run_27_lgbm_v2_2_calibrated`) from Staging to Kubernetes `mlops-prod` after operator confirmation.

## 4. Two Deliberately Injected Failure Investigations

### Failure Investigation 1 — Airflow Training Pipeline Schema Drift (`DataValidationError`)
- **Fault Injection:** Dropped required feature column `support_ticket_velocity_7d` from the input CSV (`test_04` and `test_11_injected_failure_1_airflow_schema_mismatch_diagnosis`).
- **Observed Symptoms:** Airflow task `validate_dataset` failed after 3 attempts (`try_number=3`), halting the pipeline and marking downstream tasks (`preprocess_data`, `train_model`, `evaluate_and_log_mlflow`, `apply_model_quality_gate`) as `upstream_failed`.
- **Agent Diagnosis Process:** The agent invoked `get_failed_tasks` followed by `get_task_diagnostics`, extracting the exact exception trace: `DataValidationError: Missing required column 'support_ticket_velocity_7d' in data/churn_dataset.csv (expected 18 columns for schema v2.2, found 17)`.
- **Resolution:** Restored the canonical 18-column dataset via `dvc checkout data/churn_dataset.csv.dvc` (verifying MD5 `d8e8fca2dc0f896fd7cb4cb0031ba249`) and re-ran the pipeline cleanly.

### Failure Investigation 2 — Kubernetes Pod `OOMKilled` (`Exit Code 137`) & `CrashLoopBackOff`
- **Fault Injection:** Deployed candidate image `v2.2.0-rc1` attempting to load the 612MB `StackingEnsemble` artifact (`run_26_deep_ensemble_overfit`) inside a container configured with a `512Mi` memory limit (`test_12_injected_failure_2_kubernetes_oomkilled_crashloop_diagnosis`).
- **Observed Symptoms:** Deployment `churn-inference-api` degraded to `1/2` ready replicas; Pod `churn-inference-api-7f9a2c1b0-z2q8w` entered `CrashLoopBackOff` with 6 restarts and HTTP 503 readiness probe failures on `/readyz`.
- **Agent Diagnosis Process:** The agent invoked `list_pods_and_deployments`, `diagnose_unhealthy_pods`, and `get_pod_status_and_logs`, correlating the Kubernetes node event (`OOMKilling ... limit:524288kB`) with the container log (`MemoryError: Allocation of 612MB exceeds container memory limit (512Mi)`).
- **Resolution:** Blocked promotion of `run_26_deep_ensemble_overfit` at the Airflow quality gate (`apply_model_quality_gate`) and rolled back the deployment to verified LightGBM model artifacts that fit well inside the `256Mi–512Mi` memory budget.

## 5. Known Limitations and Proposed Improvements

1. **Live Telemetry & Concept Drift Monitoring:** Currently, retraining recommendations rely on DVC dataset version lineage and MLflow evaluation metrics. Integrating Prometheus/Evidently drift metrics as a dedicated tool would allow the agent to detect real-time feature distribution shifts in production traffic.
2. **Asynchronous Approval Workflow for Cluster Mutations:** Currently, `modify_cluster_deployment` requires `confirm_cluster_change=True` in the request payload. Adding signed Slack/GitHub Actions approval webhooks with short-lived cryptographic tokens would strengthen human-in-the-loop governance for multi-cluster rollouts.
3. **Streaming Log Pagination:** `get_pod_status_and_logs` and `get_task_diagnostics` return bounded log windows to keep LLM context concise; adding cursor-based log pagination and semantic log search would improve diagnosis of long-running distributed training jobs.
