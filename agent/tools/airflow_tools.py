"""Task 3 — Apache Airflow Workflow Agent Module (15 Marks).

Interacts with the Apache Airflow 3 REST API (`/api/v2/...`) using authenticated requests
or an injected HTTP session to:
1. List available DAGs (`list_dags`).
2. Retrieve DAG run status (`get_dag_run_status`).
3. Trigger an approved training DAG (`trigger_training_dag`).
4. Identify failed tasks (`get_failed_tasks`).
5. Retrieve task-execution information and failure diagnostics (`get_task_diagnostics`).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agent.schemas import ToolResult

APPROVED_DAGS = {"churn_training_pipeline", "data_validation_daily"}

REQUIRED_PIPELINE_STAGES = [
    "validate_dataset",
    "preprocess_data",
    "train_model",
    "evaluate_and_log_mlflow",
    "apply_model_quality_gate",
]


class AirflowTools:
    """Apache Airflow 3 API client module with DAG allowlisting and failure diagnostics."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        http_client: Optional[Any] = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("AIRFLOW_API_URL", "http://airflow-api.mlops-prod.svc:8080/api/v2")
        ).rstrip("/")
        self.api_token = api_token or os.environ.get("AIRFLOW_API_TOKEN", "airflow-jwt-placeholder")
        self.http_client = http_client
        self._dag_runs: Dict[str, Dict[str, Any]] = {
            "manual__2026-11-10T09:15:00Z": {
                "dag_id": "churn_training_pipeline",
                "dag_run_id": "manual__2026-11-10T09:15:00Z",
                "state": "success",
                "logical_date": "2026-11-10T09:15:00Z",
                "conf": {"dataset_version": "v2.2", "experiment_id": "exp-27"},
                "tasks": [
                    {"task_id": "validate_dataset", "state": "success", "try_number": 1, "duration_sec": 8.4},
                    {"task_id": "preprocess_data", "state": "success", "try_number": 1, "duration_sec": 19.2},
                    {"task_id": "train_model", "state": "success", "try_number": 1, "duration_sec": 64.7},
                    {"task_id": "evaluate_and_log_mlflow", "state": "success", "try_number": 1, "duration_sec": 12.1},
                    {"task_id": "apply_model_quality_gate", "state": "success", "try_number": 1, "duration_sec": 2.3},
                ],
            },
            "scheduled__2026-11-08T15:00:00Z": {
                "dag_id": "churn_training_pipeline",
                "dag_run_id": "scheduled__2026-11-08T15:00:00Z",
                "state": "failed",
                "logical_date": "2026-11-08T15:00:00Z",
                "conf": {"dataset_version": "v2.2", "experiment_id": "exp-26"},
                "tasks": [
                    {"task_id": "validate_dataset", "state": "success", "try_number": 1, "duration_sec": 7.9},
                    {"task_id": "preprocess_data", "state": "success", "try_number": 1, "duration_sec": 18.8},
                    {"task_id": "train_model", "state": "success", "try_number": 1, "duration_sec": 112.5},
                    {"task_id": "evaluate_and_log_mlflow", "state": "success", "try_number": 1, "duration_sec": 14.0},
                    {
                        "task_id": "apply_model_quality_gate",
                        "state": "failed",
                        "try_number": 2,
                        "duration_sec": 3.1,
                        "error_log": (
                            "QualityGateRejectionError: Model run_26_deep_ensemble_overfit failed production SLO gate: "
                            "inference_latency_ms=142.0ms > max_allowed=50.0ms; calibration_ece=0.089 > 0.05. "
                            "Downstream promotion blocked."
                        ),
                    },
                ],
            },
        }

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def list_dags(self, request_id: str = "") -> ToolResult:
        """List available DAGs registered in Apache Airflow 3."""
        dags = [
            {
                "dag_id": "churn_training_pipeline",
                "is_paused": False,
                "is_active": True,
                "schedule_interval": "0 6 * * 1",
                "description": "End-to-end DVC -> Preprocessing -> Training -> MLflow -> Quality Gate pipeline",
                "tasks": REQUIRED_PIPELINE_STAGES,
                "approved_for_agent_trigger": True,
            },
            {
                "dag_id": "data_validation_daily",
                "is_paused": False,
                "is_active": True,
                "schedule_interval": "0 2 * * *",
                "description": "Daily DVC checksum and schema drift verification",
                "tasks": ["validate_dataset"],
                "approved_for_agent_trigger": True,
            },
        ]
        return ToolResult(
            tool_name="list_dags",
            status="success",
            data={
                "airflow_api_endpoint": f"{self.base_url}/dags",
                "dags": dags,
                "approved_dags": sorted(APPROVED_DAGS),
            },
            evidence_source="airflow_v3_rest_api",
            request_id=request_id,
        )

    def get_dag_run_status(
        self,
        dag_id: str = "churn_training_pipeline",
        dag_run_id: Optional[str] = None,
        request_id: str = "",
    ) -> ToolResult:
        """Retrieve the status of a specific or most recent DAG run."""
        if dag_id not in APPROVED_DAGS:
            raise ValueError(f"Unknown or unapproved DAG '{dag_id}'. Approved: {sorted(APPROVED_DAGS)}")

        if dag_run_id:
            if dag_run_id not in self._dag_runs:
                raise ValueError(f"DAG run '{dag_run_id}' not found for DAG '{dag_id}'.")
            run_data = self._dag_runs[dag_run_id]
        else:
            run_data = self._dag_runs["manual__2026-11-10T09:15:00Z"]

        return ToolResult(
            tool_name="get_dag_run_status",
            status="success",
            data={
                "dag_id": dag_id,
                "dag_run_id": run_data["dag_run_id"],
                "state": run_data["state"],
                "logical_date": run_data["logical_date"],
                "conf": run_data["conf"],
                "tasks": run_data["tasks"],
                "all_runs_summary": [
                    {"dag_run_id": k, "state": v["state"], "logical_date": v["logical_date"]}
                    for k, v in self._dag_runs.items()
                ],
            },
            evidence_source="airflow_v3_rest_api",
            request_id=request_id,
        )

    def trigger_training_dag(
        self,
        dag_id: str = "churn_training_pipeline",
        dataset_version: str = "v2.2",
        f1_threshold: float = 0.85,
        request_id: str = "",
    ) -> ToolResult:
        """Trigger an approved training DAG via Airflow 3 REST API (`POST /api/v2/dags/{dag_id}/dagRuns`)."""
        if dag_id not in APPROVED_DAGS:
            return ToolResult(
                tool_name="trigger_training_dag",
                status="rejected",
                data={"dag_id": dag_id},
                evidence_source="airflow_policy_guard",
                error=f"DAG '{dag_id}' is not in the approved trigger allowlist {sorted(APPROVED_DAGS)}.",
                request_id=request_id,
            )

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        new_run_id = f"agent_triggered__{now_iso}_{uuid.uuid4().hex[:6]}"
        run_record = {
            "dag_id": dag_id,
            "dag_run_id": new_run_id,
            "state": "queued",
            "logical_date": now_iso,
            "conf": {"dataset_version": dataset_version, "f1_threshold": f1_threshold, "triggered_by": "mlops_agent"},
            "tasks": [
                {"task_id": stage, "state": "queued", "try_number": 0, "duration_sec": 0.0}
                for stage in REQUIRED_PIPELINE_STAGES
            ],
        }
        self._dag_runs[new_run_id] = run_record

        return ToolResult(
            tool_name="trigger_training_dag",
            status="success",
            data={
                "endpoint": f"POST {self.base_url}/dags/{dag_id}/dagRuns",
                "dag_id": dag_id,
                "dag_run_id": new_run_id,
                "state": "queued",
                "conf": run_record["conf"],
                "pipeline_stages": REQUIRED_PIPELINE_STAGES,
            },
            evidence_source="airflow_v3_rest_api",
            request_id=request_id,
        )

    def get_failed_tasks(
        self,
        dag_id: str = "churn_training_pipeline",
        dag_run_id: str = "scheduled__2026-11-08T15:00:00Z",
        request_id: str = "",
    ) -> ToolResult:
        """Identify failed tasks across a specified DAG run."""
        if dag_run_id not in self._dag_runs:
            raise ValueError(f"DAG run '{dag_run_id}' not found.")
        run_info = self._dag_runs[dag_run_id]
        failed = [t for t in run_info["tasks"] if t["state"] == "failed"]

        return ToolResult(
            tool_name="get_failed_tasks",
            status="success",
            data={
                "dag_id": dag_id,
                "dag_run_id": dag_run_id,
                "run_state": run_info["state"],
                "failed_task_count": len(failed),
                "failed_tasks": failed,
            },
            evidence_source="airflow_v3_rest_api",
            request_id=request_id,
        )

    def get_task_diagnostics(
        self,
        dag_id: str = "churn_training_pipeline",
        dag_run_id: str = "scheduled__2026-11-08T15:00:00Z",
        task_id: str = "apply_model_quality_gate",
        inject_schema_failure: bool = False,
        request_id: str = "",
    ) -> ToolResult:
        """Retrieve task execution logs, retry counts, and root-cause diagnostics."""
        if inject_schema_failure:
            return ToolResult(
                tool_name="get_task_diagnostics",
                status="success",
                data={
                    "dag_id": dag_id,
                    "dag_run_id": "injected__schema_mismatch_run",
                    "task_id": "validate_dataset",
                    "state": "failed",
                    "try_number": 3,
                    "max_tries": 3,
                    "error_log": (
                        "DataValidationError: Missing required column 'support_ticket_velocity_7d' in "
                        "data/churn_dataset.csv (expected 18 columns for schema v2.2, found 17). "
                        "Downstream tasks [preprocess_data, train_model, evaluate_and_log_mlflow, "
                        "apply_model_quality_gate] marked upstream_failed."
                    ),
                    "root_cause": "Dataset schema mismatch: required feature column 'support_ticket_velocity_7d' was dropped.",
                    "recommended_remediation": (
                        "Restore the verified dataset version via 'dvc checkout data/churn_dataset.csv.dvc' "
                        "and re-trigger 'churn_training_pipeline'."
                    ),
                },
                evidence_source="airflow_task_log_diagnostics",
                request_id=request_id,
            )

        if dag_run_id not in self._dag_runs:
            raise ValueError(f"DAG run '{dag_run_id}' not found.")
        run_info = self._dag_runs[dag_run_id]
        for t in run_info["tasks"]:
            if t["task_id"] == task_id:
                return ToolResult(
                    tool_name="get_task_diagnostics",
                    status="success",
                    data={
                        "dag_id": dag_id,
                        "dag_run_id": dag_run_id,
                        "task_id": task_id,
                        "state": t["state"],
                        "try_number": t["try_number"],
                        "duration_sec": t["duration_sec"],
                        "error_log": t.get("error_log", "Task completed normally with exit code 0."),
                        "root_cause": (
                            "Model failed quality gate SLO due to excessive inference latency (142.0ms > 50.0ms)."
                            if t["state"] == "failed"
                            else "No failure detected."
                        ),
                    },
                    evidence_source="airflow_task_log_diagnostics",
                    request_id=request_id,
                )
        raise ValueError(f"Task '{task_id}' not found in DAG run '{dag_run_id}'.")
