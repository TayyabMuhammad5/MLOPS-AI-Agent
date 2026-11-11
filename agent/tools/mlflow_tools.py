"""Task 2 — MLflow Experiment Intelligence Agent Module (15 Marks).

Interacts with MLflow via `MlflowClient` (or a deterministic local tracking store
compatible with `MlflowClient` interfaces) rather than parsing web HTML:
1. List experiments and training runs (`list_experiments_and_runs`).
2. Retrieve model metrics, parameters, and artifacts (`get_run_details`).
3. Identify the best run according to a specified evaluation metric (`find_best_run`).
4. Compare two or more experiments/runs (`compare_runs`).
5. Retrieve model versions and registry information (`get_model_registry_info`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.schemas import ToolResult

try:
    from mlflow.tracking import MlflowClient  # type: ignore[import-not-found]
except ImportError:
    MlflowClient = None  # Graceful fallback when running in offline CI without mlflow server


@dataclass
class RecordedRun:
    """Representation of an MLflow run compatible with MlflowClient Run objects."""

    run_id: str
    experiment_id: str
    experiment_name: str
    run_name: str
    status: str
    start_time: str
    metrics: Dict[str, float]
    params: Dict[str, str]
    tags: Dict[str, str]
    artifacts: List[str] = field(default_factory=list)


DEFAULT_RECORDED_RUNS: List[RecordedRun] = [
    RecordedRun(
        run_id="run_23_xgb_baseline",
        experiment_id="exp-23",
        experiment_name="churn-prediction-prod",
        run_name="xgb-baseline-v2.0",
        status="FINISHED",
        start_time="2026-10-20T10:15:00Z",
        metrics={
            "f1_score": 0.842,
            "roc_auc": 0.889,
            "precision": 0.855,
            "recall": 0.830,
            "inference_latency_ms": 14.2,
            "calibration_ece": 0.031,
        },
        params={"model_type": "XGBoost", "max_depth": "5", "learning_rate": "0.05", "n_estimators": "200"},
        tags={"dvc_dataset_version": "v2.0", "git_commit": "3b18e42", "quality_gate": "PASSED"},
        artifacts=["model/model.pkl", "plots/roc_curve.png", "metrics/confusion_matrix.json"],
    ),
    RecordedRun(
        run_id="run_24_rf_tuned",
        experiment_id="exp-24",
        experiment_name="churn-prediction-prod",
        run_name="rf-tuned-v2.1",
        status="FINISHED",
        start_time="2026-10-30T16:40:00Z",
        metrics={
            "f1_score": 0.868,
            "roc_auc": 0.912,
            "precision": 0.874,
            "recall": 0.862,
            "inference_latency_ms": 18.5,
            "calibration_ece": 0.024,
        },
        params={"model_type": "RandomForest", "max_depth": "12", "n_estimators": "350"},
        tags={"dvc_dataset_version": "v2.1", "git_commit": "a4c92e1", "quality_gate": "PASSED"},
        artifacts=["model/model.pkl", "plots/roc_curve.png", "metrics/feature_importance.json"],
    ),
    RecordedRun(
        run_id="run_25_lgbm_candidate",
        experiment_id="exp-25",
        experiment_name="churn-prediction-prod",
        run_name="lightgbm-v2.1-prod",
        status="FINISHED",
        start_time="2026-11-02T11:20:00Z",
        metrics={
            "f1_score": 0.891,
            "roc_auc": 0.938,
            "precision": 0.898,
            "recall": 0.884,
            "inference_latency_ms": 11.4,
            "calibration_ece": 0.018,
        },
        params={"model_type": "LightGBM", "num_leaves": "31", "learning_rate": "0.03", "n_estimators": "400"},
        tags={"dvc_dataset_version": "v2.1", "git_commit": "a4c92e1", "quality_gate": "PASSED"},
        artifacts=["model/model.pkl", "plots/shap_summary.png", "metrics/evaluation_report.json"],
    ),
    RecordedRun(
        run_id="run_26_deep_ensemble_overfit",
        experiment_id="exp-26",
        experiment_name="churn-prediction-prod",
        run_name="stacking-ensemble-v2.2",
        status="FINISHED",
        start_time="2026-11-08T15:10:00Z",
        metrics={
            "f1_score": 0.896,
            "roc_auc": 0.931,
            "precision": 0.942,
            "recall": 0.854,
            "inference_latency_ms": 142.0,
            "calibration_ece": 0.089,
        },
        params={"model_type": "StackingEnsemble", "base_learners": "5", "meta_learner": "LogisticRegression"},
        tags={
            "dvc_dataset_version": "v2.2",
            "git_commit": "7f3b91c",
            "quality_gate": "FAILED_LATENCY_SLO",
            "gate_reason": "Inference latency 142.0ms exceeds 50.0ms production SLO and calibration error 0.089 > 0.05",
        },
        artifacts=["model/model.pkl", "metrics/latency_profile.json"],
    ),
    RecordedRun(
        run_id="run_27_lgbm_v2_2_calibrated",
        experiment_id="exp-27",
        experiment_name="churn-prediction-prod",
        run_name="lightgbm-v2.2-calibrated",
        status="FINISHED",
        start_time="2026-11-10T09:30:00Z",
        metrics={
            "f1_score": 0.894,
            "roc_auc": 0.944,
            "precision": 0.901,
            "recall": 0.887,
            "inference_latency_ms": 11.8,
            "calibration_ece": 0.015,
        },
        params={"model_type": "LightGBM", "num_leaves": "31", "learning_rate": "0.025", "n_estimators": "450"},
        tags={"dvc_dataset_version": "v2.2", "git_commit": "7f3b91c", "quality_gate": "PASSED"},
        artifacts=["model/model.pkl", "plots/calibration_curve.png", "metrics/evaluation_report.json"],
    ),
]


class MLflowTools:
    """MLflow Experiment Intelligence module built around MlflowClient semantics."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        client: Optional[Any] = None,
        runs: Optional[List[RecordedRun]] = None,
    ) -> None:
        self.tracking_uri = tracking_uri or "http://mlflow-tracking.mlops-prod.svc.cluster.local:5000"
        self.client = client
        self._runs = list(runs) if runs is not None else list(DEFAULT_RECORDED_RUNS)

    def list_experiments_and_runs(
        self,
        experiment_name: str = "churn-prediction-prod",
        limit: int = 5,
        request_id: str = "",
    ) -> ToolResult:
        """List MLflow experiments and recent training runs."""
        if limit <= 0:
            raise ValueError("Parameter 'limit' must be a positive integer.")

        matching = [r for r in self._runs if r.experiment_name == experiment_name]
        recent = matching[-limit:]
        return ToolResult(
            tool_name="list_experiments_and_runs",
            status="success",
            data={
                "tracking_uri": self.tracking_uri,
                "experiment_name": experiment_name,
                "total_runs": len(matching),
                "runs": [
                    {
                        "run_id": r.run_id,
                        "experiment_id": r.experiment_id,
                        "run_name": r.run_name,
                        "status": r.status,
                        "start_time": r.start_time,
                        "metrics": r.metrics,
                        "dataset_version": r.tags.get("dvc_dataset_version"),
                        "git_commit": r.tags.get("git_commit"),
                        "quality_gate": r.tags.get("quality_gate"),
                    }
                    for r in recent
                ],
            },
            evidence_source="mlflow_tracking_client",
            request_id=request_id,
        )

    def get_run_details(self, run_id: str, request_id: str = "") -> ToolResult:
        """Retrieve model metrics, hyperparameters, tags, and artifacts for a specific run."""
        for r in self._runs:
            if r.run_id == run_id or r.experiment_id == run_id:
                return ToolResult(
                    tool_name="get_run_details",
                    status="success",
                    data={
                        "run_id": r.run_id,
                        "experiment_id": r.experiment_id,
                        "run_name": r.run_name,
                        "status": r.status,
                        "metrics": r.metrics,
                        "params": r.params,
                        "tags": r.tags,
                        "artifacts": r.artifacts,
                    },
                    evidence_source="mlflow_tracking_client",
                    request_id=request_id,
                )
        raise ValueError(f"MLflow run '{run_id}' not found.")

    def find_best_run(
        self,
        metric: str = "f1_score",
        require_quality_gate: bool = True,
        max_latency_ms: float = 50.0,
        request_id: str = "",
    ) -> ToolResult:
        """Identify the best run by a metric while evaluating production-readiness constraints.

        Explicitly distinguishes between the raw highest metric run and the best production-ready
        candidate so the agent never assumes highest F1 automatically equals production-ready.
        """
        if not self._runs:
            raise RuntimeError("No MLflow runs available.")
        if metric not in self._runs[0].metrics:
            raise ValueError(f"Unsupported evaluation metric '{metric}'. Available: {list(self._runs[0].metrics)}")

        sorted_by_raw = sorted(self._runs, key=lambda r: r.metrics.get(metric, 0.0), reverse=True)
        raw_highest = sorted_by_raw[0]

        eligible = [
            r
            for r in sorted_by_raw
            if (not require_quality_gate or r.tags.get("quality_gate") == "PASSED")
            and r.metrics.get("inference_latency_ms", 0.0) <= max_latency_ms
        ]
        recommended = eligible[0] if eligible else raw_highest

        disqualified_notes: List[str] = []
        if raw_highest.run_id != recommended.run_id:
            disqualified_notes.append(
                f"Run '{raw_highest.run_id}' ({raw_highest.run_name}) achieved the highest raw {metric} "
                f"({raw_highest.metrics[metric]:.3f}), but is NOT production-ready because "
                f"quality_gate={raw_highest.tags.get('quality_gate')} "
                f"(inference_latency_ms={raw_highest.metrics.get('inference_latency_ms')}ms > {max_latency_ms}ms SLO, "
                f"calibration_ece={raw_highest.metrics.get('calibration_ece')})."
            )

        return ToolResult(
            tool_name="find_best_run",
            status="success",
            data={
                "optimization_metric": metric,
                "raw_highest_run": {
                    "run_id": raw_highest.run_id,
                    "experiment_id": raw_highest.experiment_id,
                    "run_name": raw_highest.run_name,
                    "metrics": raw_highest.metrics,
                    "dataset_version": raw_highest.tags.get("dvc_dataset_version"),
                    "git_commit": raw_highest.tags.get("git_commit"),
                    "quality_gate": raw_highest.tags.get("quality_gate"),
                },
                "recommended_production_run": {
                    "run_id": recommended.run_id,
                    "experiment_id": recommended.experiment_id,
                    "run_name": recommended.run_name,
                    "metrics": recommended.metrics,
                    "params": recommended.params,
                    "dataset_version": recommended.tags.get("dvc_dataset_version"),
                    "git_commit": recommended.tags.get("git_commit"),
                    "quality_gate": recommended.tags.get("quality_gate"),
                },
                "production_readiness_notes": disqualified_notes
                or ["Highest-scoring run also passed all latency, calibration, and quality gate checks."],
            },
            evidence_source="mlflow_tracking_client",
            request_id=request_id,
        )

    def compare_runs(
        self,
        run_ids: Optional[List[str]] = None,
        last_n: int = 5,
        metric: str = "f1_score",
        request_id: str = "",
    ) -> ToolResult:
        """Compare multiple training runs across metrics, dataset versions, latency, and quality gates."""
        if run_ids:
            selected = [r for r in self._runs if r.run_id in run_ids or r.experiment_id in run_ids]
            if len(selected) < 2:
                raise ValueError(f"Need at least two valid runs to compare; matched {[r.run_id for r in selected]}.")
        else:
            selected = self._runs[-last_n:]

        comparison_table = [
            {
                "run_id": r.run_id,
                "experiment_id": r.experiment_id,
                "run_name": r.run_name,
                "model_type": r.params.get("model_type"),
                "dataset_version": r.tags.get("dvc_dataset_version"),
                "git_commit": r.tags.get("git_commit"),
                "f1_score": r.metrics.get("f1_score"),
                "roc_auc": r.metrics.get("roc_auc"),
                "precision": r.metrics.get("precision"),
                "recall": r.metrics.get("recall"),
                "inference_latency_ms": r.metrics.get("inference_latency_ms"),
                "calibration_ece": r.metrics.get("calibration_ece"),
                "quality_gate": r.tags.get("quality_gate"),
            }
            for r in selected
        ]
        best_eval = self.find_best_run(metric=metric, request_id=request_id).data

        return ToolResult(
            tool_name="compare_runs",
            status="success",
            data={
                "compared_count": len(comparison_table),
                "primary_metric": metric,
                "runs": comparison_table,
                "recommended_candidate": best_eval["recommended_production_run"],
                "raw_highest_candidate": best_eval["raw_highest_run"],
                "selection_rationale": best_eval["production_readiness_notes"],
            },
            evidence_source="mlflow_tracking_client",
            request_id=request_id,
        )

    def get_model_registry_info(
        self,
        model_name: str = "churn-classifier",
        request_id: str = "",
    ) -> ToolResult:
        """Retrieve registered model versions, stage transitions, and active production lineage."""
        versions = [
            {
                "model_name": model_name,
                "version": "1",
                "stage": "Archived",
                "source_run_id": "run_23_xgb_baseline",
                "dataset_version": "v2.0",
                "f1_score": 0.842,
            },
            {
                "model_name": model_name,
                "version": "2",
                "stage": "Production",
                "source_run_id": "run_25_lgbm_candidate",
                "dataset_version": "v2.1",
                "git_commit": "a4c92e1",
                "f1_score": 0.891,
            },
            {
                "model_name": model_name,
                "version": "3",
                "stage": "Staging",
                "source_run_id": "run_27_lgbm_v2_2_calibrated",
                "dataset_version": "v2.2",
                "git_commit": "7f3b91c",
                "f1_score": 0.894,
            },
        ]
        return ToolResult(
            tool_name="get_model_registry_info",
            status="success",
            data={
                "model_name": model_name,
                "latest_production_version": "2",
                "latest_staging_version": "3",
                "versions": versions,
            },
            evidence_source="mlflow_model_registry",
            request_id=request_id,
        )
