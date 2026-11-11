"""Task 3 — Apache Airflow Training Pipeline DAG (`dags/training_pipeline.py`).

Implements the five required logical stages (Figure 2 of specification):
    validate_dataset -> preprocess_data -> train_model -> evaluate_and_log_mlflow -> apply_model_quality_gate

Enforces:
- Retry behavior (`retries=2`, `retry_delay=timedelta(minutes=2)`)
- Explicit task dependencies (`>>`)
- Structured task logging
- Hard quality-gate enforcement: a failed validation or failed model quality gate
  raises an exception and prevents downstream promotion.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from training.train import (
    apply_quality_gate_stage,
    evaluate_and_log_stage,
    preprocess_dataset_stage,
    train_model_stage,
    validate_dataset_stage,
)

logger = logging.getLogger("airflow.churn_training_pipeline")

DEFAULT_DAG_ARGS: Dict[str, Any] = {
    "owner": "mlops-engineering",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

STAGE_ORDER = [
    "validate_dataset",
    "preprocess_data",
    "train_model",
    "evaluate_and_log_mlflow",
    "apply_model_quality_gate",
]


def run_pipeline_synchronously(
    csv_path: str,
    f1_threshold: float = 0.80,
    max_latency_ms: float = 50.0,
) -> Dict[str, Any]:
    """Execute the 5-stage DAG workflow end-to-end with strict gate blocking."""
    logger.info("Stage 1/5: validate_dataset on %s", csv_path)
    val_report = validate_dataset_stage(csv_path)

    logger.info("Stage 2/5: preprocess_data")
    prep_data = preprocess_dataset_stage(csv_path)

    logger.info("Stage 3/5: train_model")
    model_artifact = train_model_stage(prep_data)

    logger.info("Stage 4/5: evaluate_and_log_mlflow")
    eval_metrics = evaluate_and_log_stage(model_artifact, prep_data)

    logger.info("Stage 5/5: apply_model_quality_gate")
    gate_decision = apply_quality_gate_stage(
        eval_metrics,
        f1_threshold=f1_threshold,
        max_latency_ms=max_latency_ms,
    )

    return {
        "dag_id": "churn_training_pipeline",
        "stages_executed": STAGE_ORDER,
        "validation": val_report,
        "metrics": eval_metrics,
        "quality_gate": gate_decision,
    }


# Register with Apache Airflow when installed in the runtime environment
try:
    from airflow import DAG  # type: ignore[import-not-found]
    from airflow.operators.python import PythonOperator  # type: ignore[import-not-found]

    with DAG(
        dag_id="churn_training_pipeline",
        default_args=DEFAULT_DAG_ARGS,
        description="Validation -> Preprocessing -> Training -> MLflow Evaluation -> Quality Gate",
        schedule="0 6 * * 1",
        start_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
        catchup=False,
        tags=["mlops", "churn", "dvc", "mlflow"],
    ) as dag:
        t1_validate = PythonOperator(
            task_id="validate_dataset",
            python_callable=validate_dataset_stage,
            op_kwargs={"csv_path": "data/churn_dataset.csv"},
        )
        t2_preprocess = PythonOperator(
            task_id="preprocess_data",
            python_callable=preprocess_dataset_stage,
            op_kwargs={"csv_path": "data/churn_dataset.csv"},
        )
        t3_train = PythonOperator(
            task_id="train_model",
            python_callable=train_model_stage,
        )
        t4_evaluate = PythonOperator(
            task_id="evaluate_and_log_mlflow",
            python_callable=evaluate_and_log_stage,
        )
        t5_gate = PythonOperator(
            task_id="apply_model_quality_gate",
            python_callable=apply_quality_gate_stage,
        )

        t1_validate >> t2_preprocess >> t3_train >> t4_evaluate >> t5_gate
except ImportError:
    dag = None
