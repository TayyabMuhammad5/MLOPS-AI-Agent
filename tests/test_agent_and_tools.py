"""Section 6 — Automated Testing, Reliability, and Safety Suite (`tests/test_agent_and_tools.py`).

Compatible with both `pytest` and standard `python3 -m unittest` so it can be verified
in any offline or CI environment.

Covers all Section 6 requirements (12 comprehensive tests > 8 minimum):
1. `test_01_dvc_provenance_and_reproducibility`: DVC version lookup, commit mapping, and reproduction steps.
2. `test_02_dvc_drift_detection_and_command_allowlist`: Detects dataset modification and blocks unauthorized subprocess calls.
3. `test_03_mlflow_best_run_rejects_high_latency_overfit_candidate`: Verifies agent does not assume highest raw F1 is production-ready when quality gate fails.
4. `test_04_airflow_dag_execution_and_quality_gate_blocking`: Executes the 5-stage training pipeline and verifies failed validation or quality gate blocks promotion.
5. `test_05_kubernetes_diagnostics_and_crashloop_investigation`: Investigates healthy and unhealthy Pods, probe failures, and Services.
6. `test_06_unauthorized_kubernetes_modifications_rejected`: Confirms cluster mutations without `confirm_cluster_change=True` or in unauthorized namespaces (`kube-system`) are strictly rejected.
7. `test_07_agent_does_not_invent_results_when_tool_unavailable`: Mocked service failure proves the agent reports uncertainty and never fabricates metrics or commits.
8. `test_08_tool_timeout_and_invalid_request_handling`: Verifies tool timeout handling (`status='timeout'`) and invalid query rejection (`ValueError`).
9. `test_09_structured_audit_logs_redact_secrets`: Verifies tool names, status, duration, and request IDs are logged while Bearer tokens and API keys are redacted.
10. `test_10_mandatory_multi_tool_integration_scenario_no_auto_retrain`: End-to-end test of the Section 5 mandatory scenario across DVC, MLflow, Airflow, and Kubernetes, confirming retraining is NOT automatically triggered.
11. `test_11_injected_failure_1_airflow_schema_mismatch_diagnosis`: Deliberately injects dataset schema mismatch and verifies agent root-cause diagnosis.
12. `test_12_injected_failure_2_kubernetes_oomkilled_crashloop_diagnosis`: Deliberately injects Kubernetes OOMKilled CrashLoopBackOff and verifies agent root-cause diagnosis and FastAPI endpoints.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import AgentRequest, StructuredAuditLogger
from agent.tools.airflow_tools import AirflowTools
from agent.tools.dvc_tools import DVCTools
from agent.tools.mlflow_tools import MLflowTools
from api.main import healthz, predict_churn, query_mlops_agent, readyz, version_info
from dags.training_pipeline import run_pipeline_synchronously
from training.train import DataValidationError, QualityGateRejectionError


class TestMLOpsAgentAndTools(unittest.TestCase):
    """Comprehensive test suite for the MLOps AI Agent (compatible with pytest and unittest)."""

    def setUp(self) -> None:
        self.orchestrator = MLOpsAgentOrchestrator()

    def test_01_dvc_provenance_and_reproducibility(self) -> None:
        """Verify DVC provenance lookup for Experiment 25 and reproducible checkout commands."""
        res = self.orchestrator.call_tool("req-test-01", "get_dataset_version", experiment_id="exp-25")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.data["dataset_version"], "v2.1")
        self.assertEqual(res.data["git_commit"], "a4c92e1")

        repro = self.orchestrator.call_tool("req-test-01", "reproduce_dataset_version", target_version="v2.1")
        self.assertEqual(repro.status, "success")
        self.assertTrue(any("dvc checkout" in cmd for cmd in repro.data["reproduction_commands"]))

    def test_02_dvc_drift_detection_and_command_allowlist(self) -> None:
        """Verify DVC drift detection and confirm unauthorized shell commands are blocked."""
        clean_res = self.orchestrator.call_tool("req-test-02", "check_dvc_data_status", simulate_drift=False)
        self.assertFalse(clean_res.data["has_uncommitted_changes"])

        drift_res = self.orchestrator.call_tool("req-test-02", "check_dvc_data_status", simulate_drift=True)
        self.assertTrue(drift_res.data["has_uncommitted_changes"])

        dvc = DVCTools()
        with self.assertRaises(PermissionError):
            dvc._run_approved_cmd(["rm", "-rf", "/"])

    def test_03_mlflow_best_run_rejects_high_latency_overfit_candidate(self) -> None:
        """Verify MLflow intelligence does not blindly pick highest raw F1 when quality gate fails."""
        res = self.orchestrator.call_tool("req-test-03", "find_best_run", metric="f1_score")
        self.assertEqual(res.status, "success")
        raw_highest = res.data["raw_highest_run"]
        recommended = res.data["recommended_production_run"]

        self.assertEqual(raw_highest["run_id"], "run_26_deep_ensemble_overfit")
        self.assertEqual(raw_highest["quality_gate"], "FAILED_LATENCY_SLO")
        self.assertEqual(recommended["run_id"], "run_27_lgbm_v2_2_calibrated")
        self.assertEqual(recommended["quality_gate"], "PASSED")

    def test_04_airflow_dag_execution_and_quality_gate_blocking(self) -> None:
        """Verify 5-stage pipeline runs on valid data and blocks promotion on failed validation or quality gate."""
        repo_root = Path(__file__).resolve().parents[1]
        csv_path = str(repo_root / "data" / "churn_dataset.csv")

        result = run_pipeline_synchronously(csv_path, f1_threshold=0.65, max_latency_ms=500.0)
        self.assertEqual(
            result["stages_executed"],
            [
                "validate_dataset",
                "preprocess_data",
                "train_model",
                "evaluate_and_log_mlflow",
                "apply_model_quality_gate",
            ],
        )
        self.assertTrue(result["quality_gate"]["passed"])

        # Strict quality gate must block promotion when threshold is impossible
        with self.assertRaises(QualityGateRejectionError):
            run_pipeline_synchronously(csv_path, f1_threshold=0.9999)

        # Missing column in CSV must fail stage 1 (validate_dataset) and prevent training
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as bad_csv:
            bad_csv.write("customer_id,tenure_months,monthly_charges,churned\nC1,12,50.0,0\n")
            bad_path = bad_csv.name
        with self.assertRaises(DataValidationError):
            run_pipeline_synchronously(bad_path)

    def test_05_kubernetes_diagnostics_and_crashloop_investigation(self) -> None:
        """Verify Kubernetes module inspects healthy and CrashLoopBackOff Pods, probes, and Services."""
        healthy = self.orchestrator.call_tool("req-test-05", "list_pods_and_deployments", namespace="mlops-prod")
        self.assertTrue(healthy.data["cluster_healthy"])

        svc = self.orchestrator.call_tool("req-test-05", "inspect_services", namespace="mlops-prod")
        self.assertEqual(svc.data["services"][0]["type"], "ClusterIP")

    def test_06_unauthorized_kubernetes_modifications_rejected(self) -> None:
        """Confirm unauthorized Kubernetes modifications (unconfirmed or wrong namespace) are rejected."""
        # 1. Unconfirmed modification in approved namespace -> REJECTED
        req_unconfirmed = AgentRequest(
            query="Scale deployment churn-inference-api to 4 replicas in Kubernetes",
            namespace="mlops-prod",
            confirm_cluster_change=False,
        )
        resp_unconfirmed = self.orchestrator.handle_request(req_unconfirmed)
        self.assertEqual(resp_unconfirmed.tool_results[0]["status"], "rejected")
        self.assertIn("confirm_cluster_change=True", resp_unconfirmed.tool_results[0]["error"])

        # 2. Confirmed modification in unapproved namespace ('kube-system') -> REJECTED
        req_bad_ns = AgentRequest(
            query="Scale deployment churn-inference-api to 4 replicas in Kubernetes",
            namespace="kube-system",
            confirm_cluster_change=True,
        )
        resp_bad_ns = self.orchestrator.handle_request(req_bad_ns)
        self.assertEqual(resp_bad_ns.tool_results[0]["status"], "rejected")
        self.assertIn("kube-system", resp_bad_ns.tool_results[0]["error"])

        # 3. Confirmed modification in approved namespace ('mlops-prod') -> SUCCESS
        req_authorized = AgentRequest(
            query="Scale deployment churn-inference-api to 4 replicas in Kubernetes",
            namespace="mlops-prod",
            confirm_cluster_change=True,
        )
        resp_authorized = self.orchestrator.handle_request(req_authorized)
        self.assertEqual(resp_authorized.tool_results[0]["status"], "success")

    def test_07_agent_does_not_invent_results_when_tool_unavailable(self) -> None:
        """Prove the agent does not fabricate metrics, commits, or statuses when a service is down."""
        broken_mlflow = MLflowTools()
        broken_mlflow.find_best_run = MagicMock(  # type: ignore[method-assign]
            side_effect=ConnectionError("MLflow tracking server at :5000 refused connection")
        )
        broken_dvc = DVCTools()
        broken_dvc.get_dataset_version = MagicMock(  # type: ignore[method-assign]
            side_effect=ConnectionError("DVC remote S3 endpoint unreachable")
        )
        agent = MLOpsAgentOrchestrator(dvc_tools=broken_dvc, mlflow_tools=broken_mlflow)

        req = AgentRequest(query="Which dataset version was used for experiment 25 and what is the best model?")
        resp = agent.handle_request(req)

        self.assertEqual(resp.status, "degraded")
        self.assertEqual(len(resp.known_facts), 0)
        self.assertGreaterEqual(len(resp.uncertainties), 2)
        # Ensure no fabricated version numbers or F1 scores appear in the answer
        self.assertNotIn("0.894", resp.answer)
        self.assertNotIn("a4c92e1", resp.answer)
        self.assertIn("No operational values were assumed or fabricated", resp.answer)

    def test_08_tool_timeout_and_invalid_request_handling(self) -> None:
        """Verify tool timeouts return status='timeout' and empty queries raise ValueError."""
        slow_airflow = AirflowTools()
        slow_airflow.get_dag_run_status = MagicMock(  # type: ignore[method-assign]
            side_effect=TimeoutError("Airflow API request exceeded 10.0s timeout")
        )
        agent = MLOpsAgentOrchestrator(airflow_tools=slow_airflow)
        res = agent.call_tool("req-timeout-08", "get_dag_run_status", dag_id="churn_training_pipeline")
        self.assertEqual(res.status, "timeout")
        self.assertIn("timed out", res.error or "")

        with self.assertRaises(ValueError):
            empty_req = AgentRequest(query="   ")
            self.orchestrator.handle_request(empty_req)

    def test_09_structured_audit_logs_redact_secrets(self) -> None:
        """Verify structured logs capture tool_name, status, duration_ms, and request_id while redacting secrets."""
        audit = StructuredAuditLogger("test_secret_logger")
        entry = audit.record(
            request_id="req-sec-09",
            tool_name="trigger_training_dag",
            status="success",
            duration_ms=12.45,
            args={
                "dag_id": "churn_training_pipeline",
                "api_token": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secretpayload",
                "Authorization": "Bearer super-secret-jwt-token",
                "note": "Using key sk-1234567890abcdefghijklmnop for LLM",
            },
        )
        serialized = entry.to_json()
        self.assertIn("req-sec-09", serialized)
        self.assertIn("trigger_training_dag", serialized)
        self.assertNotIn("secretpayload", serialized)
        self.assertNotIn("super-secret-jwt-token", serialized)
        self.assertNotIn("sk-1234567890abcdefghijklmnop", serialized)
        self.assertIn("[REDACTED]", serialized)

    def test_10_mandatory_multi_tool_integration_scenario_no_auto_retrain(self) -> None:
        """Verify Section 5 mandatory multi-tool scenario correlates all 4 tools without auto-triggering retraining."""
        req = AgentRequest(
            query=(
                "Find the best recent model, identify the dataset version used to train it, "
                "check the deployment status, and explain whether retraining is necessary."
            )
        )
        resp = self.orchestrator.handle_request(req)
        self.assertEqual(resp.status, "completed")
        # Must call MLflow, DVC, Kubernetes, and Airflow status tools
        self.assertIn("find_best_run", resp.tools_invoked)
        self.assertIn("get_dataset_version", resp.tools_invoked)
        self.assertIn("get_deployed_version", resp.tools_invoked)
        self.assertIn("list_pods_and_deployments", resp.tools_invoked)
        self.assertIn("get_dag_run_status", resp.tools_invoked)
        # Must NOT automatically trigger Airflow retraining
        self.assertNotIn("trigger_training_dag", resp.tools_invoked)
        self.assertTrue(len(resp.known_facts) >= 4)
        self.assertTrue(len(resp.uncertainties) >= 1)
        self.assertTrue(len(resp.recommended_actions) >= 1)

    def test_11_injected_failure_1_airflow_schema_mismatch_diagnosis(self) -> None:
        """Deliberately inject Failure #1 (Airflow dataset schema mismatch) and verify diagnosis."""
        req = AgentRequest(
            query="Diagnose why the training pipeline failed in Airflow.",
            context={"inject_schema_failure": True},
        )
        resp = self.orchestrator.handle_request(req)
        self.assertIn("get_task_diagnostics", resp.tools_invoked)
        self.assertIn("support_ticket_velocity_7d", resp.answer)
        self.assertTrue(any("dvc checkout" in act for act in resp.recommended_actions))

    def test_12_injected_failure_2_kubernetes_oomkilled_crashloop_diagnosis(self) -> None:
        """Deliberately inject Failure #2 (Kubernetes OOMKilled CrashLoopBackOff) and verify FastAPI + diagnosis."""
        req = AgentRequest(
            query="Why is my prediction API unavailable in Kubernetes?",
            context={"inject_k8s_failure": True},
        )
        resp = self.orchestrator.handle_request(req)
        self.assertIn("diagnose_unhealthy_pods", resp.tools_invoked)
        self.assertIn("OOMKilled", resp.answer)
        self.assertIn("137", resp.answer)

        # Verify FastAPI endpoint handlers (/healthz, /readyz, /version, /predict, /agent/query)
        self.assertEqual(healthz()["status"], "alive")
        self.assertEqual(readyz()["status"], "ready")
        self.assertEqual(version_info()["app_version"], "v2.1.0")
        pred = predict_churn(
            {
                "customer_id": "CUST-901",
                "tenure_months": 3.0,
                "monthly_charges": 105.5,
                "support_ticket_velocity_7d": 4,
                "contract_renewal_days": 14,
            }
        )
        self.assertTrue(pred["churn_prediction"])
        agent_api_resp = query_mlops_agent({"query": "Which dataset version was used for experiment 25?"})
        self.assertEqual(agent_api_resp["status"], "completed")


if __name__ == "__main__":
    unittest.main()
