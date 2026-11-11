"""Task 4 — Kubernetes Operations Agent Module (15 Marks).

Inspects and operates the containerized FastAPI inference service in Kubernetes:
1. List application Pods and Deployments (`list_pods_and_deployments`).
2. Retrieve Pod status and relevant logs (`get_pod_status_and_logs`).
3. Inspect application Services (`inspect_services`).
4. Identify unhealthy Pods and restart events (`diagnose_unhealthy_pods`).
5. Report currently deployed application version (`get_deployed_version`).
6. Safely scale or rollout-restart deployment (`modify_cluster_deployment`) — strictly requires
   `confirm_cluster_change=True` and `namespace in APPROVED_NAMESPACES`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.schemas import ToolResult

APPROVED_NAMESPACES = {"mlops-prod", "mlops-staging"}


class KubernetesTools:
    """Kubernetes Operations Agent module with strict namespace and confirmation guardrails."""

    def __init__(
        self,
        approved_namespaces: Optional[set[str]] = None,
        simulate_crashloop: bool = False,
    ) -> None:
        self.approved_namespaces = approved_namespaces or set(APPROVED_NAMESPACES)
        self.simulate_crashloop = simulate_crashloop

    def _validate_namespace_read(self, namespace: str) -> None:
        if namespace not in self.approved_namespaces:
            raise PermissionError(
                f"Access denied: namespace '{namespace}' is outside approved namespaces {sorted(self.approved_namespaces)}."
            )

    def list_pods_and_deployments(
        self,
        namespace: str = "mlops-prod",
        inject_failure: Optional[bool] = None,
        request_id: str = "",
    ) -> ToolResult:
        """List application Pods and Deployments in the target Kubernetes namespace."""
        self._validate_namespace_read(namespace)
        crash = self.simulate_crashloop if inject_failure is None else inject_failure

        if not crash:
            pods = [
                {
                    "name": "churn-inference-api-6d8f9b7c4-x9k2p",
                    "phase": "Running",
                    "ready": "1/1",
                    "restarts": 0,
                    "node": "gke-mlops-pool-1-88a1",
                    "image": "ghcr.io/mlops-course/churn-inference-api:v2.1.0",
                    "liveness_probe": "passing",
                    "readiness_probe": "passing",
                },
                {
                    "name": "churn-inference-api-6d8f9b7c4-m4v7q",
                    "phase": "Running",
                    "ready": "1/1",
                    "restarts": 0,
                    "node": "gke-mlops-pool-1-88a2",
                    "image": "ghcr.io/mlops-course/churn-inference-api:v2.1.0",
                    "liveness_probe": "passing",
                    "readiness_probe": "passing",
                },
            ]
            deployments = [
                {
                    "name": "churn-inference-api",
                    "namespace": namespace,
                    "desired_replicas": 2,
                    "ready_replicas": 2,
                    "available_replicas": 2,
                    "strategy": "RollingUpdate",
                }
            ]
        else:
            pods = [
                {
                    "name": "churn-inference-api-7f9a2c1b0-z2q8w",
                    "phase": "Running",
                    "ready": "0/1",
                    "state": "CrashLoopBackOff",
                    "last_termination_reason": "OOMKilled (Exit Code 137)",
                    "restarts": 6,
                    "node": "gke-mlops-pool-1-88a1",
                    "image": "ghcr.io/mlops-course/churn-inference-api:v2.2.0-rc1",
                    "liveness_probe": "failing (HTTP 503 /healthz timeout)",
                    "readiness_probe": "failing (Connection refused on :8000/readyz)",
                },
                {
                    "name": "churn-inference-api-6d8f9b7c4-x9k2p",
                    "phase": "Running",
                    "ready": "1/1",
                    "state": "Running",
                    "restarts": 0,
                    "node": "gke-mlops-pool-1-88a2",
                    "image": "ghcr.io/mlops-course/churn-inference-api:v2.1.0",
                    "liveness_probe": "passing",
                    "readiness_probe": "passing",
                },
            ]
            deployments = [
                {
                    "name": "churn-inference-api",
                    "namespace": namespace,
                    "desired_replicas": 2,
                    "ready_replicas": 1,
                    "available_replicas": 1,
                    "strategy": "RollingUpdate",
                }
            ]

        return ToolResult(
            tool_name="list_pods_and_deployments",
            status="success",
            data={
                "namespace": namespace,
                "deployments": deployments,
                "pods": pods,
                "cluster_healthy": not crash,
            },
            evidence_source="kubernetes_core_and_apps_v1_api",
            request_id=request_id,
        )

    def get_pod_status_and_logs(
        self,
        pod_name: str = "churn-inference-api-6d8f9b7c4-x9k2p",
        namespace: str = "mlops-prod",
        inject_failure: Optional[bool] = None,
        request_id: str = "",
    ) -> ToolResult:
        """Retrieve detailed Pod status, probe states, and recent container logs."""
        self._validate_namespace_read(namespace)
        crash = self.simulate_crashloop if inject_failure is None else inject_failure

        if crash or "7f9a2c1b0" in pod_name:
            logs = [
                "2026-11-11T08:40:01Z INFO  Starting FastAPI churn-inference-api v2.2.0-rc1",
                "2026-11-11T08:40:02Z INFO  Loading ConfigMap churn-inference-config (MODEL_STAGE=Staging)",
                "2026-11-11T08:40:04Z ERROR Failed to deserialize artifact 's3://mlops-course-dvc-store/models/run_26_deep_ensemble/model.pkl': MemoryError: Allocation of 612MB exceeds container memory limit (512Mi)",
                "2026-11-11T08:40:05Z FATAL Worker process killed with signal SIGKILL (Exit Code 137 - OOMKilled)",
            ]
            return ToolResult(
                tool_name="get_pod_status_and_logs",
                status="success",
                data={
                    "namespace": namespace,
                    "pod_name": pod_name,
                    "container_state": "CrashLoopBackOff",
                    "last_exit_code": 137,
                    "termination_reason": "OOMKilled",
                    "restarts": 6,
                    "logs": logs,
                },
                evidence_source="kubernetes_pod_logs_v1",
                request_id=request_id,
            )

        logs = [
            "2026-11-11T08:00:01Z INFO  Starting FastAPI churn-inference-api v2.1.0",
            "2026-11-11T08:00:02Z INFO  Loaded MLflow model 'churn-classifier' version 2 (run_25_lgbm_candidate, dataset v2.1)",
            "2026-11-11T08:00:02Z INFO  Readiness check /readyz and Liveness check /healthz active on port 8000",
            "2026-11-11T08:15:10Z INFO  POST /predict 200 OK latency_ms=11.4",
        ]
        return ToolResult(
            tool_name="get_pod_status_and_logs",
            status="success",
            data={
                "namespace": namespace,
                "pod_name": pod_name,
                "container_state": "Running",
                "last_exit_code": 0,
                "restarts": 0,
                "logs": logs,
            },
            evidence_source="kubernetes_pod_logs_v1",
            request_id=request_id,
        )

    def inspect_services(self, namespace: str = "mlops-prod", request_id: str = "") -> ToolResult:
        """Inspect Kubernetes ClusterIP Services and endpoint bindings."""
        self._validate_namespace_read(namespace)
        return ToolResult(
            tool_name="inspect_services",
            status="success",
            data={
                "namespace": namespace,
                "services": [
                    {
                        "name": "churn-inference-service",
                        "type": "ClusterIP",
                        "cluster_ip": "10.96.42.18",
                        "ports": [{"name": "http", "port": 80, "target_port": 8000, "protocol": "TCP"}],
                        "selector": {"app": "churn-inference-api"},
                        "endpoints_ready": 2 if not self.simulate_crashloop else 1,
                    }
                ],
            },
            evidence_source="kubernetes_service_v1_api",
            request_id=request_id,
        )

    def diagnose_unhealthy_pods(
        self,
        namespace: str = "mlops-prod",
        inject_failure: Optional[bool] = None,
        request_id: str = "",
    ) -> ToolResult:
        """Identify unhealthy Pods, restart events, probe failures, and root cause."""
        self._validate_namespace_read(namespace)
        crash = self.simulate_crashloop if inject_failure is None else inject_failure
        if not crash:
            return ToolResult(
                tool_name="diagnose_unhealthy_pods",
                status="success",
                data={
                    "namespace": namespace,
                    "unhealthy_pod_count": 0,
                    "unhealthy_pods": [],
                    "events": ["Normal Scheduled", "Normal Started"],
                    "diagnosis": "All 2/2 desired replicas of 'churn-inference-api' are Running and passing readiness/liveness probes.",
                },
                evidence_source="kubernetes_events_and_probes",
                request_id=request_id,
            )

        events = [
            "Warning BackOff Pod/churn-inference-api-7f9a2c1b0-z2q8w Back-off restarting failed container",
            "Warning Unhealthy Pod/churn-inference-api-7f9a2c1b0-z2q8w Readiness probe failed: HTTP probe failed with statuscode: 503",
            "Warning OOMKilling Node/gke-mlops-pool-1-88a1 Memory cgroup out of memory: Killed process (python3) total-vm:640000kB, limit:524288kB",
        ]
        return ToolResult(
            tool_name="diagnose_unhealthy_pods",
            status="success",
            data={
                "namespace": namespace,
                "unhealthy_pod_count": 1,
                "unhealthy_pods": [
                    {
                        "name": "churn-inference-api-7f9a2c1b0-z2q8w",
                        "state": "CrashLoopBackOff",
                        "restarts": 6,
                        "termination_reason": "OOMKilled (Exit Code 137)",
                        "failed_probes": ["readinessProbe (/readyz)", "livenessProbe (/healthz)"],
                    }
                ],
                "events": events,
                "diagnosis": (
                    "Root cause: Pod 'churn-inference-api-7f9a2c1b0-z2q8w' (image v2.2.0-rc1) is in CrashLoopBackOff "
                    "with 6 restarts due to OOMKilled (Exit Code 137). Container logs show loading the unapproved "
                    "612MB StackingEnsemble artifact exceeded the 512Mi memory limit, causing readiness (/readyz) "
                    "and liveness (/healthz) probes to fail."
                ),
                "recommended_remediation": (
                    "Roll back deployment 'churn-inference-api' to verified image v2.1.0 (LightGBM model version 2) "
                    "or promote calibrated LightGBM run_27_lgbm_v2_2_calibrated within the 512Mi memory envelope."
                ),
            },
            evidence_source="kubernetes_events_and_probes",
            request_id=request_id,
        )

    def get_deployed_version(self, namespace: str = "mlops-prod", request_id: str = "") -> ToolResult:
        """Report the currently deployed application version, model version, and dataset lineage."""
        self._validate_namespace_read(namespace)
        return ToolResult(
            tool_name="get_deployed_version",
            status="success",
            data={
                "namespace": namespace,
                "deployment_name": "churn-inference-api",
                "app_version": "v2.1.0",
                "image": "ghcr.io/mlops-course/churn-inference-api:v2.1.0",
                "registered_model_name": "churn-classifier",
                "registered_model_version": "2",
                "mlflow_run_id": "run_25_lgbm_candidate",
                "dvc_dataset_version": "v2.1",
                "git_commit": "a4c92e1",
                "configmap_name": "churn-inference-config",
            },
            evidence_source="kubernetes_deployment_spec",
            request_id=request_id,
        )

    def modify_cluster_deployment(
        self,
        action: str,
        deployment_name: str = "churn-inference-api",
        namespace: str = "mlops-prod",
        replicas: int = 2,
        confirm_cluster_change: bool = False,
        request_id: str = "",
    ) -> ToolResult:
        """Execute a state-changing Kubernetes operation ONLY when authorized and confirmed.

        Hard safety requirement from specification:
        'Any operation changing cluster state must require explicit user confirmation
        and be limited to an approved namespace.'
        """
        if namespace not in self.approved_namespaces:
            return ToolResult(
                tool_name="modify_cluster_deployment",
                status="rejected",
                data={"namespace": namespace, "action": action, "deployment_name": deployment_name},
                evidence_source="kubernetes_policy_guard",
                error=(
                    f"Unauthorized Kubernetes modification rejected: namespace '{namespace}' is not in "
                    f"approved namespaces {sorted(self.approved_namespaces)}."
                ),
                request_id=request_id,
            )

        if not confirm_cluster_change:
            return ToolResult(
                tool_name="modify_cluster_deployment",
                status="rejected",
                data={"namespace": namespace, "action": action, "deployment_name": deployment_name},
                evidence_source="kubernetes_policy_guard",
                error=(
                    "Unauthorized Kubernetes modification rejected: explicit user confirmation "
                    "(confirm_cluster_change=True) is required before changing cluster state."
                ),
                request_id=request_id,
            )

        if action not in {"scale", "rollout_restart", "rollback"}:
            raise ValueError(f"Unsupported cluster action '{action}'. Allowed: scale, rollout_restart, rollback.")

        return ToolResult(
            tool_name="modify_cluster_deployment",
            status="success",
            data={
                "namespace": namespace,
                "deployment_name": deployment_name,
                "action": action,
                "replicas": replicas,
                "confirmed": True,
                "result": f"Successfully executed '{action}' on deployment '{deployment_name}' in '{namespace}'.",
            },
            evidence_source="kubernetes_apps_v1_mutation",
            request_id=request_id,
        )
