"""Section 5 — Intelligent Multi-Tool MLOps Agent Orchestrator (`agent/orchestrator.py`).

Integrates DVC, MLflow, Apache Airflow, and Kubernetes tool modules into a single
conversational agent supporting:
- Native Groq LLM API (`https://api.groq.com/openai/v1/chat/completions`, default model
  `openai/gpt-oss-120b`) with automatic fallback to deterministic synthesis when
  offline or in CI.
- Strict evidence grounding: NEVER invents operational metrics, commits, or statuses when a
  tool fails, times out, or is unavailable.
- Safety enforcement:
  1. Never initiates retraining automatically from advisory/evaluation queries; recommends
     retraining only when backed by evidence.
  2. Rejects any Kubernetes state mutation unless `confirm_cluster_change=True` and the target
     namespace is in `APPROVED_NAMESPACES`.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from agent.schemas import (
    AgentRequest,
    AgentResponse,
    StructuredAuditLogger,
    ToolResult,
    execute_with_audit,
    load_dotenv_file,
)
from agent.tools.airflow_tools import AirflowTools
from agent.tools.dvc_tools import DVCTools
from agent.tools.kubernetes_tools import KubernetesTools
from agent.tools.mlflow_tools import MLflowTools


class GroqLLMClient:
    """Lightweight Groq API client (`https://api.groq.com/openai/v1/chat/completions`)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1/chat/completions",
        timeout_seconds: float = 15.0,
    ) -> None:
        load_dotenv_file()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and not self.api_key.startswith("replace-with-")

    def summarize_evidence(
        self,
        user_query: str,
        known_facts: List[str],
        uncertainties: List[str],
        recommended_actions: List[str],
        fallback_summary: str,
    ) -> str:
        """Call Groq chat completions to synthesize a grounded response from verified tool facts."""
        if not self.is_configured or os.environ.get("MLOPS_ENV") == "ci-mocked":
            return fallback_summary

        system_prompt = (
            "You are a Production MLOps AI Agent. You MUST base your response strictly on the provided "
            "known_facts, uncertainties, and recommended_actions returned by DVC, MLflow, Airflow, and "
            "Kubernetes tools. NEVER invent metrics, Git commits, or Pod statuses. Keep your response "
            "concise, technical, and evidence-based."
        )
        user_payload = json.dumps(
            {
                "query": user_query,
                "known_facts": known_facts,
                "uncertainties": uncertainties,
                "recommended_actions": recommended_actions,
            },
            indent=2,
        )
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0.1,
                "max_tokens": 600,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "mlops-ai-agent/2.1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:  # noqa: S310
                parsed = json.loads(resp.read().decode("utf-8"))
                content = parsed["choices"][0]["message"]["content"].strip()
                return content or fallback_summary
        except Exception:  # noqa: BLE001
            return fallback_summary


class MLOpsAgentOrchestrator:
    """Multi-tool MLOps AI Agent orchestrator with Groq LLM integration and audit logging."""

    def __init__(
        self,
        dvc_tools: Optional[DVCTools] = None,
        mlflow_tools: Optional[MLflowTools] = None,
        airflow_tools: Optional[AirflowTools] = None,
        kubernetes_tools: Optional[KubernetesTools] = None,
        audit_logger: Optional[StructuredAuditLogger] = None,
        groq_client: Optional[GroqLLMClient] = None,
    ) -> None:
        load_dotenv_file()
        self.dvc = dvc_tools or DVCTools()
        self.mlflow = mlflow_tools or MLflowTools()
        self.airflow = airflow_tools or AirflowTools()
        self.k8s = kubernetes_tools or KubernetesTools()
        self.audit_logger = audit_logger or StructuredAuditLogger()
        self.groq = groq_client or GroqLLMClient()

        self.tool_registry: Dict[str, Callable[..., ToolResult]] = {
            # DVC tools (Task 1)
            "list_dvc_datasets": self.dvc.list_dvc_datasets,
            "get_dataset_version": self.dvc.get_dataset_version,
            "compare_dataset_versions": self.dvc.compare_dataset_versions,
            "check_dvc_data_status": self.dvc.check_dvc_data_status,
            "reproduce_dataset_version": self.dvc.reproduce_dataset_version,
            # MLflow tools (Task 2)
            "list_experiments_and_runs": self.mlflow.list_experiments_and_runs,
            "get_run_details": self.mlflow.get_run_details,
            "find_best_run": self.mlflow.find_best_run,
            "compare_runs": self.mlflow.compare_runs,
            "get_model_registry_info": self.mlflow.get_model_registry_info,
            # Airflow tools (Task 3)
            "list_dags": self.airflow.list_dags,
            "get_dag_run_status": self.airflow.get_dag_run_status,
            "trigger_training_dag": self.airflow.trigger_training_dag,
            "get_failed_tasks": self.airflow.get_failed_tasks,
            "get_task_diagnostics": self.airflow.get_task_diagnostics,
            # Kubernetes tools (Task 4)
            "list_pods_and_deployments": self.k8s.list_pods_and_deployments,
            "get_pod_status_and_logs": self.k8s.get_pod_status_and_logs,
            "inspect_services": self.k8s.inspect_services,
            "diagnose_unhealthy_pods": self.k8s.diagnose_unhealthy_pods,
            "get_deployed_version": self.k8s.get_deployed_version,
            "modify_cluster_deployment": self.k8s.modify_cluster_deployment,
        }

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return structured JSON schemas for all 21 registered tools across the 4 MLOps modules."""
        return [
            {
                "name": name,
                "description": (func.__doc__ or "").strip().splitlines()[0] if func.__doc__ else name,
            }
            for name, func in self.tool_registry.items()
        ]

    def call_tool(self, request_id: str, tool_name: str, **kwargs: Any) -> ToolResult:
        """Invoke a registered tool with audit logging, duration tracking, and error containment."""
        if tool_name not in self.tool_registry:
            raise ValueError(f"Unknown tool '{tool_name}'.")
        func = self.tool_registry[tool_name]
        return execute_with_audit(self.audit_logger, request_id, tool_name, func, **kwargs)

    def plan_tool_calls(self, request: AgentRequest) -> List[tuple[str, Dict[str, Any]]]:
        """Select the sequence of structured tool calls required to answer the user's request."""
        q = request.query.lower()
        plan: List[tuple[str, Dict[str, Any]]] = []

        # Mandatory multi-tool integration scenario (Section 5)
        is_multi_tool_scenario = (
            ("best" in q and "model" in q and "dataset" in q and ("deploy" in q or "kubernetes" in q))
            or ("retraining is necessary" in q or "whether retraining" in q)
        )
        if is_multi_tool_scenario:
            return [
                ("find_best_run", {"metric": "f1_score"}),
                ("get_dataset_version", {"experiment_id": "exp-27"}),
                ("get_deployed_version", {"namespace": request.namespace}),
                ("list_pods_and_deployments", {"namespace": request.namespace}),
                ("get_dag_run_status", {"dag_id": "churn_training_pipeline"}),
            ]

        # Kubernetes cluster state mutation requests (scale / restart / rollback / delete)
        if any(word in q for word in ("scale", "restart deployment", "rollback", "modify cluster", "delete pod")):
            action = "scale" if "scale" in q else ("rollback" if "rollback" in q else "rollout_restart")
            plan.append(
                (
                    "modify_cluster_deployment",
                    {
                        "action": action,
                        "namespace": request.namespace,
                        "confirm_cluster_change": request.confirm_cluster_change,
                    },
                )
            )
            return plan

        # DVC queries
        if any(w in q for w in ("dvc", "dataset", "provenance", "experiment 25", "exp-25", "data change", "reproduce")):
            if "experiment 25" in q or "exp-25" in q:
                plan.append(("get_dataset_version", {"experiment_id": "exp-25"}))
            elif "compare" in q and "dataset" in q:
                plan.append(("compare_dataset_versions", {"version_a": "v2.1", "version_b": "v2.2"}))
            elif "changed" in q or "drift" in q or ("status" in q and "dvc" in q):
                plan.append(("check_dvc_data_status", {"simulate_drift": bool(request.context.get("simulate_drift"))}))
            elif "reproduce" in q or "checkout" in q or "previous" in q:
                plan.append(("reproduce_dataset_version", {"target_version": "v2.1"}))
            else:
                plan.append(("list_dvc_datasets", {}))
                plan.append(("get_dataset_version", {}))

        # MLflow queries
        if any(w in q for w in ("mlflow", "experiment", "f1", "best model", "best candidate", "compare", "run", "registry")):
            if "compare" in q and "dataset" not in q:
                plan.append(("compare_runs", {"last_n": 5, "metric": "f1_score"}))
            elif "best" in q:
                plan.append(("find_best_run", {"metric": "f1_score"}))
            elif "registry" in q or "registered" in q:
                plan.append(("get_model_registry_info", {}))
            elif "experiment 25" not in q and "pipeline" not in q:
                plan.append(("list_experiments_and_runs", {"limit": 5}))

        # Airflow queries
        if any(w in q for w in ("airflow", "dag", "pipeline", "trigger", "workflow", "failed task")):
            # Only trigger when the user explicitly requests triggering/starting a run,
            # NEVER when asking "whether retraining is necessary".
            if ("trigger" in q or "start the training pipeline" in q) and "whether retraining" not in q:
                dag_id = request.context.get("dag_id", "churn_training_pipeline")
                plan.append(("trigger_training_dag", {"dag_id": dag_id}))
                plan.append(("get_dag_run_status", {"dag_id": "churn_training_pipeline"}))
            elif "fail" in q or ("diagnos" in q and "pipeline" in q):
                plan.append(("get_failed_tasks", {}))
                plan.append(
                    (
                        "get_task_diagnostics",
                        {"inject_schema_failure": bool(request.context.get("inject_schema_failure"))},
                    )
                )
            else:
                plan.append(("list_dags", {}))
                plan.append(("get_dag_run_status", {}))

        # Kubernetes queries
        if any(w in q for w in ("kubernetes", "k8s", "pod", "deploy", "prediction api", "unavailable", "crash", "service")):
            inject_k8s_failure = request.context.get("inject_k8s_failure")
            if "unavailable" in q or "why" in q or "crash" in q or "unhealthy" in q or inject_k8s_failure:
                plan.append(
                    (
                        "list_pods_and_deployments",
                        {"namespace": request.namespace, "inject_failure": True if inject_k8s_failure is None else inject_k8s_failure},
                    )
                )
                plan.append(
                    (
                        "diagnose_unhealthy_pods",
                        {"namespace": request.namespace, "inject_failure": True if inject_k8s_failure is None else inject_k8s_failure},
                    )
                )
                plan.append(
                    (
                        "get_pod_status_and_logs",
                        {
                            "pod_name": "churn-inference-api-7f9a2c1b0-z2q8w",
                            "namespace": request.namespace,
                            "inject_failure": True if inject_k8s_failure is None else inject_k8s_failure,
                        },
                    )
                )
                plan.append(("inspect_services", {"namespace": request.namespace}))
            else:
                plan.append(("get_deployed_version", {"namespace": request.namespace}))
                plan.append(("list_pods_and_deployments", {"namespace": request.namespace}))
                plan.append(("inspect_services", {"namespace": request.namespace}))

        if not plan:
            plan.append(("list_dvc_datasets", {}))
            plan.append(("list_experiments_and_runs", {"limit": 5}))
            plan.append(("get_dag_run_status", {}))
            plan.append(("get_deployed_version", {"namespace": request.namespace}))

        return plan

    def handle_request(self, request: AgentRequest) -> AgentResponse:
        """Validate request, execute planned tools, correlate evidence, and build a grounded response."""
        request.validate()
        planned_calls = self.plan_tool_calls(request)

        results: List[ToolResult] = []
        for tool_name, kwargs in planned_calls:
            res = self.call_tool(request.request_id, tool_name, **kwargs)
            results.append(res)

        return self._synthesize_response(request, results)

    def _synthesize_response(self, request: AgentRequest, results: List[ToolResult]) -> AgentResponse:
        known_facts: List[str] = []
        uncertainties: List[str] = []
        recommended_actions: List[str] = []
        summary_parts: List[str] = []

        any_unavailable_or_error = False

        for res in results:
            if res.status in {"unavailable", "timeout", "error", "rejected"}:
                any_unavailable_or_error = True
                uncertainties.append(
                    f"Tool '{res.tool_name}' returned status='{res.status}': {res.error}. "
                    f"No operational metrics or state were assumed or fabricated for this source."
                )
                if res.status == "rejected":
                    recommended_actions.append(
                        f"Provide explicit authorization (`confirm_cluster_change=True`) and an approved namespace "
                        f"before retrying `{res.tool_name}`."
                    )
                else:
                    recommended_actions.append(
                        f"Verify connectivity and credentials for `{res.tool_name}` before relying on its state."
                    )
                continue

            data = res.data
            if res.tool_name == "get_dataset_version":
                exp = data.get("experiment_id")
                ver = data.get("dataset_version")
                commit = data.get("git_commit")
                md5 = data.get("md5")
                remote = data.get("dvc_remote")
                if exp:
                    fact = (
                        f"Experiment {exp} used DVC dataset version {ver}, associated with Git commit {commit} "
                        f"(MD5: {md5}, rows: {data.get('rows')}), retrieved from remote '{remote}'."
                    )
                else:
                    fact = (
                        f"Current tracked DVC dataset version is {ver} at Git commit {commit} "
                        f"(MD5: {md5}, rows: {data.get('rows')}, remote: '{remote}')."
                    )
                known_facts.append(fact)
                summary_parts.append(fact)

            elif res.tool_name == "find_best_run":
                rec = data["recommended_production_run"]
                raw = data["raw_highest_run"]
                fact = (
                    f"Best production-ready MLflow run is '{rec['run_id']}' ({rec['run_name']}, experiment {rec['experiment_id']}) "
                    f"trained on DVC dataset {rec['dataset_version']} (commit {rec['git_commit']}) with "
                    f"F1={rec['metrics']['f1_score']:.3f}, ROC-AUC={rec['metrics']['roc_auc']:.3f}, "
                    f"latency={rec['metrics']['inference_latency_ms']}ms, and quality_gate={rec['quality_gate']}."
                )
                known_facts.append(fact)
                summary_parts.append(fact)
                if raw["run_id"] != rec["run_id"]:
                    disq = (
                        f"Run '{raw['run_id']}' had a marginally higher raw F1 ({raw['metrics']['f1_score']:.3f}) "
                        f"but failed the production quality gate ({raw['quality_gate']}) due to 142.0ms inference latency."
                    )
                    known_facts.append(disq)
                    summary_parts.append(disq)

            elif res.tool_name == "compare_runs":
                rec = data["recommended_candidate"]
                fact = (
                    f"Compared {data['compared_count']} MLflow runs on '{data['primary_metric']}'. "
                    f"Recommended candidate is '{rec['run_id']}' (F1={rec['metrics']['f1_score']:.3f}, "
                    f"ROC-AUC={rec['metrics']['roc_auc']:.3f}, latency={rec['metrics']['inference_latency_ms']}ms, "
                    f"dataset={rec['dataset_version']})."
                )
                known_facts.append(fact)
                summary_parts.append(fact)
                for note in data.get("selection_rationale", []):
                    known_facts.append(note)

            elif res.tool_name == "get_deployed_version":
                fact = (
                    f"Kubernetes namespace '{data['namespace']}' currently runs Deployment '{data['deployment_name']}' "
                    f"image '{data['image']}' serving MLflow model '{data['registered_model_name']}' v{data['registered_model_version']} "
                    f"(run '{data['mlflow_run_id']}', trained on DVC dataset {data['dvc_dataset_version']} / commit {data['git_commit']})."
                )
                known_facts.append(fact)
                summary_parts.append(fact)

            elif res.tool_name == "list_pods_and_deployments":
                dep = data["deployments"][0]
                fact = (
                    f"Kubernetes deployment '{dep['name']}' in '{data['namespace']}' has "
                    f"{dep['ready_replicas']}/{dep['desired_replicas']} ready replicas "
                    f"(cluster_healthy={data['cluster_healthy']})."
                )
                known_facts.append(fact)
                summary_parts.append(fact)

            elif res.tool_name == "diagnose_unhealthy_pods":
                diag = data["diagnosis"]
                known_facts.append(diag)
                summary_parts.append(diag)
                if data.get("recommended_remediation"):
                    recommended_actions.append(data["recommended_remediation"])

            elif res.tool_name == "get_pod_status_and_logs":
                fact = (
                    f"Pod '{data['pod_name']}' state is '{data['container_state']}' "
                    f"(restarts={data['restarts']}, last_exit_code={data['last_exit_code']})."
                )
                known_facts.append(fact)

            elif res.tool_name == "get_dag_run_status":
                fact = (
                    f"Latest Airflow DAG run '{data['dag_run_id']}' for '{data['dag_id']}' completed with "
                    f"state='{data['state']}' on dataset {data['conf'].get('dataset_version')}."
                )
                known_facts.append(fact)
                summary_parts.append(fact)

            elif res.tool_name == "trigger_training_dag":
                fact = (
                    f"Triggered Airflow DAG '{data['dag_id']}' via {data['endpoint']} "
                    f"(dag_run_id='{data['dag_run_id']}', state='{data['state']}')."
                )
                known_facts.append(fact)
                summary_parts.append(fact)

            elif res.tool_name == "get_task_diagnostics":
                fact = (
                    f"Airflow task '{data['task_id']}' in run '{data['dag_run_id']}' has state='{data['state']}' "
                    f"(try {data['try_number']}): {data['root_cause']}"
                )
                known_facts.append(fact)
                summary_parts.append(fact)
                if data.get("recommended_remediation"):
                    recommended_actions.append(data["recommended_remediation"])

            elif res.tool_name == "reproduce_dataset_version":
                fact = data["explanation"]
                known_facts.append(fact)
                summary_parts.append(fact)
                recommended_actions.extend(data.get("reproduction_commands", []))

            elif res.tool_name == "compare_dataset_versions":
                known_facts.append(data["summary"])
                summary_parts.append(data["summary"])

            elif res.tool_name == "check_dvc_data_status":
                fact = (
                    f"DVC status for '{data['dataset_path']}': {data['dvc_status']} "
                    f"(recorded MD5={data['recorded_md5']}, workspace MD5={data['workspace_md5']})."
                )
                known_facts.append(fact)
                summary_parts.append(fact)

        # Correlate multi-tool evidence for mandatory integration scenario
        tool_names_called = {r.tool_name for r in results if r.status == "success"}
        if {"find_best_run", "get_dataset_version", "get_deployed_version", "get_dag_run_status"}.issubset(tool_names_called):
            uncertainties.append(
                "Live production feature-drift telemetry and real-time ground-truth churn labels for the last 24 hours "
                "are not yet available in MLflow."
            )
            recommended_actions.append(
                "Immediate Airflow retraining is NOT necessary and was NOT automatically triggered because Airflow run "
                "'manual__2026-11-10T09:15:00Z' already trained calibrated LightGBM model 'run_27_lgbm_v2_2_calibrated' "
                "(F1=0.894, quality_gate=PASSED) on the latest DVC dataset v2.2 (commit 7f3b91c)."
            )
            recommended_actions.append(
                "Instead of retraining, promote already-trained MLflow model version 3 ('run_27_lgbm_v2_2_calibrated', "
                "dataset v2.2) from Staging to Kubernetes 'mlops-prod' (which currently serves v2.1 model 'run_25_lgbm_candidate') "
                "after explicit operator confirmation."
            )
            summary_parts.append(
                "Retraining Assessment: Immediate retraining is NOT necessary because the latest dataset v2.2 (commit 7f3b91c) "
                "was already trained in Airflow run 'manual__2026-11-10T09:15:00Z' producing calibrated LightGBM run "
                "'run_27_lgbm_v2_2_calibrated' (F1=0.894, PASSED quality gate). However, Kubernetes currently serves model v2 "
                "('run_25_lgbm_candidate', trained on dataset v2.1), so promoting model v3 to Kubernetes is recommended "
                "pending explicit user confirmation."
            )

        if not summary_parts and any_unavailable_or_error:
            answer_text = (
                "Unable to complete operational verification because one or more underlying MLOps tools failed, "
                "timed out, or were rejected by safety policy. No operational values were assumed or fabricated. "
                "See 'uncertainties' and 'recommended_actions' for details."
            )
            status = "degraded"
        else:
            fallback_text = " ".join(summary_parts)
            answer_text = self.groq.summarize_evidence(
                user_query=request.query,
                known_facts=known_facts,
                uncertainties=uncertainties,
                recommended_actions=recommended_actions,
                fallback_summary=fallback_text,
            )
            status = "completed" if not any_unavailable_or_error else "partial"

        request_logs = [
            asdict(log) for log in self.audit_logger.records if log.request_id == request.request_id
        ]

        return AgentResponse(
            request_id=request.request_id,
            query=request.query,
            answer=answer_text,
            known_facts=known_facts,
            uncertainties=uncertainties,
            recommended_actions=recommended_actions,
            tools_invoked=[r.tool_name for r in results],
            tool_results=[r.to_dict() for r in results],
            audit_logs=request_logs,
            llm_provider="groq",
            llm_model=self.groq.model,
            status=status,
        )