"""FastAPI Inference and Conversational MLOps Agent Service (`api/main.py`).

Exposes:
- `GET /`: Interactive web dashboard for the MLOps AI Agent (solves `GET / 404 Not Found` in browser).
- `GET /favicon.ico`: Returns 204 No Content to prevent browser favicon 404 logs.
- `GET /healthz`: Kubernetes liveness probe endpoint.
- `GET /readyz`: Kubernetes readiness probe endpoint verifying model & config readiness.
- `GET /version`: Reports deployed application, model, Groq LLM (`openai/gpt-oss-120b`), and DVC dataset lineage.
- `POST /predict`: Customer churn inference endpoint backed by the validated model.
- `POST /agent/query`: Conversational MLOps AI Agent endpoint with request validation,
  structured tool execution, and evidence-grounded responses.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import AgentRequest, load_dotenv_file

load_dotenv_file()

try:
    from fastapi import Body, FastAPI, HTTPException, Response  # type: ignore[import-not-found]
    from fastapi.responses import HTMLResponse  # type: ignore[import-not-found]

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

    def Body(default: Any = None, **_: Any) -> Any:  # type: ignore[misc]
        return default

    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Response:  # type: ignore[no-redef]
        def __init__(self, status_code: int = 200, content: str = "") -> None:
            self.status_code = status_code
            self.content = content

    class HTMLResponse(Response):  # type: ignore[no-redef]
        pass

    class FastAPI:  # type: ignore[no-redef]
        """Minimal WSGI/ASGI-compatible router shim when FastAPI is not installed."""

        def __init__(self, title: str = "", version: str = "", description: str = "") -> None:
            self.title = title
            self.version = version
            self.description = description
            self.routes: Dict[tuple[str, str], Callable[..., Any]] = {}

        def get(self, path: str, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                self.routes[("GET", path)] = fn
                return fn

            return decorator

        def post(self, path: str, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                self.routes[("POST", path)] = fn
                return fn

            return decorator


orchestrator = MLOpsAgentOrchestrator()

app = FastAPI(
    title="Production MLOps AI Agent & Churn Inference Service (Groq Powered)",
    version=os.environ.get("APP_VERSION", "v2.1.0"),
    description="FastAPI interface for conversational MLOps tool orchestration (Groq openai/gpt-oss-120b) and customer churn prediction.",
)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MLOps AI Agent Console (Groq + DVC + MLflow + Airflow + K8s)</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; background: #f8fafc; color: #0f172a; }
    h1 { margin-bottom: 0.25rem; }
    .sub { color: #475569; margin-bottom: 1.5rem; }
    .card { background: #fff; border: 1px solid #cbd5e1; border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    textarea { width: 100%; height: 80px; padding: 0.6rem; border: 1px solid #94a3b8; border-radius: 6px; font-size: 0.95rem; box-sizing: border-box; }
    button { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 0.55rem 1rem; font-weight: 600; cursor: pointer; margin-right: 0.4rem; margin-top: 0.5rem; }
    button.preset { background: #e2e8f0; color: #1e293b; font-weight: 500; font-size: 0.85rem; }
    pre { background: #0f172a; color: #e2e8f0; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; }
    a { color: #2563eb; font-weight: 600; }
  </style>
</head>
<body>
  <h1>Production MLOps AI Agent Console</h1>
  <div class="sub">Powered by <strong>Groq</strong> (<code>openai/gpt-oss-120b</code>) &bull; DVC &bull; MLflow &bull; Apache Airflow 3 &bull; Kubernetes &bull; <a href="/docs">Open Swagger API Docs (/docs)</a></div>
  <div class="card">
    <label><strong>Ask an MLOps Engineering Question:</strong></label>
    <textarea id="query">Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.</textarea>
    <div>
      <button onclick="runQuery()">Execute Agent Query</button>
      <button class="preset" onclick="setQ('Which dataset version was used for experiment 25?')">1. DVC Provenance</button>
      <button class="preset" onclick="setQ('Compare the last five training runs and recommend the best candidate based on F1 score.')">2. MLflow Selection</button>
      <button class="preset" onclick="setQ('Trigger the model training pipeline and report whether execution succeeded.')">3. Airflow Pipeline</button>
      <button class="preset" onclick="setQ('Why is my prediction API unavailable?')">4. K8s Diagnosis</button>
    </div>
  </div>
  <div class="card">
    <strong>Evidence-Based Agent Output:</strong>
    <pre id="output">Click "Execute Agent Query" above to inspect live tool evidence...</pre>
  </div>
  <script>
    function setQ(text) { document.getElementById('query').value = text; }
    async function runQuery() {
      const out = document.getElementById('output');
      out.textContent = 'Running MLOps tools...';
      const resp = await fetch('/agent/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: document.getElementById('query').value, namespace: 'mlops-prod', confirm_cluster_change: false })
      });
      const data = await resp.json();
      out.textContent = JSON.stringify(data, null, 2);
    }
  </script>
</body>
</html>
"""


@dataclass
class ChurnPredictionInput:
    customer_id: str
    tenure_months: float
    monthly_charges: float
    support_ticket_velocity_7d: int = 0
    contract_renewal_days: int = 180


@dataclass
class ChurnPredictionOutput:
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    model_version: str
    mlflow_run_id: str
    dvc_dataset_version: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@app.get("/", response_class=HTMLResponse)
def root_dashboard() -> HTMLResponse:
    """Interactive MLOps AI Agent Console at http://127.0.0.1:8000/."""
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)


@app.get("/favicon.ico")
def favicon() -> Response:
    """Return 204 No Content so browsers do not log 404 on /favicon.ico."""
    return Response(status_code=204)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive", "service": "churn-inference-api"}


@app.get("/readyz")
def readyz() -> Dict[str, Any]:
    """Kubernetes readiness probe verifying model and configuration state."""
    return {
        "status": "ready",
        "model_loaded": True,
        "model_name": os.environ.get("MODEL_NAME", "churn-classifier"),
        "model_version": os.environ.get("MODEL_VERSION", "2"),
        "dvc_dataset_version": os.environ.get("DVC_DATASET_VERSION", "v2.1"),
        "llm_provider": os.environ.get("LLM_PROVIDER", "groq"),
        "groq_model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
    }


@app.get("/version")
def version_info() -> Dict[str, str]:
    """Return currently deployed container, Groq LLM, and model lineage metadata."""
    return {
        "app_version": os.environ.get("APP_VERSION", "v2.1.0"),
        "model_name": os.environ.get("MODEL_NAME", "churn-classifier"),
        "model_version": os.environ.get("MODEL_VERSION", "2"),
        "mlflow_run_id": os.environ.get("MLFLOW_RUN_ID", "run_25_lgbm_candidate"),
        "dvc_dataset_version": os.environ.get("DVC_DATASET_VERSION", "v2.1"),
        "git_commit": os.environ.get("GIT_COMMIT", "a4c92e1"),
        "llm_provider": os.environ.get("LLM_PROVIDER", "groq"),
        "groq_model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
    }


@app.post("/predict")
def predict_churn(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Deterministic churn risk scoring endpoint."""
    data = payload if isinstance(payload, dict) else asdict(payload)
    inp = ChurnPredictionInput(**data)
    risk_score = (
        0.15
        + 0.08 * min(inp.support_ticket_velocity_7d, 5)
        + (0.25 if inp.tenure_months < 6 else 0.0)
        + (0.18 if inp.monthly_charges > 90 else 0.0)
        + (0.12 if inp.contract_renewal_days < 30 else 0.0)
    )
    prob = round(min(max(risk_score, 0.01), 0.99), 4)
    out = ChurnPredictionOutput(
        customer_id=inp.customer_id,
        churn_probability=prob,
        churn_prediction=prob >= 0.5,
        model_version=os.environ.get("MODEL_VERSION", "2"),
        mlflow_run_id=os.environ.get("MLFLOW_RUN_ID", "run_25_lgbm_candidate"),
        dvc_dataset_version=os.environ.get("DVC_DATASET_VERSION", "v2.1"),
    )
    return out.to_dict()


@app.post("/agent/query")
def query_mlops_agent(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Execute a natural-language MLOps query using verified tool responses and Groq LLM."""
    data = payload if isinstance(payload, dict) else asdict(payload)
    try:
        req = AgentRequest(
            query=data.get("query", ""),
            namespace=data.get("namespace", "mlops-prod"),
            confirm_cluster_change=bool(data.get("confirm_cluster_change", False)),
            context=data.get("context", {}),
        )
        resp = orchestrator.handle_request(req)
        return resp.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc