"""Core schemas, structured logging, and tool models for the MLOps AI Agent (`agent/schemas.py`)."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

SECRET_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    re.compile(r"(api[_-]?key|token|secret|password|passwd|kubeconfig)\s*[:=]\s*['\"]?([^\s,'\"\}]+)", re.IGNORECASE),
    re.compile(r"((?:sk-|gsk_)[A-Za-z0-9]{16,})"),
]


def load_dotenv_file(dotenv_path: Optional[Path] = None) -> None:
    """Load key=value pairs from `.env` in the repository root without external dependencies."""
    path = dotenv_path or (Path(__file__).resolve().parents[1] / ".env")
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def sanitize_value(value: Any) -> Any:
    """Recursively scrub secrets, tokens, and API keys (including Groq gsk_ keys) from logs."""
    if isinstance(value, str):
        cleaned = value
        cleaned = SECRET_PATTERNS[0].sub(r"\1[REDACTED]", cleaned)
        cleaned = SECRET_PATTERNS[1].sub(r"\1=[REDACTED]", cleaned)
        cleaned = SECRET_PATTERNS[2].sub("[REDACTED_API_KEY]", cleaned)
        return cleaned
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for k, v in value.items():
            if any(sec in k.lower() for sec in ("secret", "token", "password", "api_key", "credential", "authorization")):
                result[k] = "[REDACTED]"
            else:
                result[k] = sanitize_value(v)
        return result
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


@dataclass
class ToolInvocationLog:
    """Structured audit record for every tool invocation."""

    request_id: str
    tool_name: str
    status: str  # "success" | "error" | "timeout" | "rejected" | "unavailable"
    duration_ms: float
    timestamp: str
    sanitized_args: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class ToolResult:
    """Standardized structured output returned by every MLOps tool."""

    tool_name: str
    status: str  # "success" | "error" | "timeout" | "rejected" | "unavailable"
    data: Dict[str, Any] = field(default_factory=dict)
    evidence_source: str = ""
    error: Optional[str] = None
    duration_ms: float = 0.0
    request_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRequest:
    """Validated user request payload for the MLOps AI Agent."""

    query: str
    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex[:10]}")
    namespace: str = "mlops-prod"
    confirm_cluster_change: bool = False
    context: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("Invalid request: 'query' must be a non-empty string.")
        if len(self.query) > 4000:
            raise ValueError("Invalid request: 'query' exceeds maximum length of 4000 characters.")


@dataclass
class AgentResponse:
    """Structured evidence-based response produced by the MLOps AI Agent."""

    request_id: str
    query: str
    answer: str
    known_facts: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    tools_invoked: List[str] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    audit_logs: List[Dict[str, Any]] = field(default_factory=list)
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    status: str = "completed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StructuredAuditLogger:
    """Structured JSON logger that records tool executions without exposing secrets."""

    def __init__(self, logger_name: str = "mlops_agent.audit") -> None:
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.records: List[ToolInvocationLog] = []

    def record(
        self,
        request_id: str,
        tool_name: str,
        status: str,
        duration_ms: float,
        args: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> ToolInvocationLog:
        entry = ToolInvocationLog(
            request_id=request_id,
            tool_name=tool_name,
            status=status,
            duration_ms=round(duration_ms, 3),
            timestamp=datetime.now(timezone.utc).isoformat(),
            sanitized_args=sanitize_value(args or {}),
            error_message=sanitize_value(error_message) if error_message else None,
        )
        self.records.append(entry)
        self.logger.info(entry.to_json())
        return entry


def execute_with_audit(
    audit_logger: StructuredAuditLogger,
    request_id: str,
    tool_name: str,
    func: Callable[..., ToolResult],
    **kwargs: Any,
) -> ToolResult:
    """Execute a tool function, measure duration, handle timeouts/errors, and write a structured log."""
    start = time.perf_counter()
    try:
        result = func(request_id=request_id, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result.duration_ms = round(elapsed_ms, 3)
        result.request_id = request_id
        audit_logger.record(
            request_id=request_id,
            tool_name=tool_name,
            status=result.status,
            duration_ms=elapsed_ms,
            args=kwargs,
            error_message=result.error,
        )
        return result
    except TimeoutError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        err_msg = f"Tool '{tool_name}' timed out: {exc}"
        audit_logger.record(
            request_id=request_id,
            tool_name=tool_name,
            status="timeout",
            duration_ms=elapsed_ms,
            args=kwargs,
            error_message=err_msg,
        )
        return ToolResult(
            tool_name=tool_name,
            status="timeout",
            data={},
            evidence_source="timeout_guard",
            error=err_msg,
            duration_ms=round(elapsed_ms, 3),
            request_id=request_id,
        )
    except ValueError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        err_msg = f"Invalid tool parameters for '{tool_name}': {exc}"
        audit_logger.record(
            request_id=request_id,
            tool_name=tool_name,
            status="error",
            duration_ms=elapsed_ms,
            args=kwargs,
            error_message=err_msg,
        )
        return ToolResult(
            tool_name=tool_name,
            status="error",
            data={},
            evidence_source="validation_guard",
            error=err_msg,
            duration_ms=round(elapsed_ms, 3),
            request_id=request_id,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        err_msg = f"Service failure in '{tool_name}': {exc}"
        audit_logger.record(
            request_id=request_id,
            tool_name=tool_name,
            status="unavailable",
            duration_ms=elapsed_ms,
            args=kwargs,
            error_message=err_msg,
        )
        return ToolResult(
            tool_name=tool_name,
            status="unavailable",
            data={},
            evidence_source="exception_guard",
            error=err_msg,
            duration_ms=round(elapsed_ms, 3),
            request_id=request_id,
        )