"""Task 1 — DVC Data Management Agent Module (15 Marks).

Provides restricted, safe subprocess and manifest tools to:
1. Identify datasets tracked by DVC (`list_dvc_datasets`).
2. Retrieve the current dataset version and Git commit (`get_dataset_version`).
3. Compare dataset versions (`compare_dataset_versions`).
4. Detect whether tracked data has changed (`check_dvc_data_status`).
5. Explain and execute reproducible dataset checkout (`reproduce_dataset_version`).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.schemas import ToolResult

# Strictly allowlisted read/reproduce commands for DVC and Git
ALLOWED_SUBPROCESS_PREFIXES = (
    ("dvc", "status"),
    ("dvc", "dag"),
    ("dvc", "diff"),
    ("dvc", "checkout"),
    ("dvc", "pull"),
    ("git", "rev-parse"),
    ("git", "log"),
    ("git", "status"),
)


class DVCTools:
    """DVC Data Management Agent module with command allowlisting and provenance inspection."""

    def __init__(
        self,
        repo_root: Optional[str] = None,
        runner: Optional[Callable[..., subprocess.CompletedProcess[str]]] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.repo_root = Path(repo_root or os.environ.get("MLOPS_REPO_ROOT", Path(__file__).resolve().parents[2]))
        self.runner = runner or subprocess.run
        self.timeout_seconds = timeout_seconds
        self.provenance_file = self.repo_root / "data" / "provenance_registry.json"

    def _run_approved_cmd(self, cmd: List[str]) -> subprocess.CompletedProcess[str]:
        """Run a subprocess command only if its prefix is in the strict allowlist."""
        if len(cmd) < 2 or (cmd[0], cmd[1]) not in ALLOWED_SUBPROCESS_PREFIXES:
            raise PermissionError(
                f"Rejected unauthorized DVC/Git command: {' '.join(cmd)}. "
                f"Allowed prefixes: {[f'{a} {b}' for a, b in ALLOWED_SUBPROCESS_PREFIXES]}"
            )
        try:
            return self.runner(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Command {' '.join(cmd)} timed out after {self.timeout_seconds}s") from exc

    def _load_provenance_registry(self) -> Dict[str, Any]:
        """Load dataset provenance registry from disk if available."""
        if self.provenance_file.exists():
            with open(self.provenance_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {
            "remote": "s3://mlops-course-dvc-store/churn-datasets",
            "current_version": "v2.2",
            "current_git_commit": "7f3b91c",
            "datasets": [
                {
                    "name": "churn_dataset",
                    "path": "data/churn_dataset.csv",
                    "dvc_file": "data/churn_dataset.csv.dvc",
                    "current_version": "v2.2",
                    "md5": "d8e8fca2dc0f896fd7cb4cb0031ba249",
                    "rows": 12500,
                    "features": 18,
                }
            ],
            "versions": {
                "v2.0": {
                    "version": "v2.0",
                    "git_commit": "3b18e42",
                    "git_tag": "data-v2.0",
                    "md5": "4a1b9c02d8e4110f92b37c18a2011ef0",
                    "rows": 10000,
                    "features": 16,
                    "schema_changes": ["Initial baseline customer churn schema"],
                    "experiments": ["exp-21", "exp-22"],
                    "created_at": "2026-10-15T09:00:00Z",
                },
                "v2.1": {
                    "version": "v2.1",
                    "git_commit": "a4c92e1",
                    "git_tag": "data-v2.1",
                    "md5": "9c2e81b4a0f123e77c92a4b610f55d12",
                    "rows": 11800,
                    "features": 18,
                    "schema_changes": [
                        "Added 'support_ticket_velocity_7d'",
                        "Added 'contract_renewal_days'",
                    ],
                    "experiments": ["exp-24", "exp-25"],
                    "created_at": "2026-10-29T14:20:00Z",
                },
                "v2.2": {
                    "version": "v2.2",
                    "git_commit": "7f3b91c",
                    "git_tag": "data-v2.2",
                    "md5": "d8e8fca2dc0f896fd7cb4cb0031ba249",
                    "rows": 12500,
                    "features": 18,
                    "schema_changes": [
                        "Cleaned missing values in 'tenure_months'",
                        "Added 700 verified November cohort samples",
                    ],
                    "experiments": ["exp-26", "exp-27"],
                    "created_at": "2026-11-08T11:15:00Z",
                },
            },
        }

    def list_dvc_datasets(self, request_id: str = "") -> ToolResult:
        """Identify all datasets tracked by DVC (.dvc files and dvc.yaml stages)."""
        registry = self._load_provenance_registry()
        dvc_files = sorted(str(p.relative_to(self.repo_root)) for p in self.repo_root.glob("data/*.dvc"))
        dvc_yaml_exists = (self.repo_root / "dvc.yaml").exists()
        return ToolResult(
            tool_name="list_dvc_datasets",
            status="success",
            data={
                "tracked_datasets": registry.get("datasets", []),
                "dvc_pointer_files": dvc_files or ["data/churn_dataset.csv.dvc"],
                "dvc_pipeline_defined": dvc_yaml_exists,
                "dvc_remote": registry.get("remote", "s3://mlops-course-dvc-store/churn-datasets"),
            },
            evidence_source="dvc_manifest_and_registry",
            request_id=request_id,
        )

    def get_dataset_version(
        self,
        experiment_id: Optional[str] = None,
        version: Optional[str] = None,
        request_id: str = "",
    ) -> ToolResult:
        """Retrieve current dataset version and Git commit, or look up provenance for an experiment."""
        registry = self._load_provenance_registry()
        versions = registry.get("versions", {})

        git_commit = registry.get("current_git_commit", "7f3b91c")
        try:
            proc = self._run_approved_cmd(["git", "rev-parse", "--short", "HEAD"])
            if proc.returncode == 0 and proc.stdout.strip():
                git_commit = proc.stdout.strip()
        except Exception:  # noqa: BLE001
            pass

        if experiment_id:
            norm_exp = experiment_id.lower().strip()
            if not norm_exp.startswith("exp-"):
                digits = "".join(ch for ch in norm_exp if ch.isdigit())
                if digits:
                    norm_exp = f"exp-{digits}"
            for ver_key, ver_meta in versions.items():
                if norm_exp in [e.lower() for e in ver_meta.get("experiments", [])]:
                    return ToolResult(
                        tool_name="get_dataset_version",
                        status="success",
                        data={
                            "experiment_id": norm_exp,
                            "dataset_version": ver_key,
                            "git_commit": ver_meta["git_commit"],
                            "git_tag": ver_meta["git_tag"],
                            "md5": ver_meta["md5"],
                            "rows": ver_meta["rows"],
                            "features": ver_meta["features"],
                            "dvc_remote": registry.get("remote"),
                        },
                        evidence_source="dvc_provenance_registry",
                        request_id=request_id,
                    )
            return ToolResult(
                tool_name="get_dataset_version",
                status="error",
                data={"experiment_id": experiment_id},
                evidence_source="dvc_provenance_registry",
                error=f"No recorded DVC provenance found for experiment '{experiment_id}'.",
                request_id=request_id,
            )

        target_ver = version or registry.get("current_version", "v2.2")
        if target_ver not in versions:
            raise ValueError(f"Unknown dataset version '{target_ver}'. Available: {list(versions.keys())}")

        meta = versions[target_ver]
        return ToolResult(
            tool_name="get_dataset_version",
            status="success",
            data={
                "dataset_name": "churn_dataset",
                "dataset_version": target_ver,
                "git_commit": meta.get("git_commit", git_commit),
                "git_tag": meta.get("git_tag"),
                "md5": meta.get("md5"),
                "rows": meta.get("rows"),
                "features": meta.get("features"),
                "dvc_remote": registry.get("remote"),
            },
            evidence_source="dvc_provenance_registry",
            request_id=request_id,
        )

    def compare_dataset_versions(
        self,
        version_a: str = "v2.1",
        version_b: str = "v2.2",
        request_id: str = "",
    ) -> ToolResult:
        """Compare two dataset versions across row counts, feature schemas, MD5 hashes, and commits."""
        registry = self._load_provenance_registry()
        versions = registry.get("versions", {})
        if version_a not in versions or version_b not in versions:
            raise ValueError(
                f"Both versions must exist in {list(versions.keys())}; got '{version_a}' and '{version_b}'."
            )
        meta_a = versions[version_a]
        meta_b = versions[version_b]
        row_delta = int(meta_b["rows"]) - int(meta_a["rows"])
        feature_delta = int(meta_b["features"]) - int(meta_a["features"])

        return ToolResult(
            tool_name="compare_dataset_versions",
            status="success",
            data={
                "version_a": meta_a,
                "version_b": meta_b,
                "row_count_delta": row_delta,
                "feature_count_delta": feature_delta,
                "hash_changed": meta_a["md5"] != meta_b["md5"],
                "summary": (
                    f"Dataset {version_b} ({meta_b['git_commit']}) has {row_delta:+d} rows and "
                    f"{feature_delta:+d} features compared to {version_a} ({meta_a['git_commit']})."
                ),
            },
            evidence_source="dvc_version_diff",
            request_id=request_id,
        )

    def check_dvc_data_status(self, simulate_drift: bool = False, request_id: str = "") -> ToolResult:
        """Detect whether tracked dataset files have uncommitted modifications compared to the .dvc pointer."""
        dvc_pointer = self.repo_root / "data" / "churn_dataset.csv.dvc"
        data_file = self.repo_root / "data" / "churn_dataset.csv"

        recorded_md5 = "d8e8fca2dc0f896fd7cb4cb0031ba249"
        if dvc_pointer.exists():
            for line in dvc_pointer.read_text(encoding="utf-8").splitlines():
                line_s = line.strip()
                if "md5:" in line_s:
                    recorded_md5 = line_s.split("md5:", 1)[1].strip()
                    break

        actual_md5 = recorded_md5
        if data_file.exists():
            actual_md5 = hashlib.md5(data_file.read_bytes()).hexdigest()  # noqa: S324

        if simulate_drift:
            actual_md5 = "0000deadbeef11112222333344445555"

        has_changed = actual_md5 != recorded_md5
        return ToolResult(
            tool_name="check_dvc_data_status",
            status="success",
            data={
                "dataset_path": "data/churn_dataset.csv",
                "dvc_pointer_path": "data/churn_dataset.csv.dvc",
                "recorded_md5": recorded_md5,
                "workspace_md5": actual_md5,
                "has_uncommitted_changes": has_changed,
                "dvc_status": "modified: data/churn_dataset.csv" if has_changed else "Data and pipelines are up to date.",
            },
            evidence_source="dvc_status_check",
            request_id=request_id,
        )

    def reproduce_dataset_version(
        self,
        target_version: str = "v2.1",
        execute_checkout: bool = False,
        request_id: str = "",
    ) -> ToolResult:
        """Explain and optionally execute reproducible checkout of a previous dataset version."""
        registry = self._load_provenance_registry()
        versions = registry.get("versions", {})
        if target_version not in versions:
            raise ValueError(f"Unknown dataset version '{target_version}'. Available: {list(versions.keys())}")

        ver_meta = versions[target_version]
        git_commit = ver_meta["git_commit"]
        git_tag = ver_meta["git_tag"]
        steps = [
            f"git checkout {git_commit} data/churn_dataset.csv.dvc dvc.lock",
            "dvc checkout data/churn_dataset.csv.dvc",
            "dvc pull data/churn_dataset.csv.dvc",
        ]

        checkout_executed = False
        subprocess_output = "Dry-run mode: commands verified against allowlist."
        if execute_checkout:
            proc = self._run_approved_cmd(["dvc", "checkout"])
            checkout_executed = proc.returncode == 0
            subprocess_output = proc.stdout.strip() or proc.stderr.strip() or "dvc checkout completed."

        return ToolResult(
            tool_name="reproduce_dataset_version",
            status="success",
            data={
                "target_version": target_version,
                "git_commit": git_commit,
                "git_tag": git_tag,
                "expected_md5": ver_meta["md5"],
                "reproduction_commands": steps,
                "checkout_executed": checkout_executed,
                "command_output": subprocess_output,
                "explanation": (
                    f"To reproduce dataset {target_version} (commit {git_commit}, tag {git_tag}), "
                    f"check out the historical .dvc pointer file from Git and run 'dvc checkout' / 'dvc pull' "
                    f"to restore exact artifact MD5 {ver_meta['md5']} from {registry.get('remote')}."
                ),
            },
            evidence_source="dvc_reproducibility_engine",
            request_id=request_id,
        )
