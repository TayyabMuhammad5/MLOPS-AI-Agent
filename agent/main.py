"""CLI and programmatic entrypoint for the MLOps AI Agent (`agent/main.py`)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import AgentRequest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Production-Ready MLOps AI Agent CLI")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=(
            "Find the best recent model, identify the dataset version used to train it, "
            "check the deployment status, and explain whether retraining is necessary."
        ),
        help="Natural-language engineering query for the MLOps Agent",
    )
    parser.add_argument("--namespace", "-n", type=str, default="mlops-prod", help="Target Kubernetes namespace")
    parser.add_argument(
        "--confirm-cluster-change",
        action="store_true",
        help="Explicitly authorize state-changing Kubernetes operations",
    )
    args = parser.parse_args(argv)

    orchestrator = MLOpsAgentOrchestrator()
    req = AgentRequest(
        query=args.query,
        namespace=args.namespace,
        confirm_cluster_change=args.confirm_cluster_change,
    )
    response = orchestrator.handle_request(req)
    print(json.dumps(response.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
