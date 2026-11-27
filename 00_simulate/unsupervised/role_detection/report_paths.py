"""Percorsi coerenti per i report della role detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPORT_ROOT = Path(__file__).resolve().parents[3] / "07_reports" / "unsupervised" / "role_detection"


def normalize_graph_mode(mode: str) -> str:
    return "address_wallet" if mode == "address_wallet_feature" else mode


def report_mode(mode: str) -> str:
    return "address_wallet_feature" if normalize_graph_mode(mode) == "address_wallet" else mode


def split_report_paths(version: str, mode: str, filename: str) -> dict[str, Path]:
    root = REPORT_ROOT / version / report_mode(mode)
    return {"history": root / "history" / filename, "analysis": root / "analysis" / filename}


def analysis_artifact_path(version: str, mode: str, filename: str) -> Path:
    return REPORT_ROOT / version / report_mode(mode) / "analysis" / filename


def write_split_json(report: dict[str, Any], paths: dict[str, Path]) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
