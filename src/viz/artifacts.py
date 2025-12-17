"""
Utilities for managing visualization artifacts.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_visual_dir(run_id: str) -> Path:
    """Get or create the output directory for visuals for a given run ID."""
    out = repo_root() / "artifacts" / "visuals" / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_pngs(run_id: str) -> list[Path]:
    """List all PNG files for a given run ID (including llm_code subfolder)."""
    out = repo_root() / "artifacts" / "visuals" / run_id
    if not out.exists():
        return []
    return sorted(out.rglob("*.png"))
