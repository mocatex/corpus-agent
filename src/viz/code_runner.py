"""
Module to run user-generated Python code in a controlled environment, capturing outputs and produced PNG files.
"""

from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from .artifacts import run_visual_dir, repo_root
from .db import query_df  # only used to build wrapper text


@dataclass
class CodeRunResult:
    out_dir: Path
    code_path: Path
    stdout_path: Path
    stderr_path: Path
    exit_code: int
    produced_pngs: list[Path]


def run_generated_code(run_id: str, user_code: str, timeout_sec: int = 60) -> CodeRunResult:
    """
    Run user-generated code in a controlled environment, capturing outputs and any produced PNG files.

    :param run_id: the pipeline run ID
    :param user_code: the user-generated Python code to execute
    :param timeout_sec: maximum time to allow for code execution
    :return: CodeRunResult with details of the execution
    """
    out_dir = run_visual_dir(run_id) / "llm_code"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    code_path = out_dir / f"generated_viz_{ts}.py"
    wrapper_path = out_dir / f"_wrapper_{ts}.py"
    stdout_path = out_dir / f"stdout_{ts}.txt"
    stderr_path = out_dir / f"stderr_{ts}.txt"

    code_path.write_text(user_code, encoding="utf-8")

    # Wrapper provides query_df + save_fig + OUT_DIR + run_id
    wrapper = f"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib.pyplot as plt

from viz.db import query_df  # SQLAlchemy-based query_df

RUN_ID = os.environ.get("RUN_ID")
OUT_DIR = Path(os.environ.get("OUT_DIR", ".")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(fig, filename: str) -> str:
    if not filename.endswith(".png"):
        raise ValueError("filename must end with .png")
    path = (OUT_DIR / filename).resolve()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)

# --- user code below ---
run_id = RUN_ID
{user_code}
"""
    wrapper_path.write_text(wrapper, encoding="utf-8")

    src_path = str(repo_root() / "src")
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path + (os.pathsep + old if old else "")
    env["RUN_ID"] = run_id
    env["OUT_DIR"] = str(out_dir)

    proc = subprocess.run(
        [sys.executable, str(wrapper_path)],
        cwd=str(out_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )

    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    produced_pngs = sorted(out_dir.glob("*.png"))

    return CodeRunResult(
        out_dir=out_dir,
        code_path=code_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        exit_code=proc.returncode,
        produced_pngs=produced_pngs,
    )
