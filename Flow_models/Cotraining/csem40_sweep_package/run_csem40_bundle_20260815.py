#!/usr/bin/env python3
"""Execute one generated CSEM sweep bundle, continuing after failed runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def expected_loss_paths(result_dir: Path, arms: str) -> list[Path]:
    arm_paths = {
        "terminal_kl": result_dir
        / "run_terminal_kl"
        / "dataframes"
        / "loss_history.csv",
        "norm": result_dir
        / "run_scale_norm"
        / "dataframes"
        / "loss_history.csv",
    }
    return [arm_paths[arm.strip()] for arm in arms.split(",") if arm.strip()]


def run_one(record: dict[str, Any], bundle_index: int) -> dict[str, Any]:
    run_id = str(record["run_id"])
    result_dir = Path(record["result_dir"])
    run_log = Path(record["run_log"])
    status_path = Path(record["status_path"])
    basedir = Path(record["basedir"])
    command = [str(part) for part in record["command"]]
    expected_paths = expected_loss_paths(
        result_dir, str(record["args"].get("arms", "terminal_kl,norm"))
    )

    if status_path.exists():
        previous = read_json(status_path)
        if previous.get("state") == "completed" and all(
            path.exists() for path in expected_paths
        ):
            print(f"[{run_id}] already complete; skipping", flush=True)
            return previous
    if result_dir.exists():
        raise FileExistsError(
            f"[{run_id}] result directory exists without a valid completed status: "
            f"{result_dir}. Refusing to overwrite it."
        )

    run_log.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    status: dict[str, Any] = {
        "bundle_index": bundle_index,
        "run_id": run_id,
        "state": "running",
        "started_unix": start_time,
        "command": command,
        "result_dir": str(result_dir),
        "run_log": str(run_log),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_atomic(status_path, status)
    print(
        f"[{run_id}] starting ({record.get('design_label')}); log={run_log}",
        flush=True,
    )

    with run_log.open("w", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(f"run_id={run_id}\n")
        log_handle.write(f"command={record.get('command_shell', command)}\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=basedir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=os.environ.copy(),
        )

    elapsed = time.time() - start_time
    missing = [str(path) for path in expected_paths if not path.exists()]
    state = "completed" if completed.returncode == 0 and not missing else "failed"
    status.update(
        {
            "state": state,
            "returncode": completed.returncode,
            "finished_unix": time.time(),
            "elapsed_seconds": elapsed,
            "missing_expected_outputs": missing,
        }
    )
    write_json_atomic(status_path, status)
    print(
        f"[{run_id}] {state}, rc={completed.returncode}, "
        f"elapsed={elapsed / 60.0:.1f}m",
        flush=True,
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle_path = args.bundle.resolve()
    bundle = read_json(bundle_path)
    bundle_index = int(bundle["bundle_index"])

    statuses: list[dict[str, Any]] = []
    for record in bundle["runs"]:
        try:
            statuses.append(run_one(record, bundle_index))
        except Exception as exception:
            run_id = str(record.get("run_id", "unknown"))
            failure = {
                "bundle_index": bundle_index,
                "run_id": run_id,
                "state": "worker_exception",
                "error": repr(exception),
                "traceback": traceback.format_exc(),
                "finished_unix": time.time(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
            status_path = Path(record["status_path"])
            write_json_atomic(status_path, failure)
            statuses.append(failure)
            print(f"[{run_id}] worker exception: {exception}", flush=True)

    bundle_status_path = (
        Path(bundle["sweep_root"])
        / "status"
        / f"bundle_{bundle_index:03d}_status.json"
    )
    write_json_atomic(
        bundle_status_path,
        {
            "bundle_index": bundle_index,
            "bundle_path": str(bundle_path),
            "statuses": statuses,
        },
    )
    failed = [status for status in statuses if status.get("state") != "completed"]
    print(
        f"bundle {bundle_index:03d}: {len(statuses) - len(failed)} completed, "
        f"{len(failed)} failed",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
