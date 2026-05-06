#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from typing import Any


DEFAULT_EXPECTED = {
    "forward": 21,
    "inverse": 35,
    "completion": 8,
    "degraded": 2,
}

OOM_RE = re.compile(
    r"CUDA out of memory|CUDA OOM|OutOfMemoryError|torch\.OutOfMemoryError|out of memory",
    re.IGNORECASE,
)
FAILED_RE = re.compile(r"FAILED\s+\S+\s+status=", re.IGNORECASE)
ERROR_RE = re.compile(
    r"Traceback \(most recent call last\):|FAILED\s+\S+\s+status=|"
    r"(?:RuntimeError|ValueError|FileNotFoundError|ModuleNotFoundError|ImportError|"
    r"AssertionError|CalledProcessError):|returned non-zero exit status",
    re.IGNORECASE,
)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_cmd(
    cmd: list[str],
    cwd: pathlib.Path,
    timeout: int = 20,
) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout.strip()
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_log(run_root: pathlib.Path, text: str) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "monitor.log").open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def load_state(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(path: pathlib.Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def has_metric_stats(path: pathlib.Path) -> bool:
    return '"metric_stats"' in read_text(path)


def count_eval_jsons(root: pathlib.Path, filename: str = "heldout_eval.json") -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob(filename) if has_metric_stats(path))


def count_degraded(root: pathlib.Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.glob("degraded_inversion_*.json") if has_metric_stats(path))


def default_experiment_root(project_root: pathlib.Path) -> pathlib.Path:
    configured = os.environ.get("GMES_EXPERIMENT_ROOT")
    if configured:
        return pathlib.Path(configured).expanduser().resolve()
    datafolder_root = project_root / "DATAFOLDER" / "ExperimentRuns"
    if datafolder_root.exists():
        return datafolder_root
    return project_root / "outputs" / "ExperimentRuns"


def read_status(run_root: pathlib.Path) -> str:
    status = read_text(run_root / "pipeline.status").strip()
    return status or "starting"


def read_tail(path: pathlib.Path, max_lines: int = 12) -> str:
    if not path.exists():
        return "(no pipeline log yet)"
    lines = read_text(path).splitlines()
    useful = [
        line
        for line in lines
        if "[epoch" in line
        or "skip " in line
        or "reuse " in line
        or "wrote " in line
        or "FAILED " in line
        or "DONE " in line
        or "Saved " in line
        or "CUDA OOM detected" in line
    ]
    selected = useful[-max_lines:] if useful else lines[-max_lines:]
    return "\n".join(selected) if selected else "(pipeline log is empty)"


def read_window(path: pathlib.Path, end_offset: int | None = None, max_bytes: int = 60000) -> str:
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        end = size if end_offset is None else min(max(end_offset, 0), size)
        start = max(0, end - max_bytes)
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(end - start)
        return data.decode("utf-8", errors="ignore")
    except OSError:
        return ""


def read_new_log(path: pathlib.Path, cursor: int) -> tuple[str, int, int]:
    if not path.exists():
        return "", 0, 0
    try:
        size = path.stat().st_size
        start = cursor if 0 <= cursor <= size else 0
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read()
        return data.decode("utf-8", errors="ignore"), start, size
    except OSError:
        return "", cursor, cursor


def classify_error(new_text: str, context: str | None = None) -> str | None:
    if not new_text:
        return None
    if OOM_RE.search(new_text):
        return "oom"
    if ERROR_RE.search(new_text):
        if FAILED_RE.search(new_text) and context and OOM_RE.search(context):
            return "oom"
        return "non_oom"
    return None


def trim_excerpt(text: str, max_lines: int = 80, max_chars: int = 12000) -> str:
    lines = text.splitlines()[-max_lines:]
    excerpt = "\n".join(lines)
    if len(excerpt) > max_chars:
        excerpt = excerpt[-max_chars:]
    return excerpt.strip() or "(empty error context)"


def event_hash(error_type: str, step: str, excerpt: str) -> str:
    normalized = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?", "<time>", excerpt)
    normalized = re.sub(r"\b\d{5,}\b", "<num>", normalized)
    payload = f"{error_type}\n{step}\n{normalized[-6000:]}"
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def gpu_summary(project_root: pathlib.Path) -> str:
    code, output = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        cwd=project_root,
        timeout=10,
    )
    if code != 0 or not output:
        return "GPU: unavailable"
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            idx, util, used, total = parts
            rows.append(f"GPU{idx}: util={util}%, mem={used}/{total} MiB")
    return "\n".join(rows) if rows else output


def active_process_summary(project_root: pathlib.Path) -> str:
    code, output = run_cmd(["pgrep", "-af", "experiments/(train|eval)_"], cwd=project_root, timeout=10)
    if code != 0 or not output:
        return "Active experiment processes: none detected"
    lines = []
    for line in output.splitlines()[:8]:
        lines.append(line[:177] + "..." if len(line) > 180 else line)
    return "Active experiment processes:\n" + "\n".join(lines)


def tmux_summary(project_root: pathlib.Path, session_name: str) -> str:
    code, _ = run_cmd(["tmux", "has-session", "-t", session_name], cwd=project_root, timeout=10)
    return f"tmux session {session_name}: {'running' if code == 0 else 'not found'}"


def active_codex_summary(project_root: pathlib.Path, state: dict[str, Any]) -> str:
    sessions = state.get("codex_sessions", {})
    active = []
    for event_id, info in sessions.items():
        session = str(info.get("session", ""))
        if not session:
            continue
        code, _ = run_cmd(["tmux", "has-session", "-t", session], cwd=project_root, timeout=10)
        if code == 0:
            active.append(f"{event_id}: {session}")
    if active:
        return "Codex repair: active\n" + "\n".join(active[:4])
    return "Codex repair: no active session"


def build_report(args: argparse.Namespace, state: dict[str, Any]) -> str:
    experiment_root = args.experiment_root
    forward = count_eval_jsons(experiment_root / "forward_exp1_braided_crossed")
    inverse = count_eval_jsons(experiment_root / "inverse_regression")
    completion = count_eval_jsons(experiment_root / "modality_completion")
    degraded = count_degraded(experiment_root / "robustness")
    done = (args.run_root / "pipeline.done").exists()
    failed = (args.run_root / "pipeline.failed").exists()
    status = read_status(args.run_root)
    if failed:
        state_name = "failed"
    elif done:
        state_name = "completed"
    else:
        state_name = "running"
    expected = args.expected
    report = [
        f"GMES experiment monitor update ({now_iso()})",
        f"Run ID: {args.run_id}",
        f"State: {state_name}; current step: {status}",
        (
            "Progress with metric_stats: "
            f"forward {forward}/{expected['forward']}, "
            f"inverse {inverse}/{expected['inverse']}, "
            f"completion {completion}/{expected['completion']}, "
            f"degraded {degraded}/{expected['degraded']}"
        ),
        tmux_summary(args.project_root, args.session_name),
        gpu_summary(args.project_root),
        active_process_summary(args.project_root),
        active_codex_summary(args.project_root, state),
        "Recent pipeline events:",
        read_tail(args.run_root / "pipeline.log", max_lines=10),
        f"Run root: {args.run_root}",
    ]
    return "\n".join(report)


def send_openclaw(args: argparse.Namespace, message: str) -> tuple[int, str]:
    if args.dry_run:
        return 0, "[dry-run] openclaw send skipped"
    if not args.openclaw_target:
        return 1, "OPENCLAW_TARGET is empty"
    return run_cmd(
        [
            args.openclaw_bin,
            "message",
            "send",
            "--channel",
            args.openclaw_channel,
            "--target",
            args.openclaw_target,
            "--message",
            message,
            "--json",
        ],
        cwd=args.project_root,
        timeout=60,
    )


def send_and_log(args: argparse.Namespace, message: str, label: str) -> None:
    code, output = send_openclaw(args, message)
    append_log(args.run_root, f"\n===== {label} =====\n{message}\nopenclaw_exit={code}\n{output}")


def codex_lock_active(args: argparse.Namespace, lock_path: pathlib.Path) -> bool:
    if not lock_path.exists():
        return False
    try:
        info = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        info = {}
    session = str(info.get("session", ""))
    if session:
        code, _ = run_cmd(["tmux", "has-session", "-t", session], cwd=args.project_root, timeout=10)
        if code == 0:
            return True
    try:
        lock_path.unlink()
    except OSError:
        pass
    return False


def shell_script_literal(value: str) -> str:
    return shlex.quote(value)


def spawn_codex_repair(
    args: argparse.Namespace,
    state: dict[str, Any],
    event_id: str,
    error_type: str,
    step: str,
    excerpt: str,
) -> tuple[bool, str]:
    repair_root = args.run_root / "codex_repair"
    repair_root.mkdir(parents=True, exist_ok=True)
    lock_path = repair_root / "active.lock"
    if codex_lock_active(args, lock_path):
        return False, f"Codex repair already active; lock={lock_path}"

    session = f"gmes_codex_repair_{args.run_id}_{event_id}".replace(".", "_")
    event_dir = repair_root / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = event_dir / "prompt.md"
    log_path = event_dir / "codex.log"
    last_message_path = event_dir / "last_message.md"
    status_path = event_dir / "status.env"
    script_path = event_dir / "run_codex.sh"

    prompt = f"""You are Codex running as an automatic repair agent for a GMESDataset experiment failure.

Constraints:
- Sandbox must remain workspace-write.
- Work in project root: {args.project_root}
- Do not interrupt the running tmux experiments window unless the pipeline is already stopped or failed.
- Prefer minimal, targeted changes. Do not revert unrelated user changes.
- Read the pipeline log, relevant per-run launcher logs, and source code before editing.
- If the failure is transient infrastructure or requires rerunning long experiments, document that clearly instead of making speculative edits.

Run context:
- Run ID: {args.run_id}
- Run root: {args.run_root}
- Current step: {step}
- Error type classified by monitor: {error_type}
- Pipeline log: {args.run_root / "pipeline.log"}

Recent error excerpt:
```text
{excerpt}
```

Tasks:
1. Identify the non-OOM root cause from logs and code.
2. Apply a minimal fix if there is a clear code or launcher issue.
3. Run lightweight validation such as syntax checks, dry-runs, or focused tests. Avoid launching full long experiments.
4. Final response must list changed files, validation performed, and whether the failed experiment step should be restarted.
"""
    write_text(prompt_path, prompt)

    lock_payload = {"event_id": event_id, "session": session, "event_dir": str(event_dir), "created_at": now_iso()}
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(lock_payload, handle, indent=2)
    except FileExistsError:
        if codex_lock_active(args, lock_path):
            return False, f"Codex repair already active; lock={lock_path}"
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(lock_payload, handle, indent=2)

    script = f"""#!/usr/bin/env bash
set -u
LOCK={shell_script_literal(str(lock_path))}
STATUS={shell_script_literal(str(status_path))}
LOG={shell_script_literal(str(log_path))}
LAST={shell_script_literal(str(last_message_path))}
PROMPT={shell_script_literal(str(prompt_path))}
PROJECT={shell_script_literal(str(args.project_root))}
CODEX={shell_script_literal(args.codex_bin)}
cleanup() {{
  rm -f "${{LOCK}}"
}}
trap cleanup EXIT
{{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "session={session}"
  echo "event_id={event_id}"
}} > "${{STATUS}}"
cd "${{PROJECT}}" || exit 1
"${{CODEX}}" exec -s workspace-write -a never -C "${{PROJECT}}" -o "${{LAST}}" - < "${{PROMPT}}" > "${{LOG}}" 2>&1
status=$?
{{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "exit_code=${{status}}"
}} >> "${{STATUS}}"
exit "${{status}}"
"""
    write_text(script_path, script)
    script_path.chmod(0o755)

    if args.dry_run:
        try:
            lock_path.unlink()
        except OSError:
            pass
        return True, f"[dry-run] would start tmux session {session}"

    code, output = run_cmd(["tmux", "new-session", "-d", "-s", session, "bash", str(script_path)], cwd=args.project_root, timeout=20)
    if code != 0:
        try:
            lock_path.unlink()
        except OSError:
            pass
        return False, f"failed to start Codex tmux session: {output}"

    sessions = state.setdefault("codex_sessions", {})
    sessions[event_id] = {
        "session": session,
        "event_dir": str(event_dir),
        "notified_complete": False,
        "started_at": now_iso(),
    }
    return True, f"started Codex repair session {session}; log={log_path}"


def make_error_message(
    args: argparse.Namespace,
    event_id: str,
    error_type: str,
    step: str,
    excerpt: str,
    codex_note: str,
) -> str:
    title = "GMES experiment error detected"
    return "\n".join(
        [
            f"{title} ({now_iso()})",
            f"Run ID: {args.run_id}",
            f"Current step: {step}",
            f"Error type: {error_type}",
            f"Event ID: {event_id}",
            f"Codex action: {codex_note}",
            f"Run root: {args.run_root}",
            "Recent error excerpt:",
            excerpt,
        ]
    )


def handle_error_event(args: argparse.Namespace, state: dict[str, Any], error_type: str, context: str) -> None:
    step = read_status(args.run_root)
    excerpt = trim_excerpt(context)
    digest = event_hash(error_type, step, excerpt)
    seen = state.setdefault("seen_error_hashes", [])
    if digest in seen:
        return
    seen.append(digest)
    del seen[:-50]

    event_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + digest[:8]
    codex_note = "not started for OOM; launcher retry or smaller batch is expected"
    if error_type != "oom":
        started, note = spawn_codex_repair(args, state, event_id, error_type, step, excerpt)
        codex_note = note if started else f"not started: {note}"

    message = make_error_message(args, event_id, error_type, step, excerpt, codex_note)
    write_text(args.run_root / "events" / f"{event_id}.txt", message + "\n")
    send_and_log(args, message, f"error_event {event_id}")


def poll_errors(args: argparse.Namespace, state: dict[str, Any]) -> None:
    log_path = args.run_root / "pipeline.log"
    if "pipeline_log_cursor" not in state:
        if args.scan_from_start or (args.run_root / "pipeline.failed").exists():
            state["pipeline_log_cursor"] = 0
        elif log_path.exists():
            state["pipeline_log_cursor"] = log_path.stat().st_size
        else:
            state["pipeline_log_cursor"] = 0

    cursor = int(state.get("pipeline_log_cursor", 0))
    new_text, start, end = read_new_log(log_path, cursor)
    state["pipeline_log_cursor"] = end

    if new_text:
        context = read_window(log_path, end_offset=start, max_bytes=args.context_bytes) + new_text
        error_type = classify_error(new_text, context)
        if error_type:
            handle_error_event(args, state, error_type, context)

    failed_file = args.run_root / "pipeline.failed"
    if failed_file.exists() and not state.get("failed_marker_seen"):
        state["failed_marker_seen"] = True
        context = read_window(log_path, max_bytes=args.context_bytes)
        error_type = classify_error(context) or "non_oom"
        handle_error_event(args, state, error_type, context)


def notify_codex_completions(args: argparse.Namespace, state: dict[str, Any]) -> None:
    sessions = state.get("codex_sessions", {})
    for event_id, info in list(sessions.items()):
        if info.get("notified_complete"):
            continue
        event_dir = pathlib.Path(str(info.get("event_dir", "")))
        status_path = event_dir / "status.env"
        if "exit_code=" not in read_text(status_path):
            continue
        last_message = read_text(event_dir / "last_message.md").strip()
        log_tail = read_tail(event_dir / "codex.log", max_lines=20)
        body = last_message if last_message else log_tail
        message = "\n".join(
            [
                f"GMES Codex repair completed ({now_iso()})",
                f"Run ID: {args.run_id}",
                f"Event ID: {event_id}",
                f"Session: {info.get('session', '')}",
                f"Status file: {status_path}",
                "Codex final output:",
                body[-6000:] if body else "(no output)",
            ]
        )
        send_and_log(args, message, f"codex_complete {event_id}")
        info["notified_complete"] = True


def any_active_codex(args: argparse.Namespace, state: dict[str, Any]) -> bool:
    for info in state.get("codex_sessions", {}).values():
        session = str(info.get("session", ""))
        if not session:
            continue
        code, _ = run_cmd(["tmux", "has-session", "-t", session], cwd=args.project_root, timeout=10)
        if code == 0:
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor GMES experiment runs and notify through OpenClaw.")
    parser.add_argument("--project-root", default=os.environ.get("GMES_PROJECT_ROOT", os.getcwd()))
    parser.add_argument("--experiment-root", default=os.environ.get("GMES_EXPERIMENT_ROOT"))
    parser.add_argument("--run-id", default=os.environ.get("GMES_RUN_ID", ""))
    parser.add_argument("--run-root", default=os.environ.get("GMES_RUN_ROOT", ""))
    parser.add_argument("--session-name", default=os.environ.get("GMES_TMUX_SESSION", "gmes_neurips_tables"))
    parser.add_argument("--report-interval-seconds", type=int, default=int(os.environ.get("GMES_MONITOR_REPORT_INTERVAL_SECONDS", os.environ.get("GMES_MONITOR_INTERVAL_SECONDS", "43200"))))
    parser.add_argument("--error-poll-seconds", type=int, default=int(os.environ.get("GMES_MONITOR_ERROR_POLL_SECONDS", "60")))
    parser.add_argument("--context-bytes", type=int, default=int(os.environ.get("GMES_MONITOR_CONTEXT_BYTES", "60000")))
    parser.add_argument("--openclaw-bin", default=os.environ.get("OPENCLAW_BIN", shutil.which("openclaw") or "openclaw"))
    parser.add_argument("--openclaw-channel", default=os.environ.get("OPENCLAW_CHANNEL", "feishu"))
    parser.add_argument("--openclaw-target", default=os.environ.get("OPENCLAW_TARGET", ""))
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN", shutil.which("codex") or "codex"))
    parser.add_argument("--scan-from-start", action="store_true", default=os.environ.get("GMES_MONITOR_SCAN_FROM_START", "0") == "1")
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("GMES_MONITOR_DRY_RUN", "0") == "1")
    parser.add_argument("--once", action="store_true", default=os.environ.get("GMES_MONITOR_ONCE", "0") == "1")
    args = parser.parse_args()

    args.project_root = pathlib.Path(args.project_root).expanduser().resolve()
    args.experiment_root = pathlib.Path(args.experiment_root).expanduser().resolve() if args.experiment_root else default_experiment_root(args.project_root)
    if args.run_root:
        args.run_root = pathlib.Path(args.run_root).expanduser().resolve()
        if not args.run_id:
            args.run_id = args.run_root.name
    else:
        if not args.run_id:
            raise SystemExit("Either --run-root or --run-id is required.")
        args.run_root = args.experiment_root / "tmux_runs" / args.run_id
    args.expected = DEFAULT_EXPECTED
    return args


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    state_path = args.run_root / "monitor_state.json"
    state = load_state(state_path)
    state.setdefault("started_at", now_iso())
    state.setdefault("monitor_version", 1)

    next_report = 0.0
    final_report_sent = bool(state.get("final_report_sent", False))
    append_log(
        args.run_root,
        (
            f"\n===== monitor_start =====\nstarted_at={now_iso()}\n"
            f"report_interval={args.report_interval_seconds}\n"
            f"error_poll={args.error_poll_seconds}\n"
            f"dry_run={args.dry_run}\n"
        ),
    )

    while True:
        poll_errors(args, state)
        notify_codex_completions(args, state)

        done = (args.run_root / "pipeline.done").exists()
        now = time.time()
        if now >= next_report or (done and not final_report_sent):
            report = build_report(args, state)
            write_text(args.run_root / "latest_report.txt", report + "\n")
            send_and_log(args, report, "report")
            next_report = now + max(args.report_interval_seconds, 1)
            if done:
                final_report_sent = True
                state["final_report_sent"] = True

        save_state(state_path, state)
        if args.once:
            break
        if done and final_report_sent and not any_active_codex(args, state):
            break
        time.sleep(max(args.error_poll_seconds, 1))


if __name__ == "__main__":
    main()
