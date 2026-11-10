"""Serial durable coordinator for the frozen full validation protocol.

Nothing starts on import. --launch starts this same coordinator without a
console window; --report only reads artifacts and writes report.md.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import socket
import sqlite3
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prepare", "memory", "retrieval", "generate", "prepare_vectors", "score")
SCORING_STAGES = ("prepare_vectors", "score")
METRICS = ("answer_correctness", "context_recall", "faithfulness")
PORT = 7860


def now():
    return datetime.now(UTC).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"1")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("该output已有coordinator或launch请求；拒绝并行启动") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_profile(path: Path, root: Path = ROOT) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not data:
        raise ValueError("profile必须是非空KB_键值JSON对象")
    profile = {}
    for key, value in data.items():
        if not re.fullmatch(r"KB_[A-Z0-9_]+", key):
            raise ValueError("profile只允许KB_环境键")
        if re.search(r"(?:^|_)(?:API_KEY|SECRET|PASSWORD|ACCESS_TOKEN|AUTH_TOKEN|CREDENTIALS?)(?:_|$)", key):
            raise ValueError("profile不得包含secret环境键")
        if value is None:
            value = ""
        elif not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if "\x00" in value or ("://" in value and urlsplit(value).username is not None):
            raise ValueError("profile值包含不支持的凭据或NUL")
        profile[key] = value
    original = dotenv_values(root / ".env")
    if profile.get("KB_PROVIDER", original.get("KB_PROVIDER", "ollama")) != "ollama":
        raise ValueError("完整真实验证要求ollama provider")
    if not profile.get("KB_DATA_DIR"):
        raise ValueError("profile必须显式指定独立KB_DATA_DIR克隆")
    ui_data = Path(original.get("KB_DATA_DIR") or "runtime")
    evaluation_data = Path(profile["KB_DATA_DIR"])
    ui_data = (root / ui_data).resolve() if not ui_data.is_absolute() else ui_data.resolve()
    evaluation_data = (root / evaluation_data).resolve() if not evaluation_data.is_absolute() else evaluation_data.resolve()
    if evaluation_data == ui_data:
        raise ValueError("评测数据目录必须与原UI数据目录分离")
    if profile.get("KB_QDRANT_URL", original.get("KB_QDRANT_URL", "")):
        raise ValueError("本流水线使用独立本地Qdrant克隆，不接受共享远端Qdrant")
    return profile


def child_environment(profile: dict[str, str] | None = None, root: Path = ROOT):
    # Start from the project's .env, not accidental KB_ values in a terminal.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("KB_")}
    environment.update(profile or {})
    environment.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(root / "src"))
    return environment


def project_python(root: Path):
    path = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not path.is_file():
        raise ValueError(f"项目Python不存在：{path}")
    return path


def dataset_schedule(root: Path):
    dataset = root / "data/eval/demo_cases.jsonl"
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    cases = [row for row in rows if row.get("split") == "test"]
    ids = [row["id"] for row in cases]
    eligible = [row for row in cases if row.get("reference") and row.get("kind") not in {"ambiguous", "unanswerable", "abstain"}]
    if len(ids) != 88 or len(set(ids)) != 88 or len(eligible) != 77:
        raise ValueError("冻结协议要求test88个唯一ID、77个可评分题；拒绝静默改变定义")
    if any(not re.fullmatch(r"[A-Za-z0-9_.:-]+", value) for value in ids):
        raise ValueError("case ID不能含逗号或不支持字符")
    return {"dataset": str(dataset.resolve()), "dataset_sha256": digest(dataset.read_bytes()), "ids": ids,
            "repeats": 3, "generation_slots": 264, "retrieval_slots": 352, "metric_slots": 693,
            "memory_conditions": 40, "memory_semantic_answers": 12}


def build_stages(root: Path, output: Path, python: Path, schedule: dict):
    benchmark = output / "benchmark"
    common = ["--run-dir", str(benchmark), "--dataset", schedule["dataset"], "--split", "test",
              "--scope", "all", "--mode", "agent", "--stability-repeats", "3",
              "--stability-ids", ",".join(schedule["ids"]), "--metric-timeout", "900", "--score-keep-alive", "10m"]
    commands = {name: [str(python), "-u", str(root / "scripts/benchmark.py"), phase, *common]
                for name, phase in (("prepare", "prepare"), ("retrieval", "retrieval"), ("generate", "generate"),
                                    ("prepare_vectors", "prepare-vectors"), ("score", "score"))}
    commands["memory"] = [str(python), "-u", str(root / "scripts/validate_memory.py"), "--output", str(output / "memory")]
    return [{"name": name, "command": commands[name]} for name in STAGES]


def frozen_definition(profile_path: Path, profile: dict, schedule: dict, stages: list):
    return {"schema": 1, "profile_sha256": digest(profile_path.read_bytes()), "profile": profile,
            "coordinator_sha256": digest(Path(__file__).read_bytes()), "schedule": schedule, "stages": stages}


def read_user_controls(output: Path):
    path = output / "user_controls.json"
    controls = read_json(path) if path.exists() else {"schema": 1, "scoring_disabled": False, "changes": []}
    if (not isinstance(controls, dict) or controls.get("schema") != 1
            or type(controls.get("scoring_disabled")) is not bool or not isinstance(controls.get("changes"), list)
            or (controls["changes"] and not controls["scoring_disabled"]
                and not (controls.get("generation_disabled") is True
                         and controls.get("score_metrics") == ["context_recall"]
                         and re.fullmatch(r"[a-f0-9]{64}", controls.get("score_driver_sha256", ""))))):
        raise ValueError("用户控制记录损坏，拒绝猜测是否执行评分")
    if "generation_disabled" in controls and type(controls["generation_disabled"]) is not bool:
        raise ValueError("用户生成控制记录损坏")
    return controls


def preserve_file(path: Path, data: bytes):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("已归档文件内容不一致，拒绝覆盖")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".tmp-" + uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def check_control_archives(output: Path, change: dict):
    old_sha = change.get("old_coordinator_sha256", "")
    old_fingerprint = change.get("previous_fingerprint", "")
    if not re.fullmatch(r"[a-f0-9]{64}", old_sha) or not re.fullmatch(r"[a-f0-9]{64}", old_fingerprint):
        raise ValueError("用户计划变更缺少原始冻结标识")
    archive = output / "controller_archive"
    source, manifest = archive / (old_sha + ".py"), archive / (old_fingerprint + ".manifest.json")
    if (not source.is_file() or digest(source.read_bytes()) != old_sha or not manifest.is_file()
            or digest(manifest.read_bytes()) != change.get("previous_manifest_sha256")):
        raise ValueError("用户计划变更的原源码/manifest归档校验失败")


def commit_disable_scoring(output: Path, manifest: dict, manifest_bytes: bytes, frozen: dict,
                           fingerprint: str, controls: dict, *, context_recall_only=False, driver_sha256=None):
    old_sha = manifest["frozen"]["coordinator_sha256"]
    if not re.fullmatch(r"[a-f0-9]{64}", old_sha):
        raise ValueError("原controller哈希格式无效，拒绝迁移")
    candidates = (output / "controller_archive" / (old_sha + ".py"),
                  output / "frozen_supervisor.py", Path(__file__))
    source = next((path.read_bytes() for path in candidates if path.is_file()
                   and digest(path.read_bytes()) == old_sha), None)
    if source is None:
        raise ValueError("缺少与原manifest哈希一致的frozen_supervisor.py，拒绝迁移")
    archive = output / "controller_archive"
    preserve_file(archive / (old_sha + ".py"), source)
    preserve_file(archive / (manifest["fingerprint"] + ".manifest.json"), manifest_bytes)
    identity = {"action": "stop_generation_context_recall_only" if context_recall_only else "disable_scoring",
                "previous_fingerprint": manifest["fingerprint"],
                "fingerprint": fingerprint, "old_coordinator_sha256": old_sha,
                "new_coordinator_sha256": frozen["coordinator_sha256"],
                "previous_manifest_sha256": digest(manifest_bytes)}
    if context_recall_only:
        identity.update(previous_change_id=controls["changes"][-1]["id"] if controls["changes"] else None,
                        score_metrics=["context_recall"], score_driver_sha256=driver_sha256,
                        generation_disabled=True)
    change = {**identity, "id": digest(canonical(identity)), "at": now(),
              "reason": "explicit_user_request",
              "skipped_stages": ["generate", "prepare_vectors"] if context_recall_only else list(SCORING_STAGES)}
    updated = {**controls, "scoring_disabled": not context_recall_only, "changes": [*controls["changes"], change]}
    if context_recall_only:
        updated.update(generation_disabled=True, score_metrics=["context_recall"], score_driver_sha256=driver_sha256)
    # Single commit point. The manifest, state and append-only log can be replayed after a crash.
    atomic_write(output / "user_controls.json", json.dumps(updated, ensure_ascii=False, indent=2))
    return updated


def sync_control_log(output: Path, controls: dict):
    path = output / "user_plan_changes.jsonl"
    try:
        recorded = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    except FileNotFoundError:
        recorded = set()
    except (ValueError, KeyError) as exc:
        raise ValueError("用户计划变更日志损坏，拒绝覆盖") from exc
    for change in controls["changes"]:
        check_control_archives(output, change)
        if change["id"] not in recorded:
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical(change) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            recorded.add(change["id"])


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied cannot prove that a PID is dead.
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def hidden_process_kwargs():
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {"start_new_session": True}


def owned_process_kwargs():
    options = hidden_process_kwargs()
    if os.name == "nt":
        options["creationflags"] |= 0x00000004  # CREATE_SUSPENDED; bind the Job before user code runs.
    return options


@contextmanager
def system_awake(warn, setter=None):
    """Temporary system-only request on this coordinator thread; display may sleep."""
    if setter is None and os.name != "nt":
        yield
        return
    if setter is None:
        try:
            import ctypes
            function = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
            function.argtypes = [ctypes.c_ulong]
            function.restype = ctypes.c_ulong
            setter = function
        except (OSError, AttributeError) as exc:
            warn("system_awake_request_failed", str(exc))
            yield
            return
    try:
        try:
            accepted = setter(0x80000001)  # Never DISPLAY_REQUIRED.
        except OSError:
            accepted = False
        if not accepted:
            warn("system_awake_request_failed")
        yield
    finally:
        try:
            restored = setter(0x80000000)
        except OSError:
            restored = False
        if not restored:
            warn("system_awake_restore_failed")


class OwnedProcessTree:
    """Windows Job Object tracks only this Popen and its inherited worker tree."""
    def __init__(self):
        self.process, self.handle, self.attached = None, None, False
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)

        class Limits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                        ("flags", wintypes.DWORD), ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t),
                        ("active_limit", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                        ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("read", "write", "other", "read_bytes", "write_bytes", "other_bytes")]

        class Extended(ctypes.Structure):
            _fields_ = [("basic", Limits), ("io", IO), ("process_memory", ctypes.c_size_t),
                        ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_longlong) for name in ("user", "kernel", "period_user", "period_kernel")] + [
                (name, wintypes.DWORD) for name in ("faults", "total", "active", "terminated")]

        class ThreadEntry(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("thread_id", wintypes.DWORD),
                        ("owner_pid", wintypes.DWORD), ("base_priority", wintypes.LONG),
                        ("delta_priority", wintypes.LONG), ("flags", wintypes.DWORD)]

        self.Accounting = Accounting
        self.ThreadEntry = ThreadEntry
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        for name in ("SetInformationJobObject", "QueryInformationJobObject"):
            function = getattr(self.kernel, name)
            function.restype = wintypes.BOOL
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.TerminateJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self.kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        for name in ("Thread32First", "Thread32Next"):
            function = getattr(self.kernel, name)
            function.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
            function.restype = wintypes.BOOL
        self.kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenThread.restype = wintypes.HANDLE
        self.kernel.GetProcessIdOfThread.argtypes = [wintypes.HANDLE]
        self.kernel.GetProcessIdOfThread.restype = wintypes.DWORD
        self.kernel.ResumeThread.argtypes = [wintypes.HANDLE]
        self.kernel.ResumeThread.restype = wintypes.DWORD
        self.handle = self.kernel.CreateJobObjectW(None, None)
        limits = Extended()
        limits.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
        if not self.handle or not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise OSError("无法建立本任务专用进程树Job Object")

    def attach(self, process):
        self.process = process
        if os.name == "nt" and not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise OSError("无法追踪本任务worker进程树；拒绝继续运行")
        self.attached = True

    def _suspended_primary_thread(self, process):
        # Popen closes CreateProcess's hThread. A new suspended process has one primary thread.
        if process.poll() is not None:
            raise OSError("挂起子进程已退出，拒绝恢复")
        snapshot = self.kernel.CreateToolhelp32Snapshot(0x00000004, 0)  # TH32CS_SNAPTHREAD.
        if snapshot == self.ctypes.c_void_p(-1).value:
            raise OSError("无法枚举自有挂起子进程线程")
        try:
            entry = self.ThreadEntry()
            entry.size = self.ctypes.sizeof(entry)
            owned = []
            present = self.kernel.Thread32First(snapshot, self.ctypes.byref(entry))
            while present:
                if entry.owner_pid == process.pid:
                    owned.append(entry.thread_id)
                present = self.kernel.Thread32Next(snapshot, self.ctypes.byref(entry))
            if self.ctypes.get_last_error() != 18 or len(owned) != 1:  # ERROR_NO_MORE_FILES.
                raise OSError("无法唯一确认自有挂起子进程主线程")
        finally:
            self.kernel.CloseHandle(snapshot)
        thread = self.kernel.OpenThread(0x0802, False, owned[0])  # QUERY_LIMITED_INFORMATION | SUSPEND_RESUME.
        if not thread:
            raise OSError("无法打开自有挂起子进程主线程")
        if self.kernel.GetProcessIdOfThread(thread) != process.pid or process.poll() is not None:
            self.kernel.CloseHandle(thread)
            raise OSError("主线程归属验证失败，拒绝恢复")
        return thread

    def attach_suspended(self, process):
        self.process = process
        if os.name != "nt":
            self.attach(process)
            return
        thread = self._suspended_primary_thread(process)
        try:
            self.attach(process)
            if self.kernel.ResumeThread(thread) != 1:
                raise OSError("自有主线程挂起计数异常，拒绝继续运行")
        finally:
            self.kernel.CloseHandle(thread)

    def stop_and_confirm(self):
        if self.process is None:
            return True
        # Popen.poll uses the owned process handle, avoiding a reused PID target.
        active = self.process.poll() is None
        if not self.attached:
            if active:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            return False  # Unknown descendants are not declared safely stopped.
        if os.name != "nt":
            if active:
                import signal
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
            return self.process.poll() is not None
        info = self.Accounting()

        def remaining():
            if not self.kernel.QueryInformationJobObject(self.handle, 1, self.ctypes.byref(info), self.ctypes.sizeof(info), None):
                raise OSError("无法确认本任务进程树退出")
            return info.active

        if remaining() and not self.kernel.TerminateJobObject(self.handle, 1):
            return False
        deadline = time.monotonic() + 10
        while remaining() and time.monotonic() < deadline:
            time.sleep(0.1)
        if remaining() != 0:
            return False
        try:
            self.process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return False
        return True

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class JournalCounts:
    def __init__(self, path: Path):
        self.path, self.offset, self.items = path, 0, {"generation": {}, "score": {}, "retrieval": {}}

    def scan(self):
        if not self.path.is_file():
            return
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read(4 * 1024 * 1024)
        end = data.rfind(b"\n")
        if end < 0:
            return
        for line in data[:end].splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            payload = record.get("payload") or {}
            phase = payload.get("phase")
            if record.get("state") != "completed" or phase not in self.items:
                continue
            if phase == "generation":
                key = payload.get("item")
            elif phase == "score":
                key = (payload.get("item"), payload.get("metric"))
            else:
                key = (payload.get("mode"), payload.get("case_id"))
            self.items[phase][key] = payload.get("status", "unknown")
        self.offset += end + 1


def memory_counts(path: Path):
    result = {"completed": 0, "planned": 40, "failed": 0, "semantic_completed": 0,
              "semantic_planned": 12, "triggered": 0, "mean_ratio_on": None}
    if not path.is_file():
        return result
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2) as connection:
            rows = [json.loads(row[0]) for row in connection.execute("SELECT body FROM results")]
        on = [row for row in rows if row.get("condition") == "on" and row.get("status") == "ok"]
        ratios = [row["ratio"] for row in on if isinstance(row.get("ratio"), (int, float))]
        result.update(completed=len(rows), failed=sum(row.get("status") != "ok" for row in rows),
                      semantic_completed=sum(row.get("answer_status") == "completed" for row in rows),
                      triggered=sum(row.get("triggered", False) for row in on),
                      mean_ratio_on=sum(ratios) / len(ratios) if ratios else None)
    except (OSError, sqlite3.Error, ValueError):
        pass
    return result


class Progress:
    def __init__(self, output: Path):
        self.output = output
        self.journal = JournalCounts(output / "benchmark/journal.jsonl")
        self.lock, self.value = threading.Lock(), {}
        self.memory_signature, self.memory_cache = None, None

    def update(self, state):
        self.journal.scan()
        controls = state.get("user_controls", {})
        selected_metrics = controls.get("score_metrics", list(METRICS))
        public = {"status": state.get("status"), "stage": state.get("current_stage"), "updated_at": state.get("updated_at")}
        public["resource_cleanup"] = state.get("resource_cleanup", {}).get("status")
        for phase, planned in (("generation", 264), ("score", 693), ("retrieval", 352)):
            if phase == "score" and selected_metrics == ["context_recall"]:
                planned = 231
                values = [value for key, value in self.journal.items[phase].items() if key[1] == "context_recall"]
            else:
                values = list(self.journal.items[phase].values())
            public[phase] = {"completed": len(values), "planned": planned,
                             "valid": sum(value == "success" for value in values),
                             "failed_or_blocked": sum(value != "success" for value in values),
                             "not_started": max(0, planned - len(values))}
        public["score"]["disabled"] = controls.get("scoring_disabled", False)
        if controls.get("generation_disabled"):
            public["generation"]["disabled"] = True
        if selected_metrics == ["context_recall"]:
            selection = read_json(self.output / "benchmark/context_recall_selection.json", {})
            summary = read_json(self.output / "benchmark/summary.json", {})
            metric = summary.get("stability", {}).get("aggregate", {}).get("metrics", {}).get("context_recall", {})
            public["score"].update(metrics=selected_metrics, selected_planned=selection.get("expected_count"),
                                   mean_valid_only=metric.get("mean_valid_only"), all_metrics_planned=693)
        database = self.output / "memory/ledger.sqlite"
        paths = [database, database.with_name(database.name + "-wal")]
        signature = tuple((path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None for path in paths)
        if signature != self.memory_signature or self.memory_cache is None:
            self.memory_cache = memory_counts(database)
            self.memory_signature = signature
        public["memory"] = copy.deepcopy(self.memory_cache)
        with self.lock:
            self.value = public

    def public(self):
        with self.lock:
            return copy.deepcopy(self.value)


PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>系统验证进度</title>
<style>body{font:16px system-ui;max-width:900px;margin:40px auto;padding:20px;background:#f7f8fc;color:#233}section{background:white;padding:24px;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:14px 8px;border-bottom:1px solid #e4e8ee}small{color:#667}</style>
<h1>系统验证进度</h1><section><h2 id="stage">正在读取进度…</h2><table>
<tr><th>验证项目</th><th>处理进度</th><th>当前结果</th></tr>
<tr><td>知识问答</td><td id="qa">0 / 264</td><td id="qa-result">有效完成 0；失败或待澄清 0</td></tr>
<tr><td>检索对比</td><td id="retrieval">0 / 352</td><td id="retrieval-result">等待处理</td></tr>
<tr><td>回答评分</td><td id="score">0 / 693</td><td id="score-result">等待处理</td></tr>
<tr><td>长对话与压缩</td><td id="memory">0 / 40</td><td id="semantic">最终追问 0 / 12</td></tr></table>
<p>处理完成不代表答案正确，最终指标以报告为准。</p><small id="notice">验证结束并释放模型后，会恢复问答界面；届时可刷新本页。</small></section>
<script>const stages={prepare:'冻结评测计划',memory:'验证长对话与压缩',retrieval:'对比检索方式',generate:'执行知识问答',prepare_vectors:'准备评分资料',score:'计算回答指标'};
function put(id,text){document.getElementById(id).textContent=text}
async function tick(){try{let r=await fetch('/api/progress',{cache:'no-store'});if(!r.ok)return;let p=await r.json(),q=p.generation||{},t=p.retrieval||{},s=p.score||{},m=p.memory||{};
put('stage',p.resource_cleanup==='checking'?'正在释放评测模型内存':stages[p.stage]||(p.status==='completed'?'验证执行已结束':'验证任务状态已更新'));
put('qa',(q.completed||0)+' / 264');put('qa-result',(q.disabled?'已停止；':'')+'有效完成 '+(q.valid||0)+'；失败或待澄清 '+(q.failed_or_blocked||0));
put('retrieval',(t.completed||0)+' / 352');put('retrieval-result','已完成 '+(t.valid||0)+'；失败 '+(t.failed_or_blocked||0));
let recallOnly=(s.metrics||[]).join(',')==='context_recall';
put('score',(s.completed||0)+' / '+(recallOnly?(s.selected_planned??'…'):(s.planned||693)));
put('score-result',s.disabled?'已停用；已评分 '+(s.completed||0)+'；未评分 '+(s.not_started||0):(recallOnly?'仅 ContextRecall；':'')+'有效 '+(s.valid||0)+'；失败或受阻 '+(s.failed_or_blocked||0)+(recallOnly?('；均值 '+(s.mean_valid_only==null?'待计算':s.mean_valid_only.toFixed(4))+'；原计划 231 项'):'；未评分 '+(s.not_started||0)));
put('memory',(m.completed||0)+' / 40');put('semantic','最终追问 '+(m.semantic_completed||0)+' / 12');
if(p.resource_cleanup==='unconfirmed')put('notice','模型内存释放尚未确认，问答界面暂未自动启动。');
}catch(e){put('notice','状态页已停止。若问答界面已恢复，请刷新本页。')}}tick();setInterval(tick,3000);</script></html>"""


class StatusPage:
    def __init__(self, progress: Progress, port=PORT):
        self.progress, self.port, self.server, self.thread = progress, port, None, None

    def start(self):
        progress = self.progress

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    body, kind, status = PAGE.encode("utf-8"), "text/html; charset=utf-8", 200
                elif self.path == "/api/progress":
                    body, kind, status = canonical(progress.public()).encode("utf-8"), "application/json; charset=utf-8", 200
                else:
                    body, kind, status = b"not found", "text/plain", 404
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, name="validation-status", daemon=True)
        self.thread.start()

    def close(self):
        if self.server:
            if self.thread and self.thread.is_alive():
                self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=3)


def port_busy(port=PORT):
    with socket.socket() as connection:
        connection.settimeout(0.2)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def validate_memory_coverage(output: Path, report: dict):
    """Terminal model failures are measurements; missing/falsified slots are not."""
    directory = output / "memory"
    manifest = read_json(directory / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("压缩实验缺少冻结manifest")
    protocol = manifest.get("protocol", {})
    if (manifest.get("real_model_validation") is not True or report.get("real_model_validation") is not True
            or report.get("estimated_token_counts") is not False
            or manifest.get("environment", {}).get("token_count_estimated") is not False):
        raise ValueError("压缩实验必须为真实模型和非估算token")
    locked = {key: manifest.get(key) for key in ("environment", "protocol", "workloads_sha256")}
    if (manifest.get("fingerprint") != digest(canonical(locked))
            or report.get("fingerprint") != manifest.get("fingerprint")):
        raise ValueError("压缩manifest/summary冻结指纹不一致")
    path = directory / "workloads.jsonl"
    if not path.is_file() or digest(path.read_bytes()) != manifest.get("workloads_sha256"):
        raise ValueError("压缩冻结workloads缺失或hash不匹配")
    workloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["case_id"]: row for row in workloads}
    counts = {kind: sum(row.get("kind") == kind for row in workloads) for kind in ("normal", "overload", "local")}
    repeats = protocol.get("repeats", {})
    if (len(by_id) != 8 or len(workloads) != 8 or counts != {"normal": 4, "overload": 2, "local": 2}
            or protocol.get("quick") is not False or protocol.get("conditions") != ["off", "on"]
            or set(repeats) != set(by_id)
            or any(type(repeats[key]) is not int or repeats[key] != (1 if by_id[key]["kind"] == "overload" else 3)
                   for key in repeats)):
        raise ValueError("压缩正式重复/病例计划不匹配")
    expected = {f"{case_id}/{repeat}/{condition}": (case_id, repeat, condition)
                for case_id, count in repeats.items() for repeat in range(1, count + 1) for condition in ("off", "on")}
    semantic_cases = {key for key, row in by_id.items() if row.get("kind") == "normal" and row.get("semantic_check")}
    semantic_planned = sum(repeats[key] * 2 for key in semantic_cases)
    if (len(expected) != 40 or protocol.get("expected_jobs") != 40 or report.get("expected_jobs") != 40
            or report.get("completed_jobs") != 40 or semantic_planned != 12
            or protocol.get("semantic_answer_jobs") != 12
            or report.get("semantic_summary", {}).get("expected_answer_jobs") != 12):
        raise ValueError("压缩40条件/12计划追问元数据不匹配")
    database = directory / "ledger.sqlite"
    if not database.is_file():
        raise ValueError("压缩ledger缺失")
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=1) as connection:
        entries = [(key, json.loads(body)) for key, body in connection.execute("SELECT job,body FROM results")]
    if len(entries) != 40 or {key for key, _ in entries} != set(expected):
        raise ValueError("压缩ledger未覆盖40唯一计划条件")
    semantic_completed, failed, fallback = 0, 0, 0
    for key, row in entries:
        identity = row.get("case_id"), row.get("repeat"), row.get("condition")
        if (type(row.get("repeat")) is not int or identity != expected[key] or row.get("kind") != by_id[identity[0]]["kind"]
                or row.get("status") not in {"ok", "error"} or row.get("token_count_estimated") is not False):
            raise ValueError("压缩ledger含非terminal、错位或估算条件")
        failed += row["status"] == "error"
        fallback += bool(row.get("fallback"))
        if row.get("answer_status") == "completed":
            if row["case_id"] not in semantic_cases or row["status"] != "ok":
                raise ValueError("压缩ledger含计划外追问完成标记")
            semantic_completed += 1
    if report.get("semantic_summary", {}).get("completed_answer_jobs") != semantic_completed:
        raise ValueError("压缩追问实际完成数与ledger不一致")
    if not isinstance(report.get("complete"), bool):
        raise ValueError("压缩summary缺少明确complete状态")
    return {"measurement_failures": bool(failed or fallback or semantic_completed < 12 or not report["complete"]),
            "failed_conditions": failed, "fallback_conditions": fallback, "terminal_conditions": 40,
            "semantic_planned": 12, "semantic_completed": semantic_completed,
            "runner_report_complete": report["complete"]}


def unload_profile_models(profile: dict, root: Path = ROOT):
    """Only unload resident models on this profile's two explicitly used services."""
    original = dotenv_values(root / ".env")
    chat = profile.get("KB_OLLAMA_BASE_URL", original.get("KB_OLLAMA_BASE_URL")) or "http://127.0.0.1:11434"
    judge = profile.get("KB_EVAL_BASE_URL", original.get("KB_EVAL_BASE_URL")) or chat
    results = []
    for base in sorted({chat.rstrip("/"), judge.rstrip("/")}):
        row = {"service": base, "unloaded_models": [], "confirmed_empty": False}
        try:
            response = httpx.get(base + "/api/ps", timeout=10, trust_env=False)
            response.raise_for_status()
            for model in response.json().get("models", []):
                name = model.get("name") or model.get("model")
                if not isinstance(name, str) or not name:
                    raise ValueError("resident model没有有效名称，不能确认安全卸载")
                stopped = httpx.post(base + "/api/generate", json={"model": name, "keep_alive": 0},
                                     timeout=60, trust_env=False)
                stopped.raise_for_status()
                row["unloaded_models"].append(name)
            confirmed = httpx.get(base + "/api/ps", timeout=10, trust_env=False)
            confirmed.raise_for_status()
            row["confirmed_empty"] = confirmed.json().get("models") == []
            if not row["confirmed_empty"]:
                row["error"] = "卸载后仍有驻留模型，未确认资源释放"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
    return {"confirmed_empty": all(row["confirmed_empty"] for row in results), "services": results, "at": now()}


def validate_artifacts(name: str, output: Path, schedule: dict):
    report = read_json(output / ("memory/summary.json" if name == "memory" else "benchmark/summary.json"))
    if not isinstance(report, dict):
        raise ValueError("阶段没有提交可读取的summary.json")
    stable = report.get("stability", {})
    aggregate = stable.get("aggregate", {})
    if name == "prepare":
        if stable.get("ids") != schedule["ids"] or stable.get("repeats") != 3 \
                or aggregate.get("planned_generations") != 264:
            raise ValueError("prepare未锁定全部test88三轮计划")
    elif name == "memory":
        return validate_memory_coverage(output, report)
    elif name == "retrieval":
        modes = report.get("retrieval_ablation", {})
        if any(modes.get(mode, {}).get("attempted_count") != 88 for mode in ("dense", "hybrid", "rerank", "parent")):
            raise ValueError("四种检索消融352slot尚未完整提交")
    elif name == "generate":
        if aggregate.get("planned_generations") != 264 or not aggregate.get("generation_attempted_all"):
            raise ValueError("264个生成slot尚未全部提交")
    elif name == "prepare_vectors":
        vectors = report.get("vector_preparation", {})
        if not isinstance(vectors.get("successful_eligible_generations"), int) \
                or vectors.get("successful_eligible_generations") != vectors.get("logical_items_with_vector_files"):
            raise ValueError("仍有成功且符合评分条件的生成缺少向量")
    elif name == "score":
        scores = aggregate.get("metrics", {})
        if any(scores.get(metric, {}).get("eligible_count") != 231
               or scores.get(metric, {}).get("not_started_count") != 0 for metric in METRICS):
            raise ValueError("693个指标slot尚未全部提交（失败/blocked/null仍保留分母）")


def validate_context_recall_artifacts(output: Path, driver_sha256: str):
    selection = read_json(output / "benchmark/context_recall_selection.json")
    manifest = read_json(output / "benchmark/manifest.json", {})
    if (not isinstance(selection, dict) or selection.get("metric") != "context_recall"
            or selection.get("driver_sha256") != driver_sha256
            or selection.get("frozen_hash") != manifest.get("frozen_hash")
            or not isinstance(selection.get("items"), list)
            or selection.get("expected_count") != len(selection["items"])):
        raise ValueError("ContextRecall评分样本快照缺失或冻结标识不匹配")
    completed = set()
    for line in (output / "benchmark/journal.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        row = record.get("payload") or {}
        if record.get("state") == "completed" and row.get("phase") == "score" and row.get("metric") == "context_recall":
            completed.add((row.get("item"), row.get("generation_hash")))
    expected = {(row["item"], row["generation_hash"]) for row in selection["items"]}
    if len(expected) != len(selection["items"]) or not expected.issubset(completed):
        raise ValueError("已冻结的ContextRecall部分样本尚未全部提交；未生成项仍保留缺失")
    return {"metric": "context_recall", "selected_terminal_count": len(expected), "original_eligible_count": 231}


def write_report(output: Path):
    state = read_json(output / "pipeline_state.json", {})
    benchmark = read_json(output / "benchmark/summary.json", {})
    memory = read_json(output / "memory/summary.json", {})
    aggregate = benchmark.get("stability", {}).get("aggregate", {})
    lines = ["# 完整验证执行报告", "", f"执行状态：{state.get('status', 'not_started')}。质量目标达成判断：未作出。", "",
        "固定计划：test88全部题目独立三轮，共264个生成slot；四检索模式352slot；77题×3轮×3指标=693slot。",
        "压缩计划：正常4例各3次、局部2例各3次、超窗口2例各1次，on/off共40条件，12次最终追问。", "",
        "## 阶段", "", "| 阶段 | 状态 | exitcode |", "| --- | --- | --- |"]
    for stage in state.get("stages", []):
        lines.append(f"| {stage['name']} | {stage.get('status')} | {json.dumps(stage.get('exitcode'))} |")
    if state.get("user_controls", {}).get("scoring_disabled"):
        lines.extend(["", "回答评分及评分资料准备已按用户要求停用；已产生记录保留。",
                      "尚未执行的评分计为未评分，不计为成功或实际失败；本次不构成完整RAGAS评测。"])
    if state.get("user_controls", {}).get("score_metrics") == ["context_recall"]:
        selection = read_json(output / "benchmark/context_recall_selection.json", {})
        lines.extend(["", "知识问答生成已按用户要求停止；仅恢复ContextRecall，跳过其不需要的向量准备。",
                      f"评分使用停止时已提交的样本，共{selection.get('expected_count', '待冻结')}个符合指标条件的位置。",
                      "未生成项仍为缺失；其他指标未启用。原231个ContextRecall位置和693个总指标位置的分母保留。",
                      "有效项均值仅代表这批已生成样本，不是完整题集或三轮稳定性结果。"])
    lines.extend(["", "## 全部三轮评分", "", "均值仅基于有效评分；null、失败、blocked及未开始项保留各自计数和分母。", "",
                  "| 指标 | 有效/符合评分条件 | 失败或blocked | 未开始 | 有效项均值 |", "| --- | --- | --- | --- | --- |"])
    for metric in METRICS:
        row = aggregate.get("metrics", {}).get(metric, {})
        lines.append(f"| {metric} | {row.get('valid_count', 0)}/{row.get('eligible_count', 231)} | "
                     f"{row.get('failed_or_blocked_count', 0)} | {row.get('not_started_count', 231)} | "
                     f"{json.dumps(row.get('mean_valid_only'))} |")
    selected = {"generation_statuses": aggregate.get("statuses"),
                "repeat_mean_statistics": benchmark.get("stability", {}).get("repeat_mean_statistics"),
                "retrieval_ablation": benchmark.get("retrieval_ablation"),
                "memory_groups": memory.get("groups"), "memory_cases": memory.get("cases"),
                "memory_semantic_summary": memory.get("semantic_summary"),
                "memory_target_failures": memory.get("target_failures"), "memory_errors": memory.get("errors"),
                "warnings": state.get("warnings", []), "resource_cleanup": state.get("resource_cleanup"),
                "user_controls": state.get("user_controls"),
                "ui_restore": state.get("ui_restore")}
    lines.extend(["", "## 实测汇总字段", "", "```json", json.dumps(selected, ensure_ascii=False, indent=2), "```", "",
                  "normal、overload、local分别列示；没有把超窗口比率纳入正常0.55–0.70目标。",
                  "main的一轮同时属于全部三轮repeat0，计数使用stability.aggregate，未重复相加。", ""])
    text = "\n".join(lines)
    atomic_write(output / "report.md", text)
    return text


class Coordinator:
    def __init__(self, root: Path, output: Path, profile_path: Path, *, resume=False,
                 executor=None, status_factory=StatusPage, ui_restorer=None, model_unloader=None,
                 process_tree_factory=OwnedProcessTree, awake_guard=system_awake, launch_id=None,
                 disable_scoring=False, context_recall_only=False):
        self.root, self.output, self.profile_path = root.resolve(), output.resolve(), profile_path.resolve()
        self.resume, self.executor, self.status_factory, self.ui_restorer = resume, executor, status_factory, ui_restorer
        self.model_unloader = model_unloader or unload_profile_models
        self.process_tree_factory, self.awake_guard = process_tree_factory, awake_guard
        self.active_process = self.active_tree = None
        self.child_stop_unconfirmed, self.pending_warnings = False, []
        self.state = None
        self.launch_id = launch_id or uuid4().hex
        self.disable_scoring = disable_scoring
        self.context_recall_only = context_recall_only
        if disable_scoring and context_recall_only:
            raise ValueError("停用评分与仅恢复ContextRecall不能同时指定")
        self.progress = Progress(self.output)

    def save(self):
        self.state["updated_at"] = now()
        atomic_write(self.output / "pipeline_state.json", json.dumps(self.state, ensure_ascii=False, indent=2))
        self.progress.update(self.state)

    def warn(self, code, detail=""):
        self.state.setdefault("warnings", []).append({"at": now(), "code": code, "detail": detail})
        self.save()

    def execute(self, stage, command, environment):
        with Path(stage["log"]).open("ab", buffering=0) as log:
            log.write(("\n" + now() + " " + canonical(command) + "\n").encode("utf-8"))
            tree = self.process_tree_factory()
            try:
                process = subprocess.Popen(command, cwd=self.root, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, **owned_process_kwargs())
                self.active_process, self.active_tree = process, tree
                stage["pid"] = process.pid
                tree.attach_suspended(process)
                self.save()
                while process.poll() is None:
                    self.save()
                    time.sleep(2)
                return process.returncode
            finally:
                if self.active_process is None:
                    tree.close()
                elif not self.stop_active_child():
                    raise RuntimeError("当前任务子进程/worker树退出未确认，禁止下一阶段或启动UI")

    def stop_active_child(self):
        if self.child_stop_unconfirmed:
            return False
        if self.active_process is None:
            return True
        confirmed = False
        try:
            confirmed = self.active_tree.stop_and_confirm()
        except Exception:
            confirmed = False
        finally:
            self.active_tree.close()
        if confirmed:
            self.active_process = self.active_tree = None
        else:
            self.child_stop_unconfirmed = True
        return confirmed

    def restore_ui(self, python):
        if self.ui_restorer:
            self.state["ui_restore"] = self.ui_restorer(self)
            self.save()
            return
        if port_busy():
            self.state["ui_restore"] = {"status": "skipped_port_busy", "port": PORT}
            self.save()
            return
        original = dotenv_values(self.root / ".env")
        if str(original.get("KB_PORT") or "7860") != "7860" or (original.get("KB_PROVIDER") or "ollama") != "ollama":
            self.state["ui_restore"] = {"status": "skipped_original_env_not_real_ui_7860"}
            self.save()
            return
        command = [str(python), "-u", "-m", "agentic_kb", "serve"]
        path = self.output / "logs/ui.log"
        with path.open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=self.root, env=child_environment(root=self.root), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, **hidden_process_kwargs())
        self.state["ui_restore"] = {"status": "starting", "pid": process.pid, "command": command, "log": str(path)}
        self.save()
        for _ in range(60):
            if port_busy():
                self.state["ui_restore"]["status"] = "listening"
                break
            if process.poll() is not None:
                self.state["ui_restore"].update(status="exited", exitcode=process.returncode)
                break
            time.sleep(0.5)
        self.save()

    def run(self):
        def note(code, detail=""):
            if self.state is None:
                self.pending_warnings.append({"at": now(), "code": code, "detail": detail})
            else:
                self.warn(code, detail)

        with exclusive_lock(self.output / ".coordinator.lock"):
            with self.awake_guard(note):
                result = self._run()
            if self.state is not None:
                write_report(self.output)
            return result

    def _run(self):
        profile = load_profile(self.profile_path, self.root)
        python = project_python(self.root)
        schedule = dataset_schedule(self.root)
        stages = build_stages(self.root, self.output, python, schedule)
        frozen = frozen_definition(self.profile_path, profile, schedule, stages)
        fingerprint = digest(canonical(frozen))
        manifest_path = self.output / "pipeline_manifest.json"
        stored = read_json(manifest_path)
        controls = read_user_controls(self.output)
        if stored:
            if not self.resume:
                raise ValueError("已有流水线，使用--resume；不覆盖旧实验")
            if stored.get("fingerprint") != digest(canonical(stored.get("frozen"))):
                raise ValueError("原pipeline manifest冻结指纹损坏，拒绝resume")
            self.state = read_json(self.output / "pipeline_state.json")
            if not self.state:
                raise ValueError("流水线state丢失，拒绝猜测进度")
            if self.state.get("resource_cleanup", {}).get("child_processes_confirmed_stopped") is False:
                raise RuntimeError("前次worker树退出未确认，拒绝自动恢复新阶段或UI；保留原状态")
            for stage in self.state["stages"]:
                unknown_tree = self.state.get("resource_cleanup", {}).get("child_processes_confirmed_stopped") is False
                if (stage.get("status") == "running" or unknown_tree) and pid_alive(stage.get("pid")):
                    raise RuntimeError("原阶段子进程仍存活，拒绝并发恢复；不杀进程")
            if self.state.get("fingerprint") != stored["fingerprint"]:
                committed = next((change for change in controls["changes"]
                    if change.get("previous_fingerprint") == self.state.get("fingerprint")
                    and change.get("fingerprint") == stored["fingerprint"]), None)
                if committed is None:
                    raise ValueError("state与manifest冻结指纹不一致，拒绝resume")
                check_control_archives(self.output, committed)
        else:
            if self.resume:
                raise ValueError("无可恢复流水线")
            stored = {"fingerprint": fingerprint, "frozen": frozen, "created_at": now()}
            atomic_write(manifest_path, json.dumps(stored, ensure_ascii=False, indent=2))
            self.state = {"fingerprint": fingerprint, "status": "ready", "current_stage": None,
                          "coordinator_pid": os.getpid(), "created_at": now(), "warnings": [],
                          "stages": [{**item, "status": "pending", "pid": None, "exitcode": None,
                                      "attempts": [], "log": str(self.output / "logs" / (item["name"] + ".log"))}
                                     for item in stages]}
        drift = stored["fingerprint"] != fingerprint
        committed = next((change for change in controls["changes"]
            if change.get("previous_fingerprint") == stored["fingerprint"]
            and change.get("fingerprint") == fingerprint
            and change.get("new_coordinator_sha256") == frozen["coordinator_sha256"]), None)
        if drift:
            old_definition, new_definition = copy.deepcopy(stored["frozen"]), copy.deepcopy(frozen)
            old_definition.pop("coordinator_sha256", None)
            new_definition.pop("coordinator_sha256", None)
            if old_definition != new_definition or not (self.disable_scoring or self.context_recall_only or committed):
                raise ValueError("profile、coordinator自身hash或固定计划发生变化，拒绝resume")
            if committed:
                check_control_archives(self.output, committed)
        if self.disable_scoring and (not controls["scoring_disabled"] or (drift and committed is None)):
            controls = commit_disable_scoring(self.output, stored, manifest_path.read_bytes(),
                                              frozen, fingerprint, controls)
        if self.context_recall_only:
            driver = self.root / "scripts/score_context_recall.py"
            if not driver.is_file():
                raise ValueError("ContextRecall独立评分入口不存在")
            driver_sha256 = digest(driver.read_bytes())
            if controls.get("score_driver_sha256") and controls["score_driver_sha256"] != driver_sha256:
                raise ValueError("ContextRecall评分入口hash发生变化，拒绝覆盖已冻结选择")
            if (controls["scoring_disabled"] or controls.get("generation_disabled") is not True
                    or controls.get("score_metrics") != ["context_recall"]
                    or controls.get("score_driver_sha256") != driver_sha256 or (drift and committed is None)):
                controls = commit_disable_scoring(self.output, stored, manifest_path.read_bytes(),
                    frozen, fingerprint, controls, context_recall_only=True, driver_sha256=driver_sha256)
        if controls.get("score_metrics") == ["context_recall"]:
            driver = self.root / "scripts/score_context_recall.py"
            if not driver.is_file() or digest(driver.read_bytes()) != controls.get("score_driver_sha256"):
                raise ValueError("ContextRecall评分入口hash发生变化，拒绝恢复")
        if drift:
            stored = {**stored, "fingerprint": fingerprint, "frozen": frozen, "updated_at": now()}
            atomic_write(manifest_path, json.dumps(stored, ensure_ascii=False, indent=2))
        sync_control_log(self.output, controls)
        self.state["fingerprint"] = fingerprint
        self.state["user_controls"] = {"scoring_disabled": controls["scoring_disabled"],
                                       "change_ids": [change["id"] for change in controls["changes"]]}
        for key in ("generation_disabled", "score_metrics", "score_driver_sha256"):
            if key in controls:
                self.state["user_controls"][key] = controls[key]
        for stage in self.state["stages"]:
            selected_only = controls.get("score_metrics") == ["context_recall"]
            should_skip = (stage["name"] == "generate" and controls.get("generation_disabled")) or (
                stage["name"] == "prepare_vectors" and selected_only)
            if should_skip and stage["status"] != "completed":
                if stage["status"] == "running" and stage["attempts"]:
                    stage["attempts"][-1].update(status="stopped_by_user", finished_at=now(), pid=stage.get("pid"))
                stage.update(status="skipped_by_user", pid=None, skipped_at=now(), skip_reason="explicit_user_request")
            if (stage["name"] == "score" and selected_only and not controls["scoring_disabled"]
                    and stage["status"] == "skipped_by_user"):
                stage.update(status="pending", pid=None)
                stage.pop("skipped_at", None)
                stage.pop("skip_reason", None)
                self.state["status"] = "ready"
        if controls["scoring_disabled"]:
            for stage in self.state["stages"]:
                if stage["name"] in SCORING_STAGES and stage["status"] not in {"completed", "skipped_by_user"}:
                    stage.update(status="skipped_by_user", skipped_at=now(), skip_reason="explicit_user_request")
        self.state["coordinator_pid"] = os.getpid()
        self.state["launch_id"] = self.launch_id
        self.state["warnings"].extend(self.pending_warnings)
        (self.output / "logs").mkdir(exist_ok=True)
        if self.state["status"] == "completed":
            self.save()
            write_report(self.output)
            return 0
        self.state["status"] = "running"
        self.save()
        server = self.status_factory(self.progress)
        result_code = 0
        try:
            try:
                server.start()
                self.state["status_page"] = {"status": "listening", "host": "127.0.0.1", "port": PORT}
                self.save()
            except OSError as exc:
                self.warn("status_port_busy_or_unavailable", str(exc))
            for stage in self.state["stages"]:
                if stage["status"] in {"completed", "skipped_by_user"}:
                    continue
                command = list(stage["command"])
                if stage["name"] == "score" and controls.get("score_metrics") == ["context_recall"]:
                    # The original frozen command/benchmark definition remains intact.
                    # This explicit user override is recorded in controls and actual_command.
                    command[2:4] = [str(self.root / "scripts/score_context_recall.py")]
                if stage["name"] == "memory" and (self.output / "memory/manifest.json").exists():
                    command.append("--resume")
                self.state["current_stage"] = stage["name"]
                stage.pop("error", None)  # Previous failure remains in attempts; current status describes this attempt.
                stage.update(status="running", pid=None, exitcode=None, started_at=now(), actual_command=command)
                stage["attempts"].append({"started_at": stage["started_at"], "command": command})
                self.save()
                try:
                    code = (self.executor(stage, command, child_environment(profile, self.root), self)
                            if self.executor else self.execute(stage, command, child_environment(profile, self.root)))
                    stage["exitcode"] = code
                    if code != 0 and not (stage["name"] == "memory" and code == 2):
                        raise RuntimeError(f"子进程exitcode={code}")
                    if stage["name"] == "score" and controls.get("score_metrics") == ["context_recall"]:
                        measurement = validate_context_recall_artifacts(self.output, controls["score_driver_sha256"])
                        stage["measurement_summary"] = measurement
                    else:
                        measurement = validate_artifacts(stage["name"], self.output, schedule)
                    if stage["name"] == "memory":
                        expected_code = 0 if measurement["runner_report_complete"] else 2
                        if code != expected_code or (code == 2 and not measurement["measurement_failures"]):
                            raise ValueError("压缩进程退出码与已提交测量状态不一致")
                        stage["measurement_failures"] = measurement["measurement_failures"]
                        stage["measurement_summary"] = measurement
                        if measurement["measurement_failures"]:
                            self.warn("memory_complete_with_measurement_failures", canonical(measurement))
                except Exception as exc:
                    stage.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=now())
                    stage["attempts"][-1].update(exitcode=stage.get("exitcode"), finished_at=stage["finished_at"],
                                                  error=stage["error"])
                    self.state["status"] = "failed"
                    self.save()
                    result_code = 2
                    break
                stage.update(status="completed", finished_at=now())
                stage["attempts"][-1].update(exitcode=code, finished_at=stage["finished_at"], pid=stage.get("pid"))
                self.save()
                write_report(self.output)
            else:
                self.state.update(status="completed", current_stage=None, finished_at=now())
            self.save()
        except KeyboardInterrupt:
            self.state.update(status="interrupted", finished_at=now())
            self.save()
            result_code = 130
        except Exception as exc:
            self.state.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=now())
            self.save()
            result_code = 2
        finally:
            self.state["resource_cleanup"] = {"status": "checking", "confirmed_empty": False}
            self.save()
            child_stopped = self.stop_active_child()
            if child_stopped:
                try:
                    cleanup = self.model_unloader(profile, self.root)
                except Exception as exc:
                    cleanup = {"confirmed_empty": False, "error": f"{type(exc).__name__}: {exc}"}
            else:
                cleanup = {"confirmed_empty": False, "error": "owned_child_tree_stop_not_confirmed"}
            cleanup["child_processes_confirmed_stopped"] = child_stopped
            cleanup["status"] = "confirmed_empty" if cleanup.get("confirmed_empty") else "unconfirmed"
            self.state["resource_cleanup"] = cleanup
            self.save()
            server.close()
            self.state["status_page"] = {"status": "stopped", "port": PORT}
            self.save()
            if cleanup.get("confirmed_empty"):
                try:
                    self.restore_ui(python)
                except Exception as exc:
                    self.state["ui_restore"] = {"status": "failed", "error": str(exc)}
                    self.warn("ui_restore_failed", str(exc))
            else:
                self.state["ui_restore"] = {"status": "not_started_unload_unconfirmed"}
                if result_code == 0:
                    self.state["status"] = "completed_cleanup_failed"
                    result_code = 2
                self.warn("resource_unload_unconfirmed_ui_not_started")
            write_report(self.output)
        return result_code


def launch(args):
    output = args.output.resolve()
    with exclusive_lock(output / ".launch.lock"):
        with exclusive_lock(output / ".coordinator.lock"):
            pass
        profile = load_profile(args.profile.resolve(), ROOT)
        del profile  # Validation only; launcher never applies benchmark overrides to itself.
        launch_id = uuid4().hex
        command = [str(project_python(ROOT)), "-u", str(Path(__file__).resolve()),
                   "--profile", str(args.profile.resolve()), "--output", str(output), "--launch-id", launch_id]
        if args.resume:
            command.append("--resume")
        if getattr(args, "disable_scoring", False):
            command.append("--disable-scoring")
        if getattr(args, "context_recall_only", False):
            command.append("--context-recall-only")
        (output / "logs").mkdir(exist_ok=True)
        with (output / "logs/coordinator.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=child_environment(root=ROOT), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, **hidden_process_kwargs())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = read_json(output / "pipeline_state.json", {})
            actual_pid = state.get("coordinator_pid")
            if state.get("launch_id") == launch_id and type(actual_pid) is int and actual_pid > 0:
                print(canonical({"coordinator_pid": actual_pid, "launcher_pid": process.pid,
                                 "status": state.get("status"), "output": str(output)}))
                return 0
            if process.poll() is not None:
                raise RuntimeError(f"后台coordinator启动失败，exitcode={process.returncode}；见独立coordinator.log")
            time.sleep(0.1)
        print(canonical({"coordinator_pid": None, "launcher_pid": process.pid,
                         "status": "starting", "output": str(output)}))
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", action="store_true")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--disable-scoring", action="store_true", help="停用回答评分和评分资料准备，并持久保存此选择")
    controls.add_argument("--context-recall-only", action="store_true", help="停止生成，仅对已提交样本评分ContextRecall，持久保存选择")
    parser.add_argument("--launch-id", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.report:
        print(write_report(args.output.resolve()))
        return 0
    if not args.profile:
        parser.error("执行或launch需要--profile；--report无需profile")
    if args.launch:
        return launch(args)
    return Coordinator(ROOT, args.output, args.profile, resume=args.resume,
                       launch_id=args.launch_id, disable_scoring=args.disable_scoring,
                       context_recall_only=args.context_recall_only).run()


if __name__ == "__main__":
    raise SystemExit(main())
