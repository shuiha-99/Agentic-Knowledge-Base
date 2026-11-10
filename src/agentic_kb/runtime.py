"""Start an explicitly configured local Ollama without changing user settings."""
from __future__ import annotations

import json
import os
import subprocess
import time

import httpx

from .config import Settings


def ollama_status(settings: Settings) -> dict:
    try:
        response = httpx.get(settings.ollama_base_url + "/api/tags", timeout=3, trust_env=False)
        response.raise_for_status()
        return {"available": True, "models": [m["name"] for m in response.json().get("models", [])]}
    except (httpx.HTTPError, ValueError) as exc:
        return {"available": False, "models": [], "error": str(exc)}


def start_ollama(settings: Settings) -> dict:
    status = ollama_status(settings)
    if status["available"]:
        return {**status, "started": False}
    executable = settings.ollama_executable
    if executable is None or not executable.is_file():
        raise RuntimeError("请在 .env 中配置 KB_OLLAMA_EXECUTABLE，或自行启动 Ollama 服务。")
    service_dir = settings.data_dir / "ollama"
    service_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = service_dir / "profile"
    profile_dir.mkdir(exist_ok=True)
    environment = os.environ.copy()
    # Go's os.UserHomeDir on Windows uses USERPROFILE. This change is restricted
    # to this child process so its generated key stays inside this project.
    environment["USERPROFILE"] = str(profile_dir)
    environment["OLLAMA_HOST"] = settings.ollama_base_url.removeprefix("http://").removeprefix("https://")
    environment["OLLAMA_NOPRUNE"] = "true"
    environment["OLLAMA_NO_CLOUD"] = "true"
    environment["OLLAMA_NUM_PARALLEL"] = str(settings.max_parallel_tasks)
    environment["OLLAMA_MAX_LOADED_MODELS"] = "1"
    if settings.ollama_models_dir:
        environment["OLLAMA_MODELS"] = str(settings.ollama_models_dir.resolve())
    else:
        environment["OLLAMA_MODELS"] = str(service_dir / "models")
    with (service_dir / "stdout.log").open("ab") as stdout, (service_dir / "stderr.log").open("ab") as stderr:
        process = subprocess.Popen(
            [str(executable), "serve"], cwd=service_dir, env=environment,
            stdout=stdout, stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    (service_dir / "process.json").write_text(json.dumps({"pid": process.pid, "executable": str(executable)}), encoding="utf-8")
    for _ in range(30):
        status = ollama_status(settings)
        if status["available"]:
            return {**status, "started": True, "pid": process.pid}
        if process.poll() is not None:
            raise RuntimeError(f"Ollama 启动失败，请查看 {service_dir / 'stderr.log'}")
        time.sleep(0.3)
    raise RuntimeError(f"Ollama 尚未就绪，请查看 {service_dir / 'stderr.log'}")

