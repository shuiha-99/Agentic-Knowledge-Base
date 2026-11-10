"""Frozen, resumable generation / per-metric scoring / retrieval benchmark.

Only explicit generate/score/all/retrieval phases invoke real models. `prepare`
reads metadata; `report` prints the existing report without rerunning anything.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib.metadata
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx

# Freeze and execute this checkout, independently of an older installed wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_kb.config import Settings  # noqa: E402
from agentic_kb.evaluation import BenchmarkRun, _RagasScorer, _safe_settings, _versions, load_dataset  # noqa: E402


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def exclusive_run(directory: Path):
    """OS locks are released after a crash; stale PID files cannot block resume."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as stream:
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
            raise RuntimeError("另一个进程正在执行此评测目录，拒绝并发计分") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def environment_snapshot(settings: Settings, root: Path, dataset: Path, *, score_keep_alive="10m", metric_timeout=900) -> dict:
    """Freeze installed versions, model revisions and active corpus, not names only."""
    source_paths = [*sorted((root / "src" / "agentic_kb").glob("*.py")),
                    root / "scripts" / "benchmark.py", root / "scripts" / "ragas_worker.py",
                    root / "pyproject.toml", root / "uv.lock", root / "docs" / "requirements-ragas.lock.txt"]
    source = {str(path.relative_to(root)): file_sha(path) for path in source_paths if path.is_file()}
    worker_path = Path(os.getenv("KB_RAGAS_WORKER_PATH", os.getenv("KB_RAGAS_WORKER", str(root / "scripts" / "ragas_worker.py")))).resolve()
    if not worker_path.is_file():
        raise ValueError(f"实际选用的RAGAS worker文件不存在，拒绝冻结：{worker_path}")
    worker_identity = {"path": str(worker_path), "sha256": file_sha(worker_path)}
    python = Path(os.getenv("KB_RAGAS_PYTHON", str(root / ".venv-ragas" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))))
    code = "import json,importlib.metadata as m; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions() if d.metadata['Name']}))"
    response = subprocess.run([str(python), "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
    worker_versions = json.loads(response.stdout)
    if worker_versions.get("ragas") != "0.3.3":
        raise ValueError("评测worker必须使用ragas==0.3.3")
    judge_model = os.getenv("KB_EVAL_MODEL", settings.chat_model)
    judge_base_url = os.getenv("KB_EVAL_BASE_URL", settings.ollama_base_url)
    structured_model = getattr(settings, "structured_model", None)
    structured_tokenizer = getattr(settings, "structured_tokenizer_model", None)
    if structured_model and not structured_tokenizer:
        raise ValueError("启用结构化辅助模型时必须冻结其structured_tokenizer_model")
    model_roles = {"chat": (settings.ollama_base_url, settings.chat_model),
                   "judge": (judge_base_url, judge_model)}
    if structured_model:
        model_roles["structured"] = (settings.ollama_base_url, structured_model)
    servers = {}
    for base_url in dict.fromkeys(base.rstrip("/") for base, _ in model_roles.values()):
        response = httpx.get(base_url + "/api/version", timeout=10, trust_env=False)
        response.raise_for_status()
        version = response.json().get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"无法冻结Ollama服务实际版本：{base_url}")
        servers[base_url] = {"version": version}
    models = {}
    for base_url, name in dict.fromkeys(model_roles.values()):
        response = httpx.get(base_url.rstrip("/") + "/api/tags", timeout=10, trust_env=False)
        response.raise_for_status()
        tag = name if ":" in name else name + ":latest"
        match = next((model for model in response.json()["models"] if model["name"] == tag), None)
        if not match or not match.get("digest"):
            raise ValueError(f"无法冻结Ollama模型实际digest：{name}")
        models[base_url + "/" + name] = {key: match[key] for key in ("name", "digest", "size")}
    hf = {}
    hf_names = [settings.embedding_model, settings.reranker_model, settings.tokenizer_model]
    if structured_tokenizer:
        hf_names.append(structured_tokenizer)
    for model in dict.fromkeys(hf_names):
        cache = settings.models_dir / ("models--" + model.replace("/", "--"))
        snapshots = cache / "snapshots"
        refs = cache / "refs"
        if not snapshots.is_dir() or not any(snapshots.iterdir()):
            raise ValueError(f"模型缓存尚未准备，不能仅冻结名称：{model}")
        hf[model] = {"snapshot_revisions": sorted(path.name for path in snapshots.iterdir() if path.is_dir()),
                     "refs": {str(path.relative_to(refs)): path.read_text(encoding="utf-8").strip() for path in sorted(refs.rglob("*")) if path.is_file()} if refs.exists() else {},
                     "configuration_sha256": {str(path.relative_to(cache)): file_sha(path) for path in sorted(snapshots.rglob("*.json")) if path.is_file()}}
    database = settings.data_dir / "documents.sqlite"
    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as connection:
        documents = [json.loads(row[0]) for row in connection.execute("SELECT data FROM documents ORDER BY doc_id")]
    if not documents:
        raise ValueError("知识库尚未建库，拒绝冻结空语料")
    for document in documents:
        if file_sha(Path(document["path"])) != document["sha256"]:
            raise ValueError(f"活动文档内容与索引元数据不一致：{document['name']}")
    return {"dataset_path": str(dataset.resolve()), "dataset_sha256": file_sha(dataset),
            "settings": _safe_settings(settings), "source_sha256": source, "application_versions": _versions(),
            "all_application_distributions": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]},
            "documents": documents, "ollama_models": models, "hf_models": hf, "ollama_servers": servers,
            "ollama_model_roles": {role: base_url + "/" + name for role, (base_url, name) in model_roles.items()},
            "structured_routing": {"helper_model": structured_model, "helper_tokenizer": structured_tokenizer,
                                   "schema_allowlist": [name.strip() for name in settings.structured_model_schemas.split(",") if name.strip()],
                                   "empty_allowlist_routes_all_structured": True},
            "vector_execution": {"retain_embedding_model_between_items": True,
                                 "unload_ollama_before_encoding": True, "release_embedding_before_judge": True,
                                 "retrieval_phase_retains_only_dense_and_reranker": True},
            "retrieval_ablation_protocol": {"requested_k": 5, "metric_document_k": 5,
                "label_to_search_mode": {"dense": "dense", "hybrid": "hybrid", "rerank": "rerank", "parent": "full"}},
            "judge": {"model": judge_model, "base_url": judge_base_url, "temperature": 0,
                      "answer_correctness_weights": [0.75, 0.25], "worker_versions": worker_versions,
                      "worker_python": str(python.resolve()), "worker": worker_identity,
                      "output_tokens": max(2048, settings.output_reserve_tokens),
                      "score_keep_alive": score_keep_alive, "metric_timeout_seconds": metric_timeout}}


def unload_ollama_before_vectors(settings):
    """Only unload; no prompt and no generation. BGE must not overlap a judge."""
    for base_url in {settings.ollama_base_url, os.getenv("KB_EVAL_BASE_URL", settings.ollama_base_url)}:
        response = httpx.get(base_url.rstrip("/") + "/api/ps", timeout=10, trust_env=False)
        response.raise_for_status()
        for model in response.json().get("models", []):
            stopped = httpx.post(base_url.rstrip("/") + "/api/generate", json={"model": model["name"], "keep_alive": 0},
                                 timeout=60, trust_env=False)
            stopped.raise_for_status()


def make_scorer(settings, embedder, args):
    scorer = _RagasScorer(settings, SimpleNamespace(retriever=SimpleNamespace(embedder=embedder)),
                          score_keep_alive=args.score_keep_alive)
    scorer.config["metric_timeout_seconds"] = args.metric_timeout
    return scorer


async def execute(args):
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        raise ValueError("请从项目根目录运行，避免worker或模型路径发生歧义")
    directory = args.run_dir.resolve()
    if args.phase == "report":
        print((directory / "summary.json").read_text(encoding="utf-8"))
        return
    settings = Settings.from_env()
    if settings.provider != "ollama":
        raise ValueError("正式benchmark拒绝offline-test；协议测试使用注入的测试对象")
    cases = load_dataset(args.dataset, args.split)
    environment = environment_snapshot(settings, root, args.dataset, score_keep_alive=args.score_keep_alive,
                                       metric_timeout=args.metric_timeout)
    stability_ids = args.stability_ids.split(",") if args.stability_ids else None
    with exclusive_run(directory):
        run = BenchmarkRun(directory, cases, environment, mode=args.mode,
                           stability_repeats=args.stability_repeats, stability_ids=stability_ids)
        run.progress = lambda event: print(json.dumps(event, ensure_ascii=False), flush=True)
        if args.phase == "prepare":
            print(json.dumps({"run_id": run.manifest["run_id"], "frozen_hash": run.manifest["frozen_hash"],
                              "case_count": len(cases), "stability_ids": run.stability_ids}, ensure_ascii=False, indent=2))
            run.write_report()
            return
        llm = embedder = reranker = None
        if args.phase != "score":
            from agentic_kb.providers import build_providers
            llm, embedder, reranker = build_providers(settings)
        knowledge_base = engine = None
        try:
            if args.phase in {"generate", "all", "retrieval"}:
                from agentic_kb.graph import GraphEngine
                from agentic_kb.retrieval import KnowledgeBase
                knowledge_base = KnowledgeBase(settings, embedder, reranker, token_counter=llm.count_tokens)
                engine = GraphEngine(settings, llm, knowledge_base)
            if args.phase in {"generate", "all"}:
                await run.generate(engine, scope=args.scope, retry_failed=args.retry_failed, limit=args.limit)
            if args.phase in {"prepare-vectors", "all"}:
                unload_ollama_before_vectors(settings)
                previous_low_memory = settings.low_memory_mode
                settings.low_memory_mode = False  # Only BGE is loaded in this isolated phase.
                try:
                    await run.prepare_vectors(make_scorer(settings, embedder, args), scope=args.scope, limit=args.limit)
                finally:
                    settings.low_memory_mode = previous_low_memory
                    for adapter in (embedder, reranker):
                        if adapter is not None and hasattr(adapter, "_model"):
                            adapter._model = None
                    gc.collect()
            if args.phase in {"score", "all"}:
                scorer = make_scorer(settings, None, args)
                await run.score(scorer, scope=args.scope, retry_failed=args.retry_failed, limit=args.limit)
            if args.phase == "retrieval":
                unload_ollama_before_vectors(settings)
                previous_low_memory = settings.low_memory_mode
                settings.low_memory_mode = False
                try:
                    await run.retrieval_ablation(knowledge_base, limit=args.limit)
                finally:
                    settings.low_memory_mode = previous_low_memory
                    for adapter in (embedder, reranker):
                        if adapter is not None and hasattr(adapter, "_model"):
                            adapter._model = None
                    gc.collect()
            print(json.dumps(run.write_report(), ensure_ascii=False, indent=2))
        finally:
            if engine:
                await engine.close()
            if knowledge_base:
                knowledge_base.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="冻结数据与版本，断点执行完整生成、逐指标评分和固定范围稳定性（默认小集，可指定全量）")
    parser.add_argument("phase", choices=["prepare", "generate", "prepare-vectors", "score", "all", "retrieval", "report"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/demo_cases.jsonl"))
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--mode", default="agent", choices=["agent", "dense", "hybrid", "rerank", "parent"])
    parser.add_argument("--scope", default="all", choices=["all", "main", "stability"])
    parser.add_argument("--stability-repeats", type=int, default=3)
    parser.add_argument("--stability-ids", help="冻结前指定，逗号分隔；传所选split全部ID即全量重复，不允许事后改选")
    parser.add_argument("--retry-failed", action="store_true", help="显式增加失败项attempt，旧结果仍保留")
    parser.add_argument("--limit", type=int, help="本次最多处理多少缺失case/metric，仅控制调度，不改变冻结样本")
    parser.add_argument("--metric-timeout", type=float, default=900, help="独立judge指标超时；各阶段保持一致")
    parser.add_argument("--score-keep-alive", default="10m", help="独立judge驻留；评分阶段不再加载BGE，各阶段冻结相同值")
    asyncio.run(execute(parser.parse_args()))
