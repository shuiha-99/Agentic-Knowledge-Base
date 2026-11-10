"""Reproducible JSONL evaluation; workflow checks are not model-quality scores."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from .config import Settings
    from .graph import GraphEngine


@dataclass
class EvaluationCase:
    id: str
    question: str
    family: str = ""
    split: str = "test"
    kind: str = "answerable"
    reference: str = ""
    reference_documents: list[str] = field(default_factory=list)
    reference_doc_ids: list[str] = field(default_factory=list)
    reference_quotes: dict[str, str] = field(default_factory=dict)
    clarification: str | None = None
    turns: list[dict[str, str]] = field(default_factory=list)
    expected_contains: list[str] = field(default_factory=list)
    review_status: str = "needs_review"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvaluationCase:
        data = {key: value for key, value in raw.items() if key in cls.__dataclass_fields__}
        if not isinstance(data.get("id"), str) or not data["id"].strip():
            raise ValueError("每条评测题必须包含非空字符串 id")
        if not isinstance(data.get("question"), str) or not data["question"].strip():
            raise ValueError(f"题目 {data['id']} 缺少 question")
        case = cls(**data)
        case.family = case.family or case.id
        if case.split not in {"dev", "test", "train"}:
            raise ValueError(f"题目 {case.id} 的 split 必须为 train/dev/test")
        for name in ("reference_documents", "reference_doc_ids", "expected_contains"):
            values = getattr(case, name)
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError(f"题目 {case.id} 的 {name} 必须为字符串列表")
        if not isinstance(case.turns, list) or any(not isinstance(t, dict) or not isinstance(t.get("question"), str) for t in case.turns):
            raise ValueError(f"题目 {case.id} 的 turns 必须为包含 question 的对象列表")
        return case


def load_dataset(path: Path, split: str | None = None) -> list[EvaluationCase]:
    cases = []
    seen: set[str] = set()
    family_splits: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("每行必须为JSON对象")
            case = EvaluationCase.from_dict(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path.name}:{line_number}: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"重复题目 ID：{case.id}")
        seen.add(case.id)
        prior_split = family_splits.setdefault(case.family, case.split)
        if prior_split != case.split:
            raise ValueError(f"问题族 {case.family} 同时出现在 {prior_split}/{case.split}，可能泄漏")
        if split is None or case.split == split:
            cases.append(case)
    if not cases:
        raise ValueError("评测集为空，或所选划分没有题目")
    return cases


def family_split(family: str) -> str:
    """Assign complete problem families, never near-duplicate questions, to a split."""
    return "dev" if int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 4 == 0 else "test"


def _safe_settings(settings) -> dict[str, Any]:
    data = settings.model_dump(mode="json") if hasattr(settings, "model_dump") else vars(settings)
    return {k: "[REDACTED]" if any(secret in k.lower() for secret in ("password", "api_key", "secret")) else str(v) if isinstance(v, Path) else v for k, v in data.items()}


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("agentic-kb", "langgraph", "langchain-core", "langchain-ollama", "qdrant-client", "ragas", "gradio", "FlagEmbedding"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _source_fingerprints() -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob("*.py")}


def retrieval_metrics(case: EvaluationCase, sources: list[dict[str, Any]], k: int = 5) -> dict[str, float | None]:
    by_id = bool(case.reference_doc_ids)
    relevant = set(case.reference_doc_ids if by_id else case.reference_documents)
    if not relevant:
        return {f"document_recall_at_{k}": None, "document_mrr": None}
    ranked = []
    for source in sources:
        key = source.get("doc_id", "") if by_id else source.get("document_name", "")
        if key and key not in ranked:
            ranked.append(key)
    recall = len(relevant.intersection(ranked[:k])) / len(relevant)
    rank = next((i for i, key in enumerate(ranked, 1) if key in relevant), None)
    return {f"document_recall_at_{k}": recall, "document_mrr": 1.0 / rank if rank else 0.0}


def _finite_number(value) -> float | None:
    if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


class _RagasScorer:
    """A persistent process isolates Ragas 0.3 / LangChain 0.3 from the app."""

    def __init__(self, settings, engine, *, score_keep_alive=None):
        project_root = Path.cwd()
        self.worker_path = Path(os.getenv("KB_RAGAS_WORKER_PATH", os.getenv("KB_RAGAS_WORKER", str(project_root / "scripts" / "ragas_worker.py")))).resolve()
        default_python = project_root / ".venv-ragas" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.python = Path(os.getenv("KB_RAGAS_PYTHON", str(default_python))).resolve()
        if not self.python.is_file() or not self.worker_path.is_file():
            raise RuntimeError("RAGAS 需要独立 .venv-ragas 环境；按 docs/EVALUATION.md 安装 docs/requirements-ragas.lock.txt，或设置 KB_RAGAS_PYTHON 与 KB_RAGAS_WORKER")
        self.embedder = getattr(getattr(engine, "retriever", None), "embedder", None)
        self.judge_model = os.getenv("KB_EVAL_MODEL", settings.chat_model)
        self.judge_base_url = os.getenv("KB_EVAL_BASE_URL", settings.ollama_base_url)
        self.config = {"model": self.judge_model, "base_url": self.judge_base_url, "context_window_tokens": settings.context_window_tokens, "output_tokens": max(2048, settings.output_reserve_tokens), "request_timeout_seconds": settings.request_timeout_seconds, "metric_timeout_seconds": settings.run_timeout_seconds, "low_memory_mode": getattr(settings, "low_memory_mode", True)}
        if score_keep_alive is not None:
            self.config["score_keep_alive"] = score_keep_alive
        self.process = None
        self.stderr_task = None
        self.stderr_tail = ""
        self.versions = {}
        self._embedding_cache = {}

    async def _drain_stderr(self):
        while self.process and self.process.stderr:
            data = await self.process.stderr.read(4096)
            if not data:
                return
            self.stderr_tail = (self.stderr_tail + data.decode("utf-8", errors="replace"))[-12000:]

    async def _ensure_process(self):
        if self.process is not None and self.process.returncode is None:
            return
        env = dict(os.environ, PYTHONUTF8="1", RAGAS_DO_NOT_TRACK="true")
        self.process = await asyncio.create_subprocess_exec(str(self.python), str(self.worker_path), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env, limit=4 * 1024 * 1024)
        self.stderr_task = asyncio.create_task(self._drain_stderr())

    async def prepare_embeddings(self, reference: str, answer: str) -> dict[str, list[float]]:
        if self.embedder is None:
            raise RuntimeError("此评分进程不加载向量模型，必须先prepare-vectors并传入预计算向量")
        texts = [reference or " ", answer or " "]
        key = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
        if key not in self._embedding_cache:
            vectors = await asyncio.to_thread(self.embedder.encode, texts)
            self._embedding_cache[key] = {text: vector.dense for text, vector in zip(texts, vectors, strict=True)}
        return self._embedding_cache[key]

    async def score(self, case: EvaluationCase, answer: str, contexts: list[str], *,
                    metric_names: list[str] | None = None,
                    embeddings: dict[str, list[float]] | None = None) -> tuple[dict[str, float | None], dict[str, str]]:
        requested = metric_names or ["answer_correctness", "context_recall", "faithfulness"]
        if any(name not in {"answer_correctness", "context_recall", "faithfulness"} for name in requested):
            raise ValueError("Unsupported RAGAS metric")
        try:
            vectors = embeddings if embeddings is not None else await self.prepare_embeddings(case.reference, answer)
            request = {"config": self.config, "metrics": requested, "sample": {"user_input": case.question, "response": answer, "reference": case.reference, "retrieved_contexts": contexts}, "embeddings": vectors}
            await self._ensure_process()
            async with asyncio.timeout(self.config["metric_timeout_seconds"] * len(requested) + 60):
                self.process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
                await self.process.stdin.drain()
                line = await self.process.stdout.readline()
                if not line:
                    raise RuntimeError("RAGAS worker 意外退出：" + self.stderr_tail[-3000:])
                response = json.loads(line)
            self.versions = response.get("versions", {})
            scores = {name: _finite_number(response.get("scores", {}).get(name)) for name in requested}
            errors = response.get("errors", {})
            for name in requested:
                if scores[name] is None and name not in errors:
                    errors[name] = "worker未返回该指标的有效有限数值"
            return scores, errors
        except asyncio.CancelledError:
            await self.close()
            raise
        except Exception as exc:
            await self.close()
            return {name: None for name in requested}, {"worker": str(exc)}

    async def close(self):
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2)
            except TimeoutError:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=2)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            self.process = None
        if self.stderr_task:
            await self.stderr_task
            self.stderr_task = None


async def _run_turn(engine, question, thread_id, mode, clarification=None):
    events = []
    sources, metrics = [], {}
    answer, error, pending = "", None, False
    token_text = ""
    for attempt in range(2):
        needs_resume = False
        async for event in engine.astream(question if attempt == 0 else None, thread_id, clarification=clarification if attempt else None, mode=mode):
            events.append(event)
            kind = event.get("type")
            if kind == "token":
                token_text += event.get("text", "")
            elif kind == "sources":
                sources = event.get("sources", [])
            elif kind == "done":
                answer = event.get("answer") or token_text
                metrics = event.get("metrics", {})
            elif kind == "error":
                error = event.get("text", "执行失败")
            elif kind == "clarification":
                pending = True
                needs_resume = bool(clarification) and attempt == 0
        if needs_resume:
            pending = False
            continue
        break
    status = "error" if error else "clarification_required" if pending else "success" if answer else "incomplete"
    return {"status": status, "answer": answer, "sources": sources, "metrics": metrics, "error": error, "events": events}


async def evaluate_dataset(settings: Settings, engine: GraphEngine, dataset_path: Path, output_dir: Path, *, mode: str = "agent", use_ragas: bool = False, split: str | None = "test") -> dict:
    if use_ragas and settings.provider == "offline-test":
        raise ValueError("离线测试替身只能验证流程，禁止生成 RAGAS 质量成绩")
    scorer = _RagasScorer(settings, engine) if use_ragas else None
    try:
        return await _evaluate_dataset(settings, engine, dataset_path, output_dir, mode=mode, use_ragas=use_ragas, split=split, scorer=scorer)
    finally:
        if scorer:
            await scorer.close()


async def _evaluate_dataset(settings, engine, dataset_path, output_dir, *, mode, use_ragas, split, scorer):
    """Run all selected cases. Failures remain rows and are counted in the report."""
    cases = load_dataset(dataset_path, split)
    if use_ragas and settings.provider == "offline-test":
        raise ValueError("离线测试替身只能验证流程，禁止生成 RAGAS 质量成绩")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.jsonl"
    if result_path.exists():
        raise ValueError("输出目录已有 results.jsonl，请使用新的目录以保留实验记录")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode, "split": split, "case_count": len(cases),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "settings": _safe_settings(settings), "versions": _versions(),
        "source_sha256": _source_fingerprints(),
        "quality_evaluation": settings.provider != "offline-test",
        "ragas_enabled": use_ragas,
        "judge": {"model": scorer.judge_model, "base_url": scorer.judge_base_url, "temperature": 0, "answer_correctness_weights": [0.75, 0.25]} if scorer else None,
        "dataset_review_status": sorted({case.review_status for case in cases}),
        "notice": "演示或待复核题集不能被描述为独立真实企业评测；离线测试模式不代表大模型质量。",
    }
    retriever = getattr(engine, "retriever", None)
    if retriever is not None and hasattr(retriever, "list_documents"):
        documents = retriever.list_documents()
        manifest["knowledge_base_documents"] = [document.model_dump(mode="json") for document in documents]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dataset_snapshot.jsonl").write_text("\n".join(json.dumps(asdict(case), ensure_ascii=False) for case in cases) + "\n", encoding="utf-8")
    rows = []
    with result_path.open("w", encoding="utf-8") as output:
        for case in cases:
            thread_id = f"eval-{uuid4().hex}-{case.id}"
            started = time.perf_counter()
            try:
                warmup = []
                async with asyncio.timeout(settings.run_timeout_seconds * (len(case.turns) + 2)):
                    for turn in case.turns:
                        prior = await _run_turn(engine, turn["question"], thread_id, mode, turn.get("clarification"))
                        warmup.append(prior)
                        if prior["status"] != "success":
                            raise RuntimeError(f"前置对话未完成：{prior['status']} {prior.get('error') or ''}")
                    result = await _run_turn(engine, case.question, thread_id, mode, case.clarification)
                result["warmup_turns"] = warmup
            except TimeoutError:
                result = {"status": "timeout", "answer": "", "sources": [], "metrics": {}, "error": "评测题超时", "events": []}
            except Exception as exc:
                result = {"status": "error", "answer": "", "sources": [], "metrics": {}, "error": str(exc), "events": []}
            result.update(case=asdict(case), thread_id=thread_id, latency_seconds=time.perf_counter() - started)
            actual_contexts = result["metrics"].get("generation_contexts")
            result["retrieved_contexts"] = actual_contexts if isinstance(actual_contexts, list) else [source.get("text", "") for source in result["sources"]]
            result["context_scope"] = "final_generation_contexts" if isinstance(actual_contexts, list) else "retrieved_sources_before_generation_compression"
            result["retrieval"] = retrieval_metrics(case, result["sources"])
            before = _finite_number(result["metrics"].get("context_tokens_before"))
            after = _finite_number(result["metrics"].get("context_tokens_after"))
            result["compression_ratio"] = after / before if before and after is not None else None
            result["reference_token_coverage"] = sum(1 for term in case.expected_contains if term.casefold() in result["answer"].casefold()) / len(case.expected_contains) if case.expected_contains else None
            eligible = case.kind not in {"ambiguous", "unanswerable", "abstain"} and bool(case.reference)
            result["ragas_eligible"] = eligible
            result["ragas_scores"], result["ragas_errors"] = {}, {}
            if scorer and eligible and result["status"] == "success":
                result["ragas_scores"], result["ragas_errors"] = await scorer.score(case, result["answer"], result["retrieved_contexts"])
            elif scorer and eligible:
                result["ragas_errors"] = {"generation": result["error"] or result["status"]}
            output.write(json.dumps(result, ensure_ascii=False, allow_nan=False, default=str) + "\n")
            output.flush()
            rows.append(result)
    report = summarize(rows, manifest)
    if scorer:
        manifest["judge"]["versions"] = scorer.versions
        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def summarize(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    statuses = {status: sum(row["status"] == status for row in rows) for status in ("success", "clarification_required", "incomplete", "timeout", "error")}
    kinds = {kind: {"count": sum(row["case"]["kind"] == kind for row in rows), "completed": sum(row["case"]["kind"] == kind and row["status"] == "success" for row in rows)} for kind in sorted({row["case"]["kind"] for row in rows})}
    retrieval = {}
    for key in ("document_recall_at_5", "document_mrr"):
        values = [row["retrieval"][key] for row in rows if row["retrieval"][key] is not None]
        retrieval[key] = {"mean": statistics.mean(values) if values else None, "count": len(values)}
    eligible = [row for row in rows if row["ragas_eligible"]]
    score_names = sorted({name for row in rows for name in row["ragas_scores"]})
    if manifest["ragas_enabled"]:
        score_names = sorted(set(score_names) | {"answer_correctness", "context_recall", "faithfulness"})
    ragas = {}
    for name in score_names:
        values = [row["ragas_scores"][name] for row in eligible if row["ragas_scores"].get(name) is not None]
        ragas[name] = {"mean_valid_only": statistics.mean(values) if values else None, "valid_count": len(values), "eligible_count": len(eligible), "missing_or_failed_count": len(eligible) - len(values)}
    compressed = [row for row in rows if row["compression_ratio"] is not None]
    before = sum(row["metrics"]["context_tokens_before"] for row in compressed)
    after = sum(row["metrics"]["context_tokens_after"] for row in compressed)
    latencies = sorted(row["latency_seconds"] for row in rows)
    return {
        "case_count": len(rows), "statuses": statuses, "by_kind": kinds,
        "quality_evaluation": manifest["quality_evaluation"], "ragas_enabled": manifest["ragas_enabled"],
        "retrieval_unit": "document; deduplicated final source ordering, not chunk Recall@K and not RAGAS ContextRecall",
        "retrieval": retrieval, "ragas": ragas,
        "compression": {"definition": "after_tokens / before_tokens", "measured_count": len(compressed), "mean_sample_ratio": statistics.mean([row["compression_ratio"] for row in compressed]) if compressed else None, "token_weighted_ratio": after / before if before else None},
        "latency_seconds": {"mean": statistics.mean(latencies) if latencies else None, "p95": latencies[min(len(latencies) - 1, math.ceil(len(latencies) * .95) - 1)] if latencies else None},
        "notice": "未自动把无答案题的完成执行当作正确拒答；关键字覆盖仅作诊断。质量结论需参考逐题结果及人工核验。",
    }


RAGAS_METRICS = ("answer_correctness", "context_recall", "faithfulness")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """A committed artifact is either complete or absent, including after restart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class _RunJournal:
    def __init__(self, path: Path):
        self.path = path
        self.records = []
        self.completed: dict[str, dict] = {}
        self.started: set[str] = set()
        if path.exists():
            raw = path.read_bytes()
            offset = 0
            lines = raw.splitlines(keepends=True)
            for index, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if index != len(lines) - 1 or line.endswith(b"\n"):
                        raise ValueError("journal中间记录损坏，拒绝静默忽略") from None
                    backup = path.with_name(path.name + ".uncommitted-tail-" + uuid4().hex)
                    backup.write_bytes(raw[offset:])
                    with path.open("r+b") as stream:
                        stream.truncate(offset)
                        stream.flush()
                        os.fsync(stream.fileno())
                    break
                self._index(record)
                offset += len(line)
            if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
                with path.open("ab") as stream:
                    stream.write(b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())

    def _index(self, record):
        self.records.append(record)
        if record["state"] == "started":
            self.started.add(record["key"])
        elif record["state"] == "completed":
            previous = self.completed.get(record["key"])
            if previous is not None and canonical_hash(previous) != canonical_hash(record["payload"]):
                raise ValueError("journal唯一键重复且内容不同")
            self.completed[record["key"]] = record["payload"]
            self.started.discard(record["key"])

    def append(self, key, state, payload=None):
        record = {"key": key, "state": state, "at": datetime.now(UTC).isoformat(), "payload": payload}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._index(record)

    def begin(self, key):
        if key not in self.completed and key not in self.started:
            self.append(key, "started")

    def commit(self, key, payload):
        if key in self.completed:
            if canonical_hash(self.completed[key]) != canonical_hash(payload):
                raise ValueError("拒绝覆写已提交的评测项")
            return
        self.append(key, "completed", payload)


class BenchmarkRun:
    """Frozen two-stage benchmark with durable per-turn and per-metric commits.

    Take an exclusive OS lock around calls in a runner. A terminal failure is
    retained and skipped on resume; retry_failed creates a separate attempt.
    An uncommitted in-flight model call is never silently repeated.
    """

    def __init__(self, directory: Path, cases: list[EvaluationCase], environment: dict,
                 *, mode="agent", stability_repeats=3, stability_ids: list[str] | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cases = {case.id: case for case in cases}
        if len(self.cases) != len(cases):
            raise ValueError("重复case ID")
        if stability_repeats < 1:
            raise ValueError("重复次数必须为正")
        default_selection = stability_ids is None
        if default_selection:
            stability_ids = []
            for kind in ("single_document", "cross_document", "unanswerable"):
                matching = sorted(case.id for case in cases if case.kind == kind)
                if matching:
                    stability_ids.append(matching[0])
            if not stability_ids:
                stability_ids = sorted(self.cases)[:3]
        if len(set(stability_ids)) != len(stability_ids) or any(key not in self.cases for key in stability_ids):
            raise ValueError("稳定性集合必须是固定题集内互不重复的ID")
        frozen = {"schema": 1, "environment": environment, "cases": [asdict(case) for case in cases],
                  "mode": mode, "stability_ids": stability_ids, "stability_repeats": stability_repeats,
                  "stability_selection": ("first case per predetermined kind" if default_selection else "explicit case IDs")
                                         + ", before scores; repeat0 reused from main"}
        path = self.directory / "manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("frozen_hash") != canonical_hash(frozen):
                raise ValueError("冻结清单不匹配：代码、数据、文档、模型、配置或样本计划已改变，拒绝续跑")
        else:
            manifest = {"run_id": uuid4().hex, "created_at": datetime.now(UTC).isoformat(),
                        "frozen_hash": canonical_hash(frozen), "frozen": frozen}
            atomic_json(path, manifest)
        self.manifest = manifest
        self.mode = mode
        self.stability_ids = stability_ids
        self.stability_repeats = stability_repeats
        self.journal = _RunJournal(self.directory / "journal.jsonl")
        self.progress = None
        (self.directory / "artifacts").mkdir(exist_ok=True)

    def _notify(self, phase, **values):
        if self.progress is not None:
            self.progress({"phase": phase, "at": datetime.now(UTC).isoformat(), **values})

    def slots(self, scope="all"):
        if scope not in {"all", "main", "stability"}:
            raise ValueError("scope必须为all/main/stability")
        values = [(case_id, 0) for case_id in self.cases] if scope != "stability" else [(case_id, 0) for case_id in self.stability_ids]
        if scope != "main":
            values.extend((case_id, repeat) for repeat in range(1, self.stability_repeats) for case_id in self.stability_ids)
        return values

    def item_id(self, case_id, repeat):
        return f"{self.mode}:{case_id}:repeat{repeat}"

    def _latest(self, phase, item, **filters):
        values = [row for row in self.journal.completed.values() if row.get("phase") == phase
                  and row.get("item") == item and all(row.get(k) == v for k, v in filters.items())]
        return max(values, key=lambda row: row.get("attempt", 1)) if values else None

    async def _turn(self, engine, question, clarification, thread_id, key):
        cached = self.journal.completed.get(key)
        if cached:
            return cached
        event_path = self.directory / "artifacts" / (canonical_hash(key) + ".events.jsonl")
        events = []
        if event_path.exists():
            lines = event_path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                try:
                    events.append(json.loads(line))
                except ValueError:
                    if index != len(lines) - 1:
                        raise
        if key in self.journal.started:
            if any(event.get("type") == "done" for event in events):
                result = self._event_result(events)
                result["recovered_from_durable_events"] = True
            else:
                result = {"status": "interrupted", "answer": "", "sources": [], "metrics": {},
                          "error": "进程在此turn提交前中断；是否已有外部模型调用完成未知，未静默重做"}
            result["event_file"] = str(event_path)
            self.journal.commit(key, result)
            return result
        self.journal.begin(key)
        started = time.monotonic()
        try:
            for part in range(2):
                async for event in engine.astream(question if part == 0 else None, thread_id,
                        clarification=clarification if part else None, mode=self.mode):
                    events.append(event)
                    with event_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                if not (part == 0 and clarification and events and events[-1].get("type") == "clarification"):
                    break
            result = self._event_result(events)
        except asyncio.CancelledError:
            self.journal.commit(key, {"status": "interrupted", "answer": "", "sources": [], "metrics": {},
                                     "error": "显式取消，未自动重复调用", "event_file": str(event_path)})
            raise
        except Exception as exc:
            result = {"status": "error", "answer": "", "sources": [], "metrics": {}, "error": str(exc)}
        result.update(event_file=str(event_path), latency_seconds=time.monotonic() - started)
        self.journal.commit(key, result)
        return result

    @staticmethod
    def _event_result(events):
        done = next((event for event in reversed(events) if event.get("type") == "done"), None)
        sources = next((event.get("sources", []) for event in reversed(events) if event.get("type") == "sources"), [])
        errors = [event.get("text", "执行失败") for event in events if event.get("type") == "error"]
        pending = bool(events and events[-1].get("type") == "clarification")
        return {"status": "error" if errors else "success" if done else "clarification_required" if pending else "incomplete",
                "answer": done.get("answer", "") if done else "", "metrics": done.get("metrics", {}) if done else {},
                "sources": sources, "error": "；".join(errors) or None}

    async def generate(self, engine, *, scope="all", retry_failed=False, limit=None):
        processed = 0
        for case_id, repeat in self.slots(scope):
            item = self.item_id(case_id, repeat)
            previous = self._latest("generation", item)
            if previous and (previous["status"] == "success" or not retry_failed):
                continue
            if limit is not None and processed >= limit:
                break
            attempt = previous.get("attempt", 1) + 1 if previous else 1
            key = f"generation:{item}:attempt{attempt}"
            # A case can resume between turns with the same durable graph thread.
            thread_id = "benchmark-" + canonical_hash([self.manifest["run_id"], item, attempt])[:40]
            self.journal.begin(key)
            self._notify("generation_started", case_id=case_id, repeat=repeat, attempt=attempt)
            case = self.cases[case_id]
            started = time.monotonic()
            turns = [*case.turns, {"question": case.question, "clarification": case.clarification}]
            results = []
            for index, turn in enumerate(turns):
                result = await self._turn(engine, turn["question"], turn.get("clarification"), thread_id,
                                          f"{key}:turn{index}")
                results.append(result)
                if result["status"] != "success":
                    break
            result = dict(results[-1])
            final_turn_reached = len(results) == len(turns)
            if not final_turn_reached:
                result.update(answer="", error="前置对话未完成：" + str(result.get("error") or result["status"]))
            contexts = result["metrics"].get("generation_contexts")
            result.update(phase="generation", item=item, case_id=case_id, repeat=repeat, attempt=attempt,
                          thread_id=thread_id, case_hash=canonical_hash(asdict(case)), final_turn_reached=final_turn_reached,
                          completed_at=datetime.now(UTC).isoformat(), resume_processing_seconds=time.monotonic() - started,
                          latency_seconds=sum(turn["latency_seconds"] for turn in results)
                          if all(isinstance(turn.get("latency_seconds"), (int, float)) for turn in results) else None,
                          turns=results, retrieved_contexts=contexts if isinstance(contexts, list) else [],
                          context_scope="final_generation_contexts" if isinstance(contexts, list) else "not_recorded")
            result["generation_hash"] = canonical_hash(result)
            self.journal.commit(key, result)
            self._notify("generation_completed", case_id=case_id, repeat=repeat,
                         status=result["status"], elapsed_seconds=result["latency_seconds"])
            processed += 1
            self.write_report()
        return self.write_report()

    def _vector_path(self, case, generation):
        key = canonical_hash([case.reference, generation["answer"], self.manifest["frozen_hash"]])
        return self.directory / "artifacts" / (key + ".embeddings.json")

    def _load_vectors(self, case, generation):
        path = self._vector_path(case, generation)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("frozen_hash") != self.manifest["frozen_hash"] or payload.get("input_hash") != canonical_hash([case.reference, generation["answer"]]):
            raise ValueError("预计算向量的配置或样本摘要不匹配")
        vectors = payload.get("vectors")
        if not isinstance(vectors, dict) or payload.get("vectors_hash") != canonical_hash(vectors):
            raise ValueError("预计算向量损坏")
        for text in (case.reference or " ", generation["answer"] or " "):
            vector = vectors.get(text)
            if not isinstance(vector, list) or not vector or any(_finite_number(value) is None for value in vector):
                raise ValueError("预计算向量缺少样本文本或包含无效数值")
        return vectors

    def _eligible_generations(self, scope):
        values = []
        for case_id, repeat in self.slots(scope):
            item = self.item_id(case_id, repeat)
            generation = self._latest("generation", item)
            if generation is None:
                raise ValueError("当前scope仍有未提交的生成项，请先完成generate；不边生成边评分")
            case = self.cases[case_id]
            if case.kind not in {"ambiguous", "unanswerable", "abstain"} and case.reference:
                values.append((case_id, repeat, case, item, generation))
        return values

    async def prepare_vectors(self, scorer, *, scope="all", limit=None):
        """No judge requests in this stage. Caller unloads Ollama before BGE."""
        processed = 0
        try:
            for case_id, repeat, case, item, generation in self._eligible_generations(scope):
                if generation["status"] != "success" or generation["context_scope"] != "final_generation_contexts":
                    continue
                path = self._vector_path(case, generation)
                key = "vectors:" + canonical_hash([case.reference, generation["answer"], self.manifest["frozen_hash"]])
                if path.exists():
                    vectors = self._load_vectors(case, generation)
                    if key not in self.journal.completed:
                        self.journal.commit(key, {"phase": "vectors", "item": item, "case_id": case_id, "repeat": repeat,
                            "status": "success", "path": str(path), "vectors_hash": canonical_hash(vectors),
                            "recovered_from_atomic_cache": True})
                    continue
                if limit is not None and processed >= limit:
                    break
                self.journal.begin(key)
                self._notify("vectors_started", case_id=case_id, repeat=repeat)
                vectors = await scorer.prepare_embeddings(case.reference, generation["answer"])
                payload = {"frozen_hash": self.manifest["frozen_hash"],
                           "input_hash": canonical_hash([case.reference, generation["answer"]]),
                           "vectors_hash": canonical_hash(vectors), "vectors": vectors}
                atomic_json(path, payload)
                self._load_vectors(case, generation)
                self.journal.commit(key, {"phase": "vectors", "item": item, "case_id": case_id, "repeat": repeat,
                    "status": "success", "path": str(path), "vectors_hash": payload["vectors_hash"]})
                processed += 1
                self._notify("vectors_completed", case_id=case_id, repeat=repeat)
        finally:
            await scorer.close()
        return self.write_report()

    async def score(self, scorer, *, scope="all", retry_failed=False, limit=None):
        processed = 0
        try:
            eligible = self._eligible_generations(scope)
            # Preflight every needed vector BEFORE starting any judge request.
            for _case_id, _repeat, case, item, generation in eligible:
                needed = any((previous := self._latest("score", item, metric=metric, generation_hash=generation["generation_hash"])) is None
                             or (retry_failed and previous["status"] != "success") for metric in RAGAS_METRICS)
                if needed and generation["status"] == "success" and generation["context_scope"] == "final_generation_contexts":
                    if not self._vector_path(case, generation).is_file():
                        raise ValueError("缺少预计算向量：先完成prepare-vectors；score阶段禁止临时加载BGE")
                    self._load_vectors(case, generation)
            # Metric-major batches retain the same judge/template prefix in KV cache.
            for metric in RAGAS_METRICS:
                for case_id, repeat, case, item, generation in eligible:
                    generation_hash = generation["generation_hash"]
                    previous = self._latest("score", item, metric=metric, generation_hash=generation_hash)
                    if previous and (previous["status"] == "success" or not retry_failed):
                        continue
                    if limit is not None and processed >= limit:
                        return self.write_report()
                    attempt = previous.get("attempt", 1) + 1 if previous else 1
                    key = f"score:{item}:{generation_hash}:{metric}:attempt{attempt}"
                    was_inflight = key in self.journal.started
                    self.journal.begin(key)
                    self._notify("metric_started", case_id=case_id, repeat=repeat, metric=metric, attempt=attempt)
                    started = time.monotonic()
                    value, errors, versions = None, {}, {}
                    if was_inflight:
                        status, errors = "interrupted", {"interrupted": "该指标上次未提交，外部调用完成情况未知；未静默重复计分"}
                    elif generation["status"] != "success":
                        status, errors = "blocked_generation", {"generation": generation.get("error") or generation["status"]}
                    elif generation["context_scope"] != "final_generation_contexts":
                        status, errors = "error", {"contexts": "未记录实际最终上下文，拒绝用未压缩原文替代"}
                    else:
                        try:
                            vectors = self._load_vectors(case, generation)
                            scores, errors = await scorer.score(case, generation["answer"], generation["retrieved_contexts"],
                                                               metric_names=[metric], embeddings=vectors)
                            value = _finite_number(scores.get(metric))
                            versions = dict(getattr(scorer, "versions", {}))
                            status = "success" if value is not None and not errors else "error"
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            status, errors = "error", {"scoring": str(exc)}
                    self.journal.commit(key, {"phase": "score", "item": item, "case_id": case_id, "repeat": repeat,
                        "generation_hash": generation_hash, "metric": metric, "attempt": attempt, "status": status,
                        "score": value, "errors": errors, "worker_versions": versions,
                        "completed_at": datetime.now(UTC).isoformat(), "latency_seconds": time.monotonic() - started})
                    processed += 1
                    self._notify("metric_completed", case_id=case_id, repeat=repeat, metric=metric,
                                 status=status, score=value)
                    self.write_report()
        finally:
            await scorer.close()
        return self.write_report()

    async def retrieval_ablation(self, retriever, *, modes=("dense", "hybrid", "rerank", "parent"), limit=None):
        processed = 0
        for case in self.cases.values():
            for mode in modes:
                search_mode = "full" if mode == "parent" else mode
                key = f"retrieval:{mode}:{case.id}"
                if key in self.journal.completed:
                    continue
                if limit is not None and processed >= limit:
                    return self.write_report()
                was_inflight = key in self.journal.started
                self.journal.begin(key)
                sources, error = [], None
                started = time.monotonic()
                try:
                    if was_inflight:
                        raise RuntimeError("上次检索未提交，未静默重复执行")
                    evidence = await asyncio.to_thread(retriever.search, case.question, mode=search_mode, top_k=5)
                    sources = [item.model_dump() if hasattr(item, "model_dump") else item for item in evidence]
                except Exception as exc:
                    error = str(exc)
                self.journal.commit(key, {"phase": "retrieval", "case_id": case.id, "mode": mode,
                    "search_mode": search_mode, "requested_k": 5,
                    "status": "error" if error else "success", "error": error, "sources": sources,
                    "metrics": retrieval_metrics(case, sources), "latency_seconds": time.monotonic() - started})
                processed += 1
        return self.write_report()

    def _report_slots(self, slots):
        rows = [(case_id, repeat, self._latest("generation", self.item_id(case_id, repeat))) for case_id, repeat in slots]
        statuses = Counter(row[2]["status"] if row[2] else "not_started" for row in rows)
        eligible = [(case_id, repeat, generation) for case_id, repeat, generation in rows
                    if self.cases[case_id].reference and self.cases[case_id].kind not in {"ambiguous", "unanswerable", "abstain"}]
        metrics = {}
        for metric in RAGAS_METRICS:
            records = [self._latest("score", self.item_id(case_id, repeat), metric=metric,
                                   generation_hash=generation["generation_hash"]) if generation else None
                       for case_id, repeat, generation in eligible]
            values = [row["score"] for row in records if row and row["status"] == "success"]
            items = {self.item_id(case_id, repeat) for case_id, repeat in slots}
            attempts = [row for row in self.journal.completed.values() if row.get("phase") == "score"
                        and row.get("item") in items and row.get("metric") == metric]
            metrics[metric] = {"eligible_count": len(eligible), "valid_count": len(values),
                "not_started_count": sum(row is None for row in records),
                "failed_or_blocked_count": sum(row is not None and row["status"] != "success" for row in records),
                "mean_valid_only": statistics.mean(values) if values else None,
                "stddev_population": statistics.pstdev(values) if values else None,
                "min": min(values) if values else None, "max": max(values) if values else None}
            metrics[metric].update(total_attempt_count=len(attempts),
                                   historical_failed_attempt_count=sum(row["status"] != "success" for row in attempts))
        return {"planned_generations": len(slots), "statuses": dict(statuses), "metrics": metrics,
                "generation_attempted_all": statuses.get("not_started", 0) == 0,
                "eligible_metrics_attempted_all": all(value["not_started_count"] == 0 for value in metrics.values()),
                "all_metric_scores_valid": all(value["valid_count"] == value["eligible_count"] for value in metrics.values())}

    def write_report(self):
        main_slots = [(case_id, 0) for case_id in self.cases]
        stable_slots = [(case_id, repeat) for case_id in self.stability_ids for repeat in range(self.stability_repeats)]
        stability_scope = "full_dataset" if len(self.stability_ids) == len(self.cases) else "subset" if self.stability_ids else "disabled"
        scope_label = {"full_dataset": "锁定完整题集", "subset": "锁定小集", "disabled": "未选择题目"}[stability_scope]
        report = {"run_id": self.manifest["run_id"], "frozen_hash": self.manifest["frozen_hash"],
                  "updated_at": datetime.now(UTC).isoformat(), "main": self._report_slots(main_slots),
                  "stability": {"ids": self.stability_ids, "repeats": self.stability_repeats,
                                "scope": stability_scope, "selected_case_count": len(self.stability_ids),
                                "dataset_case_count": len(self.cases),
                                "repeat0_reused_from_main": True, "aggregate": self._report_slots(stable_slots),
                                "per_case": {case_id: self._report_slots([(case_id, repeat) for repeat in range(self.stability_repeats)]) for case_id in self.stability_ids}},
                  "notice": f"main为第一轮；稳定性计划为{scope_label}{len(self.stability_ids)}题、共{self.stability_repeats}轮，首轮复用main；计划不等于已完成，见执行计数。失败、未评分和中断均保留分母；不等于人工gold或生产效果。"}
        per_repeat = {str(repeat): self._report_slots([(case_id, repeat) for case_id in self.stability_ids])
                      for repeat in range(self.stability_repeats)}
        report["stability"]["per_repeat"] = per_repeat
        repeat_statistics = {}
        for metric in RAGAS_METRICS:
            means = [row["metrics"][metric]["mean_valid_only"] for row in per_repeat.values()
                     if row["metrics"][metric]["eligible_count"] > 0
                     and row["metrics"][metric]["valid_count"] == row["metrics"][metric]["eligible_count"]]
            repeat_statistics[metric] = {"planned_repeats": self.stability_repeats,
                "complete_scored_repeat_count": len(means),
                "mean_of_complete_repeat_means": statistics.mean(means) if means else None,
                "stddev_of_complete_repeat_means": statistics.pstdev(means) if means else None,
                "min_repeat_mean": min(means) if means else None, "max_repeat_mean": max(means) if means else None}
        report["stability"]["repeat_mean_statistics"] = repeat_statistics
        report["inflight_keys"] = sorted(self.journal.started)
        vector_generations = [(self.cases[case_id], generation) for case_id, repeat in self.slots()
                              if (generation := self._latest("generation", self.item_id(case_id, repeat)))
                              and generation["status"] == "success" and self.cases[case_id].reference
                              and self.cases[case_id].kind not in {"ambiguous", "unanswerable", "abstain"}]
        report["vector_preparation"] = {"successful_eligible_generations": len(vector_generations),
            "logical_items_with_vector_files": sum(self._vector_path(case, generation).is_file() for case, generation in vector_generations)}
        retrieval = {}
        for mode in ("dense", "hybrid", "rerank", "parent"):
            records = [row for row in self.journal.completed.values() if row.get("phase") == "retrieval" and row.get("mode") == mode]
            values = [row["metrics"]["document_recall_at_5"] for row in records if row["metrics"]["document_recall_at_5"] is not None]
            retrieval[mode] = {"planned_count": len(self.cases), "attempted_count": len(records),
                "search_mode": "full" if mode == "parent" else mode, "requested_k": 5,
                "failed_count": sum(row["status"] != "success" for row in records),
                "specified_reference_document_recall_at_5": statistics.mean(values) if values else None,
                "denominator_with_reference_documents": len(values)}
        report["retrieval_ablation"] = retrieval
        atomic_json(self.directory / "summary.json", report)
        return report
