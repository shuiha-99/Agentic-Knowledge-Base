"""Replay only ContextRecall for the frozen, already committed generations.

No generation, embedding preparation or other metric is invoked. The original
benchmark manifest and planned denominators stay intact; the immutable selection
records this user's narrower scoring request separately.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark import environment_snapshot, exclusive_run, file_sha, make_scorer  # noqa: E402

from agentic_kb.config import Settings  # noqa: E402
from agentic_kb.evaluation import (  # noqa: E402
    BenchmarkRun,
    _finite_number,
    atomic_json,
    canonical_hash,
    load_dataset,
)

METRIC = "context_recall"


def freeze_selection(run: BenchmarkRun, scope: str) -> dict:
    """Pin every existing generation, then select eligible terminal records.

    Missing generation slots are neither created nor scored. Including excluded
    cases in the snapshot also detects a generation process continuing in parallel.
    """
    snapshot, items = [], []
    for case_id, repeat in run.slots(scope):
        item = run.item_id(case_id, repeat)
        generation = run._latest("generation", item)
        if generation is None:
            continue
        generation_hash = generation.get("generation_hash")
        if not generation_hash or generation_hash != canonical_hash({
            key: value for key, value in generation.items() if key != "generation_hash"
        }):
            raise ValueError(f"已提交生成内容与generation_hash不符：{item}")
        identity = {"item": item, "generation_hash": generation_hash}
        snapshot.append(identity)
        case = run.cases[case_id]
        if case.reference and case.kind not in {"ambiguous", "unanswerable", "abstain"}:
            items.append(identity)
    frozen = {"schema": 1, "metric": METRIC, "driver_sha256": file_sha(Path(__file__).resolve()),
              "frozen_hash": run.manifest["frozen_hash"], "scope": scope,
              "items": items, "expected_count": len(items), "generation_snapshot": snapshot}
    path = run.directory / "context_recall_selection.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if canonical_hash({key: value for key, value in stored.items() if key != "created_at"}) != canonical_hash(frozen):
            raise ValueError("ContextRecall选择已冻结：驱动、范围或生成item/hash已变化，拒绝混入新结果")
        return stored
    selection = {**frozen, "created_at": datetime.now(UTC).isoformat()}
    atomic_json(path, selection)
    return selection


async def score_existing(run: BenchmarkRun, scorer, *, scope="all", retry_failed=False, limit=None) -> dict:
    """Use the original metric hash/attempt journal, without its all-slot gate."""
    processed = 0
    try:
        selection = freeze_selection(run, scope)
        generations = {run.item_id(case_id, repeat): (case_id, repeat, run._latest(
            "generation", run.item_id(case_id, repeat))) for case_id, repeat in run.slots(scope)}
        for identity in selection["items"]:
            item, generation_hash = identity["item"], identity["generation_hash"]
            case_id, repeat, generation = generations[item]
            previous = run._latest("score", item, metric=METRIC, generation_hash=generation_hash)
            if previous and (previous["status"] == "success" or not retry_failed):
                continue
            if limit is not None and processed >= limit:
                break
            attempt = previous.get("attempt", 1) + 1 if previous else 1
            key = f"score:{item}:{generation_hash}:{METRIC}:attempt{attempt}"
            was_inflight = key in run.journal.started
            run.journal.begin(key)
            run._notify("metric_started", case_id=case_id, repeat=repeat, metric=METRIC, attempt=attempt)
            started = time.monotonic()
            value, errors, versions = None, {}, {}
            if was_inflight:
                status, errors = "interrupted", {"interrupted": "该指标上次未提交，外部调用完成情况未知；未静默重复计分"}
            elif generation["status"] != "success":
                status, errors = "blocked_generation", {"generation": generation.get("error") or generation["status"]}
            elif generation.get("context_scope") != "final_generation_contexts":
                status, errors = "error", {"contexts": "未记录实际最终上下文，拒绝用未压缩原文替代"}
            else:
                contexts = generation.get("retrieved_contexts")
                if not isinstance(contexts, list) or any(not isinstance(text, str) for text in contexts):
                    status, errors = "error", {"contexts": "实际最终上下文格式无效"}
                else:
                    try:
                        # Ragas LLMContextRecall uses reference + contexts only.
                        # Empty cache avoids encoding; no dummy embedding is made.
                        scores, errors = await scorer.score(run.cases[case_id], generation["answer"], contexts,
                                                            metric_names=[METRIC], embeddings={})
                        value = _finite_number(scores.get(METRIC))
                        versions = dict(getattr(scorer, "versions", {}))
                        if value is None and not errors:
                            errors = {METRIC: "worker未返回该指标的有效有限数值"}
                        status = "success" if value is not None and not errors else "error"
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        status, errors = "error", {"scoring": str(exc)}
            run.journal.commit(key, {"phase": "score", "item": item, "case_id": case_id, "repeat": repeat,
                "generation_hash": generation_hash, "metric": METRIC, "attempt": attempt, "status": status,
                "score": value, "errors": errors, "worker_versions": versions,
                "completed_at": datetime.now(UTC).isoformat(), "latency_seconds": time.monotonic() - started})
            processed += 1
            run._notify("metric_completed", case_id=case_id, repeat=repeat, metric=METRIC, status=status, score=value)
            run.write_report()
    finally:
        await scorer.close()
    return run.write_report()


async def execute(args):
    if Path.cwd().resolve() != ROOT:
        raise ValueError("请从项目根目录运行，保持冻结路径一致")
    settings = Settings.from_env()
    if settings.provider != "ollama":
        raise ValueError("真实ContextRecall评分拒绝offline-test")
    directory = args.run_dir.resolve()
    if not (directory / "manifest.json").is_file():
        raise ValueError("必须指定已有冻结benchmark目录，禁止新建生成实验")
    cases = load_dataset(args.dataset, args.split)
    environment = environment_snapshot(settings, ROOT, args.dataset,
                                       score_keep_alive=args.score_keep_alive, metric_timeout=args.metric_timeout)
    stability_ids = args.stability_ids.split(",") if args.stability_ids else None
    with exclusive_run(directory):
        run = BenchmarkRun(directory, cases, environment, mode=args.mode,
                           stability_repeats=args.stability_repeats, stability_ids=stability_ids)
        run.progress = lambda event: print(json.dumps(event, ensure_ascii=False), flush=True)
        scorer = make_scorer(settings, None, args)
        report = await score_existing(run, scorer, scope=args.scope, retry_failed=args.retry_failed, limit=args.limit)
        print(json.dumps(report, ensure_ascii=False, indent=2))


def parser():
    result = argparse.ArgumentParser(description="仅重放已提交结果的ContextRecall；不生成答案、不加载向量模型")
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--dataset", type=Path, default=Path("data/eval/demo_cases.jsonl"))
    result.add_argument("--split", choices=["dev", "test"], default="test")
    result.add_argument("--mode", choices=["agent", "dense", "hybrid", "rerank", "parent"], default="agent")
    result.add_argument("--scope", choices=["all", "main", "stability"], default="all")
    result.add_argument("--stability-repeats", type=int, default=3)
    result.add_argument("--stability-ids")
    result.add_argument("--metric-timeout", type=float, default=900)
    result.add_argument("--score-keep-alive", default="10m")
    result.add_argument("--retry-failed", action="store_true")
    result.add_argument("--limit", type=int)
    return result


if __name__ == "__main__":
    asyncio.run(execute(parser().parse_args()))
