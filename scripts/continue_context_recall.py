"""Serial ContextRecall continuation; original benchmark files are read-only inputs."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import shutil
import statistics
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from benchmark import environment_snapshot, exclusive_run, file_sha, make_scorer, unload_ollama_before_vectors  # noqa: E402

from agentic_kb.config import Settings  # noqa: E402
from agentic_kb.evaluation import BenchmarkRun, EvaluationCase, atomic_json, canonical_hash  # noqa: E402

METRIC = "context_recall"
LOG = logging.getLogger("context_recall_continuation")


def now():
    return datetime.now(UTC).isoformat()


def eligible(case):
    return bool(case.reference) and case.kind not in {"ambiguous", "unanswerable", "abstain"}


def latest_valid(run, item):
    records = [v for v in run.journal.completed.values() if v.get("phase") == "score"
               and v.get("item") == item and v.get("metric") == METRIC and v.get("status") == "success"
               and isinstance(v.get("score"), (int, float)) and math.isfinite(v["score"])]
    return records[-1] if records else None


def inherited_inflight_repairs(run, source_records):
    """An old uncommitted turn is not the explicitly requested new generation.

    The original journal can contain a started generation without a terminal
    generation record. Closing that old attempt as interrupted must be followed
    by one fresh attempt under the user's request to fill the four missing cases.
    """
    completed = {r["key"] for r in source_records if r["state"] == "completed"}
    pending = set()
    for row in source_records:
        key = row["key"]
        if row["state"] != "started" or not key.startswith("generation:") or ":turn" in key or key in completed:
            continue
        item, original_attempt = key[len("generation:"):].rsplit(":attempt", 1)
        generation = run._latest("generation", item)
        if (generation and generation.get("status") == "interrupted"
                and generation.get("attempt", 1) == int(original_attempt) and not latest_valid(run, item)):
            pending.add(item)
    return pending


def summarize(rows):
    values = [r["score"] for r in rows if r["status"] == "success"]
    failed = sum(r["status"] not in {"success", "unscored"} for r in rows)
    return {"total": len(rows), "valid_count": len(values), "failed_count": failed,
            "unscored_count": len(rows) - len(values) - failed,
            "mean_valid_only": statistics.mean(values) if values else None,
            "median_valid_only": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "stddev_population": statistics.pstdev(values) if values else None,
            "zero_count": sum(v == 0 for v in values),
            "zero_sample_ids": [r["sample_id"] for r in rows if r["score"] == 0],
            "zero_question_ids": sorted({r["question_id"] for r in rows if r["score"] == 0}),
            "coverage": len(values) / len(rows) if rows else None,
            "statistics_policy": "Valid finite RAGAS scores only; failures remain null and in total/coverage denominator."}


def detail(run, slots):
    rows = []
    for cid, repeat in slots:
        item = run.item_id(cid, repeat)
        generation = run._latest("generation", item)
        score = latest_valid(run, item)
        if score is None and generation is not None:
            # Attempts are numbered within each generation hash. Selecting by
            # item alone can pick an older blocked attempt with the same number.
            score = run._latest("score", item, metric=METRIC,
                                generation_hash=generation["generation_hash"])
        rows.append({"question_id": cid, "repeat": repeat, "sample_id": f"{cid}:repeat{repeat}",
                     "score": score.get("score") if score and score["status"] == "success" else None,
                     "retrieved_contexts_count": len(generation.get("retrieved_contexts", [])) if generation else 0,
                     "status": score["status"] if score else "unscored",
                     "generation_status": generation["status"] if generation else "not_started",
                     "errors": score.get("errors", {}) if score else {},
                     "generation_hash": generation.get("generation_hash") if generation else None})
    return rows


async def generate_one(run, engine, cid, repeat):
    item = run.item_id(cid, repeat)
    previous = run._latest("generation", item)
    attempt = previous.get("attempt", 1) + 1 if previous else 1
    key = f"generation:{item}:attempt{attempt}"
    # Resume a started but uncommitted generation without inventing a new attempt.
    if key not in run.journal.started:
        run.journal.begin(key)
    thread_id = "benchmark-" + canonical_hash([run.manifest["run_id"], item, attempt])[:40]
    case = run.cases[cid]
    started = time.monotonic()
    turns = [*case.turns, {"question": case.question, "clarification": case.clarification}]
    results = []
    for i, turn in enumerate(turns):
        result = await run._turn(engine, turn["question"], turn.get("clarification"), thread_id, f"{key}:turn{i}")
        results.append(result)
        if result["status"] != "success":
            LOG.error("Generation turn failed: %s turn=%s payload=%s", item, i, json.dumps(result, ensure_ascii=False))
            break
    result = dict(results[-1])
    reached = len(results) == len(turns)
    if not reached:
        result.update(answer="", error="Prerequisite conversation failed: " + str(result.get("error") or result["status"]))
    contexts = result.get("metrics", {}).get("generation_contexts")
    result.update(phase="generation", item=item, case_id=cid, repeat=repeat, attempt=attempt,
                  thread_id=thread_id, case_hash=canonical_hash(asdict(case)), final_turn_reached=reached,
                  completed_at=now(), resume_processing_seconds=time.monotonic() - started,
                  turns=results, retrieved_contexts=contexts if isinstance(contexts, list) else [],
                  context_scope="final_generation_contexts" if isinstance(contexts, list) else "not_recorded")
    result["generation_hash"] = canonical_hash(result)
    run.journal.commit(key, result)
    return result


async def score_one(run, scorer, cid, repeat, generation):
    item = run.item_id(cid, repeat)
    prior = run._latest("score", item, metric=METRIC, generation_hash=generation["generation_hash"])
    attempt = prior.get("attempt", 1) + 1 if prior else 1
    key = f"score:{item}:{generation['generation_hash']}:{METRIC}:attempt{attempt}"
    inflight = key in run.journal.started
    run.journal.begin(key)
    started = time.monotonic()
    value, errors = None, {}
    if inflight:
        status, errors = "scoring_failed", {"interrupted": "Uncommitted score found; not silently repeated."}
    elif generation["status"] != "success":
        status, errors = "generation_failed", {"generation": generation.get("error") or generation["status"]}
    elif generation.get("context_scope") != "final_generation_contexts":
        status, errors = "scoring_failed", {"contexts": "Actual final generation contexts not recorded."}
    else:
        try:
            contexts = generation["retrieved_contexts"]
            if not isinstance(contexts, list) or any(not isinstance(c, str) for c in contexts):
                raise ValueError("Invalid retrieved_contexts")
            scores, errors = await scorer.score(run.cases[cid], generation["answer"], contexts,
                                                 metric_names=[METRIC], embeddings={})
            raw = scores.get(METRIC)
            value = float(raw) if isinstance(raw, (int, float)) and math.isfinite(raw) and 0 <= raw <= 1 else None
            if value is None and not errors:
                errors = {METRIC: "No finite RAGAS score in [0,1]."}
            status = "success" if value is not None and not errors else "scoring_failed"
            if status != "success":
                value = None
        except Exception:
            LOG.exception("Scoring exception: %s", item)
            status, errors = "scoring_failed", {"exception": __import__("traceback").format_exc()}
    row = {"phase": "score", "item": item, "case_id": cid, "repeat": repeat,
           "generation_hash": generation["generation_hash"], "metric": METRIC, "attempt": attempt,
           "status": status, "score": value, "errors": errors, "worker_versions": dict(scorer.versions),
           "completed_at": now(), "latency_seconds": time.monotonic() - started}
    run.journal.commit(key, row)
    if errors:
        LOG.error("Sample failed: %s %s", item, json.dumps(row, ensure_ascii=False))
    return row


async def execute(args):
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Continuation must use a separate sibling directory.")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=output / "exceptions.log", encoding="utf-8", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)
    started = time.monotonic()
    seed = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    frozen = seed["frozen"]
    settings = Settings(**frozen["environment"]["settings"])
    # Initialize the same local model cache environment without loading .env overrides.
    os.environ.setdefault("HF_HOME", str(settings.models_dir / "hf"))
    os.environ.setdefault("TORCH_HOME", str(settings.models_dir / "torch"))
    if args.start_ollama:
        from agentic_kb.runtime import start_ollama
        service_settings = settings.model_copy(update={"data_dir": ROOT / "runtime/local_ollama_service"})
        LOG.info("Local Ollama service: %s", start_ollama(service_settings))
    actual = environment_snapshot(settings, ROOT, Path(frozen["environment"]["dataset_path"]),
                                  score_keep_alive=args.score_keep_alive, metric_timeout=args.metric_timeout)
    if canonical_hash(actual) != canonical_hash(frozen["environment"]):
        raise ValueError("Frozen environment drift: " + str([k for k in actual if actual[k] != frozen['environment'][k]]))
    with exclusive_run(source), exclusive_run(output):
        provenance_path = output / "source_snapshot.json"
        source_hashes = {name: file_sha(source / name) for name in
                         ["manifest.json", "journal.jsonl", "summary.json", "context_recall_selection.json"]}
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if source_hashes != provenance["source_sha256"]:
                raise ValueError("Original benchmark changed since continuation started.")
        else:
            provenance = {"created_at": now(), "source": str(source), "source_sha256": source_hashes,
                          "driver_sha256": file_sha(Path(__file__)), "metric": METRIC,
                          "deduplication": "question_id + repeat; original 64 successful records immutable"}
            atomic_json(provenance_path, provenance)
            for name in ["manifest.json", "journal.jsonl"]:
                shutil.copyfile(source / name, output / name)
        run = BenchmarkRun(output, [EvaluationCase(**c) for c in frozen["cases"]], actual,
                           mode=frozen["mode"], stability_repeats=frozen["stability_repeats"],
                           stability_ids=frozen["stability_ids"])
        initial_generations = {}
        source_records = [json.loads(line) for line in (source / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
        for record in source_records:
            value = record.get("payload") or {}
            if record["state"] == "completed" and value.get("phase") == "generation":
                initial_generations[value["item"]] = value
        all_slots = [(c, r) for c, r in run.slots("all") if eligible(run.cases[c])]
        slots = all_slots if args.total_scope == "all" else [(c, r) for c, r in all_slots if r == 0]
        state_path = output / "continuation_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
            "created_at": now(), "active_seconds": 0.0, "processed": [], "initial_valid": 64}
        state.setdefault("invocations", []).append({"at": now(), "pid": os.getpid(),
            "driver_sha256": file_sha(Path(__file__)), "work_scope": args.work_scope, "total_scope": args.total_scope})
        done = set(state["processed"])
        # A new explicit continuation request may authorize one fresh attempt for
        # a turn left uncommitted by the last process. Keep this local receipt
        # separate from the immutable original benchmark and its nine retries.
        resume_selection_path = output / "resume_inflight_selection.json"
        resume_selection = json.loads(resume_selection_path.read_text(encoding="utf-8")) if resume_selection_path.exists() else {"records": []}
        inherited_repairs = inherited_inflight_repairs(run, source_records + resume_selection["records"])
        done.difference_update(inherited_repairs)
        if inherited_repairs:
            LOG.warning("Explicit missing-case generation still needed after inherited interrupted turn: %s", sorted(inherited_repairs))
        preserved = {v["item"]: canonical_hash(v) for v in run.journal.completed.values()
                     if v.get("phase") == "score" and v.get("metric") == METRIC and v.get("status") == "success"}
        if len(preserved) < 64:
            raise ValueError("Missing original 64 successful scores.")
        scorer = make_scorer(settings, None, args)
        # Preserve every worker diagnostic, including caught timeout/parser errors.
        async def drain_stderr():
            while scorer.process and scorer.process.stderr:
                data = await scorer.process.stderr.read(4096)
                if not data:
                    return
                text = data.decode("utf-8", errors="replace")
                scorer.stderr_tail = (scorer.stderr_tail + text)[-12000:]
                with (output / "ragas_worker.stderr.log").open("a", encoding="utf-8") as stream:
                    stream.write(text)
        scorer._drain_stderr = drain_stderr
        engine = kb = None
        batch = []
        initial_active = state["active_seconds"]

        def save(batch_end=False):
            rows = detail(run, slots)
            summary = {**summarize(rows), "updated_at": now(), "metric": METRIC,
                       "ragas_version": "0.3.3", "model": settings.chat_model,
                       "active_seconds": initial_active + time.monotonic() - started,
                       "wall_seconds": (datetime.now(UTC) - datetime.fromisoformat(state["created_at"])).total_seconds(),
                       "original_64_preserved": all(canonical_hash(latest_valid(run, item)) == h for item, h in preserved.items()),
                       "work_scope": args.work_scope, "total_scope": args.total_scope}
            state.update(active_seconds=summary["active_seconds"], processed=sorted(done), updated_at=now(),
                         summary=summary, current_sample=None)
            atomic_json(state_path, state)
            atomic_json(output / "repair_queue.json", {"updated_at": now(), "pending": sorted(inherited_repairs),
                "reason": "Fill original unscored case after closing its inherited uncommitted generation; one fresh attempt only."})
            atomic_json(output / "context_recall_results.json", {"summary": summary, "per_sample": rows})
            with (output / "per_sample.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=["question_id", "repeat", "score", "retrieved_contexts_count", "status"])
                writer.writeheader()
                writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)
            lines = ["# ContextRecall 评测结果", "",
                     "状态：" + ("全部评测位置已处理" if summary["unscored_count"] == 0 else "运行中，以下为中间结果"), "",
                     f"适用评测位置：{summary['total']}；按 question_id＋repeat 去重。",
                     f"有效评分：{summary['valid_count']}；失败：{summary['failed_count']}；未评分：{summary['unscored_count']}。", "",
                     "仅计算 RAGAS 0.3.3 的 context_recall，评判模型为本地 Ollama qwen2.5:7b。",
                     "失败保留 null，并保留在总数和评分覆盖率的分母中；以下分数统计仅基于有效评分。", "",
                     "|统计项|值|", "|---|---:|"]
            for label, key in [("均值", "mean_valid_only"), ("中位数", "median_valid_only"),
                               ("最小值", "min"), ("最大值", "max"), ("总体标准差", "stddev_population")]:
                value = summary[key]
                lines.append(f"|{label}|{value:.8f}|" if value is not None else f"|{label}|null|")
            lines += ["", f"零分位置数：{summary['zero_count']}。",
                      "零分 question_id：" + "、".join(summary["zero_question_ids"]),
                      "零分完整位置：" + "、".join(summary["zero_sample_ids"]), "",
                      f"本轮累计运行耗时：{summary['active_seconds']:.1f} 秒；含暂停的墙钟时间：{summary['wall_seconds']:.1f} 秒。",
                      f"原有64条有效评分保持不变：{summary['original_64_preserved']}。", "",
                      "结果 JSON：" + str(output / "context_recall_results.json"),
                      "分批中间结果目录：" + str(output / "batches"),
                      "异常日志：" + str(output / "exceptions.log"),
                      "评分器诊断日志：" + str(output / "ragas_worker.stderr.log"), "",
                      "|question_id|repeat|score|retrieved_contexts 条数|status|",
                      "|---|---:|---:|---:|---|"]
            lines += [f"|{r['question_id']}|{r['repeat']}|{r['score'] if r['score'] is not None else 'null'}|{r['retrieved_contexts_count']}|{r['status']}|" for r in rows]
            (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            if batch_end and batch:
                atomic_json(output / "batches" / f"batch_{len(done):04d}.json", {"summary": summary, "samples": list(batch)})
                batch.clear()
            return summary

        summary = save()
        print(json.dumps({"event": "start", **summary}, ensure_ascii=False), flush=True)
        work = [(c, r) for c, r in slots if args.work_scope == "all" or r == 0]
        # Retry the nine previously blocked first, then fill missing generation slots.
        work.sort(key=lambda slot: (0 if (g := run._latest("generation", run.item_id(*slot))) and g["status"] != "success" else 1, slot[1], slot[0]))
        try:
            for cid, repeat in work:
                item = run.item_id(cid, repeat)
                if latest_valid(run, item) or item in done:
                    continue
                state.update(current_sample=item, current_started_at=now())
                atomic_json(state_path, state)
                print(json.dumps({"event": "sample_started", "sample_id": item}, ensure_ascii=False), flush=True)
                generation = run._latest("generation", item)
                initial = initial_generations.get(item)
                already_generated_this_run = generation and generation.get("attempt", 1) > (initial.get("attempt", 1) if initial else 0)
                if generation is None or (generation["status"] != "success" and (not already_generated_this_run or item in inherited_repairs)):
                    await scorer.close()
                    unload_ollama_before_vectors(settings)
                    if engine is None:
                        from agentic_kb.graph import GraphEngine
                        from agentic_kb.providers import build_providers
                        from agentic_kb.retrieval import KnowledgeBase
                        llm, embedder, reranker = build_providers(settings)
                        kb = KnowledgeBase(settings, embedder, reranker, token_counter=llm.count_tokens)
                        engine = GraphEngine(settings, llm, kb)
                    generation = await generate_one(run, engine, cid, repeat)
                row = await score_one(run, scorer, cid, repeat, generation)
                done.add(item)
                inherited_repairs.discard(item)
                batch.append(row)
                summary = save(batch_end=len(batch) >= args.batch_size)
                print(json.dumps({"event": "sample_completed", "sample_id": item, "status": row["status"],
                                  "score": row["score"], "elapsed_seconds": summary["active_seconds"]}, ensure_ascii=False), flush=True)
                completed = summary["valid_count"] + summary["failed_count"]
                if len(done) % 20 == 0:
                    print(f"PROGRESS {completed}/{summary['total']} mean={summary['mean_valid_only']}", flush=True)
            summary = save(batch_end=True)
            print(json.dumps({"event": "work_scope_finished", **summary}, ensure_ascii=False), flush=True)
        except BaseException:
            LOG.exception("Continuation interrupted or failed")
            save(batch_end=True)
            raise
        finally:
            await scorer.close()
            if engine:
                await engine.close()
            if kb:
                kb.close()
            if source_hashes != {name: file_sha(source / name) for name in source_hashes}:
                raise RuntimeError("Original source files changed during continuation.")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "runtime/final_validation_v3/benchmark")
    p.add_argument("--output", type=Path, default=ROOT / "runtime/context_recall_continuation_20261109")
    p.add_argument("--work-scope", choices=["first-round", "all"], default="first-round")
    p.add_argument("--total-scope", choices=["main", "all"], default="all")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--start-ollama", action="store_true", help="Start the frozen local Ollama service in this evaluation process tree if absent.")
    p.add_argument("--metric-timeout", type=float, default=900.0)
    p.add_argument("--score-keep-alive", default="10m")
    return p


if __name__ == "__main__":
    try:
        asyncio.run(execute(parser().parse_args()))
    except BaseException:
        LOG.exception("Fatal continuation error")
        raise
