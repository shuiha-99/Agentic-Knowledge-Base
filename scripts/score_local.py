"""Score one recorded real answer without regenerating it or opening Qdrant.

This is a small-sample integration check, never a benchmark or a stable score.
Run from the project root after verify_local.py has finished using the models.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from agentic_kb.config import Settings
from agentic_kb.evaluation import EvaluationCase, _RagasScorer, _versions
from agentic_kb.providers import build_providers

METRICS = ("answer_correctness", "context_recall", "faithfulness")
REFERENCE = (
    "AtlasDesk v2 的 API 默认监听 8088 端口。"
    "/healthz 是存活探针，正常返回 HTTP 200；"
    "/readyz 是就绪探针，检查数据库和 Redis，任意依赖不可用时返回 HTTP 503。"
    "负载均衡只向 readiness 成功的实例发送流量。"
)
ANCHORS = [
    "API 默认监听 8088 端口",
    "`/healthz` 是存活探针，正常返回 HTTP 200",
    "`/readyz` 检查数据库与 Redis，任意依赖不可用时返回 HTTP 503",
    "负载均衡只向 readiness 成功的实例发流量",
]


def read_contexts(row: dict, results_path: Path) -> tuple[list[str], str, dict]:
    metrics = row["done"].get("metrics", {})
    if "generation_contexts" in metrics:
        contexts = metrics["generation_contexts"]
        if not isinstance(contexts, list) or any(not isinstance(text, str) for text in contexts):
            raise ValueError("generation_contexts 必须为字符串列表，不能用其他字段静默代替")
        return contexts, "done.metrics.generation_contexts", {
            "generation_context_kind": metrics.get("generation_context_kind", "not_recorded"),
            "evidence_ids": metrics.get("generation_context_evidence_ids", []),
        }

    records = [record for record in metrics.get("compression_records", [])
               if isinstance(record, dict) and record.get("stage") == "global_evidence"]
    if not records or any(record.get("triggered") is not False for record in records):
        raise ValueError(
            "旧结果缺少实际generation_contexts，且不能证明global_evidence未压缩；"
            "拒绝用模型可能未见的完整原文进行RAGAS评分。"
        )
    event_file = results_path.parent / f"{row['case']}.events.jsonl"
    if event_file.is_file():
        raw = event_file.read_bytes()
        sources = None
        for line in raw.decode("utf-8-sig").splitlines():
            event = json.loads(line)
            if event.get("type") == "sources":
                sources = event.get("sources", [])
        if sources is not None:
            return [source["text"] for source in sources], "reconstructed_sources_event_global_evidence_not_triggered", {
                "event_file": str(event_file), "event_file_sha256": hashlib.sha256(raw).hexdigest(),
                "global_evidence_records": records,
                "limitation": "旧结果未直接记录最终模型输入；仅在明确未压缩时重建来源文本。",
            }
    task_results = metrics.get("task_results", [])
    if isinstance(task_results, dict):
        task_results = list(task_results.values())
    sources = []
    seen = set()
    for result in task_results:
        for source in result.get("evidence", []):
            key = source.get("evidence_id") or source.get("parent_id") or source.get("text")
            if key not in seen:
                seen.add(key)
                sources.append(source)
    if not sources:
        raise ValueError("旧结果没有可重建来源的sources事件或task_results.evidence")
    return [source["text"] for source in sources], "reconstructed_task_evidence_global_evidence_not_triggered", {
        "global_evidence_records": records,
        "limitation": "旧结果未直接记录最终模型输入；仅在明确未压缩时按任务证据重建。",
    }


async def run(args) -> int:
    input_path = args.results.resolve()
    raw_input = input_path.read_bytes()
    records = json.loads(raw_input.decode("utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError("输入必须是verify_local.py生成的results.json数组")
    matched = [row for row in records if row.get("case") == args.case]
    if len(matched) != 1:
        raise ValueError(f"应恰好有一条case={args.case}，实际找到{len(matched)}条")
    row = matched[0]
    if not isinstance(row.get("done"), dict) or not str(row["done"].get("answer", "")).strip():
        raise ValueError("所选样例没有完成的非空答案，不重新生成或编造答案")
    answer = row["done"]["answer"]
    if "离线流程测试" in answer or "非模型回答" in answer:
        raise ValueError("输入为离线测试答案，禁止报告RAGAS质量分数")
    source_manifest = input_path.parent / "manifest.json"
    manifest_info = {}
    if source_manifest.is_file():
        manifest_raw = source_manifest.read_bytes()
        original_manifest = json.loads(manifest_raw.decode("utf-8-sig"))
        provider = original_manifest.get("settings", {}).get("provider")
        if provider and provider != "ollama":
            raise ValueError(f"原验收使用provider={provider}，不是实际Ollama模型问答")
        manifest_info = {"path": str(source_manifest), "sha256": hashlib.sha256(manifest_raw).hexdigest(),
                         "generation_settings": original_manifest.get("settings", {})}
    contexts, scope, context_evidence = read_contexts(row, input_path)
    if not contexts or not any(text.strip() for text in contexts):
        raise ValueError("没有实际或可证明未压缩的参考上下文，拒绝将空样例冒充完整评分验收")

    root = Path(__file__).resolve().parents[1]
    reference_path = root / "data" / "demo_docs" / "02_deploy_v2.md"
    reference_bytes = reference_path.read_bytes()
    reference_text = reference_bytes.decode("utf-8-sig")
    missing = [anchor for anchor in ANCHORS if anchor not in reference_text]
    if missing:
        raise ValueError(f"参考文档已变化，需人工复核参考答案。缺少锚点：{missing}")
    settings = Settings.from_env()
    if settings.provider != "ollama":
        raise ValueError("当前provider必须为ollama，禁止使用测试向量进行AnswerCorrectness评分")
    settings.request_timeout_seconds = args.request_timeout
    settings.run_timeout_seconds = args.metric_timeout
    settings.low_memory_mode = True
    _, embedder, _ = build_providers(settings)
    engine_view = SimpleNamespace(retriever=SimpleNamespace(embedder=embedder))
    case = EvaluationCase(id=f"local-{args.case}", question=row["question"], kind="single_document",
                          reference=REFERENCE, reference_documents=[reference_path.name],
                          review_status="reference_anchors_checked_for_this_single_integration_sample")

    # A unique sibling directory of the input file preserves every scoring attempt.
    output_dir = input_path.parent / f"score-{args.case}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    output_dir.mkdir(parents=False, exist_ok=False)
    sample = {"case": args.case, "question": row["question"], "response": answer,
              "reference": REFERENCE, "retrieved_contexts": contexts, "context_scope": scope,
              "context_evidence": context_evidence, "original_errors": row.get("errors", [])}
    (output_dir / "sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / reference_path.name).write_bytes(reference_bytes)
    print(f"OUTPUT={output_dir}", flush=True)
    started = time.monotonic()
    scorer = None
    scores = dict.fromkeys(METRICS)
    errors = {}
    try:
        scorer = _RagasScorer(settings, engine_view)
        values, errors = await scorer.score(case, answer, contexts)
        scores.update(values)
    except Exception as exc:
        errors["scoring"] = f"{type(exc).__name__}: {exc}"
    finally:
        if scorer:
            await scorer.close()
    report = {
        "created_at": datetime.now(UTC).isoformat(), "small_sample_only": True, "case_count": 1,
        "notice": "仅验证一条已记录真实问答的评分链路；不是总体成绩，不证明稳定达到任何项目指标。",
        "case": args.case, "input": {"path": str(input_path), "sha256": hashlib.sha256(raw_input).hexdigest(),
                                      "original_manifest": manifest_info},
        "reference_evidence": {"path": str(reference_path), "sha256": hashlib.sha256(reference_bytes).hexdigest(),
                               "anchors": ANCHORS, "reference": REFERENCE},
        "context_scope": scope, "context_evidence": context_evidence,
        "scores": scores, "errors": errors, "elapsed_seconds": time.monotonic() - started,
        "judge": {"model": scorer.judge_model if scorer else None,
                  "base_url": scorer.judge_base_url if scorer else None,
                  "temperature": 0, "answer_correctness_weights": [0.75, 0.25],
                  "worker_versions": scorer.versions if scorer else {},
                  "worker_python": str(scorer.python) if scorer else None,
                  "worker_path": str(scorer.worker_path) if scorer else None,
                  "config": scorer.config if scorer else None},
        "application_versions": _versions(),
        "embedding_model": settings.embedding_model,
        "embedding_fingerprint": getattr(embedder, "fingerprint", None),
    }
    (output_dir / "score.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"small_sample_only": True, "scores": scores, "errors": errors,
                      "elapsed_seconds": report["elapsed_seconds"], "report": str(output_dir / "score.json")}, ensure_ascii=False, indent=2), flush=True)
    return 1 if errors or any(value is None for value in scores.values()) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只评分既有真实问答，不重复生成；一条样例不代表整体效果")
    parser.add_argument("results", type=Path, help="verify_local.py输出的results.json")
    parser.add_argument("--case", choices=["simple"], default="simple")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--metric-timeout", type=float, default=900.0)
    try:
        sys.exit(asyncio.run(run(parser.parse_args())))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"拒绝评分：{exc}", file=sys.stderr)
        sys.exit(2)
