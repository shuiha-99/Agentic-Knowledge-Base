"""Real-model acceptance run. Its small sample is never a benchmark claim."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agentic_kb.config import Settings
from agentic_kb.graph import GraphEngine
from agentic_kb.providers import build_providers
from agentic_kb.retrieval import KnowledgeBase


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--case", action="append", default=[], help="Only run the named case(s)")
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.output_reserve_tokens = 512
    settings.request_timeout_seconds = 300
    settings.task_timeout_seconds = 1800
    settings.run_timeout_seconds = 5400
    settings.rerank_top_k = 4
    settings.evidence_token_budget = 3500
    source_start = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path("src/agentic_kb").glob("*.py")}
    llm, embedder, reranker = build_providers(settings)
    kb = KnowledgeBase(settings, embedder, reranker, token_counter=llm.count_tokens)
    output = settings.data_dir / "verification" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={output}", flush=True)
    try:
        if not args.skip_index:
            started = time.monotonic()
            # This CLI phase performs no generation/reranking; keeping only the
            # embedder resident avoids reloading its weights once per document.
            settings.low_memory_mode = False
            for path in sorted(Path("data/demo_docs").glob("*.md")):
                record = kb.ingest([path])[0]
                print(f"INDEX {record.name} {record.parent_count}/{record.child_count}", flush=True)
            embedder._model = None
            gc.collect()
            settings.low_memory_mode = True
            print(f"INDEX_SECONDS={time.monotonic()-started:.2f}", flush=True)
        engine = GraphEngine(settings, llm, kb)
        shared = uuid4().hex
        cases = [("simple", "AtlasDesk v2 默认监听哪个端口？存活和就绪探针分别是什么？", shared, None)]
        if not args.quick:
            cases += [
                ("followup", "那旧版 v1 的端口和健康检查呢？", shared, None),
                ("complex", "比较 AtlasDesk v1 与 v2 的部署差异，说明 v1.8 到 v2.0 的配置迁移，以及恢复写入后如何回滚。", uuid4().hex, None),
                ("no_answer", "AtlasDesk 研发平台组员工2026年10月的平均薪资是多少？", uuid4().hex, None),
                ("clarification", "这个版本怎么升级？", uuid4().hex, "AtlasDesk 从 v1.8 升级至 v2.0。"),
            ]
        results = []
        for name, question, thread_id, clarification in cases:
            if args.case and name not in args.case:
                continue
            started = time.monotonic()
            print(f"CASE={name} QUESTION={question}", flush=True)
            events = []
            async def consume(question, clarification=None):
                async for event in engine.astream(question, thread_id, clarification=clarification):
                    events.append(event)
                    with (output / f"{name}.events.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                    if event["type"] in {"stage", "error", "clarification"}:
                        print(event, flush=True)
            await consume(question)
            paused = any(e["type"] == "clarification" for e in events)
            if paused and clarification:
                await consume(None, clarification)
            done = next((e for e in reversed(events) if e["type"] == "done"), None)
            result = {"case": name, "question": question, "thread_id": thread_id,
                      "elapsed_seconds": time.monotonic()-started, "clarification_seen": paused,
                      "done": done, "retrieval_trace": kb.last_trace,
                      "errors": [e for e in events if e["type"] == "error"]}
            results.append(result)
            (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"RESULT {name}: seconds={result['elapsed_seconds']:.1f} done={bool(done)}", flush=True)
            if done:
                print(done["answer"], flush=True)
        manifest = {"settings": settings.model_dump(mode="json"), "small_sample_only": True,
                    "documents": [d.model_dump() for d in kb.list_documents()],
                    "source_sha256": source_start,
                    "source_changed_during_run": source_start != {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path("src/agentic_kb").glob("*.py")}}
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        kb.close()


if __name__ == "__main__":
    asyncio.run(main())
