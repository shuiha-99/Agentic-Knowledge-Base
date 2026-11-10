"""Evaluation must retain failures and keep workflow tests out of quality claims."""
import json
import sys
from types import SimpleNamespace

import pytest

from agentic_kb.evaluation import (
    EvaluationCase,
    evaluate_dataset,
    family_split,
    load_dataset,
    retrieval_metrics,
)


def write_dataset(path, cases):
    path.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases), encoding="utf-8")
    return path


def test_dataset_rejects_family_leakage_and_duplicate_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    write_dataset(path, [{"id": "a", "question": "a", "family": "same", "split": "dev"}, {"id": "b", "question": "b", "family": "same", "split": "test"}])
    with pytest.raises(ValueError, match="可能泄漏"):
        load_dataset(path)
    write_dataset(path, [{"id": "a", "question": "a"}, {"id": "a", "question": "b"}])
    with pytest.raises(ValueError, match="重复题目"):
        load_dataset(path)


def test_document_recall_deduplicates_sources():
    case = EvaluationCase(id="x", question="q", reference_documents=["a.md", "b.md"])
    result = retrieval_metrics(case, [{"document_name": "z.md"}, {"document_name": "a.md"}, {"document_name": "a.md"}])
    assert result == {"document_recall_at_5": .5, "document_mrr": .5}
    assert retrieval_metrics(case, [])["document_recall_at_5"] == 0
    assert retrieval_metrics(EvaluationCase(id="z", question="q"), [])["document_mrr"] is None


def test_family_split_is_deterministic():
    assert family_split("v2-deployment") == family_split("v2-deployment")
    assert {family_split(str(i)) for i in range(100)} == {"dev", "test"}


def test_browser_session_factory_does_not_share_thread_ids(tmp_path):
    from agentic_kb.ui import _Conversations

    store = _Conversations(tmp_path / "ui.sqlite")
    def factory():
        return store.get(None)
    first, second = factory(), factory()
    assert first["thread_id"] != second["thread_id"]
    store.put(first)
    first["history"].append({"role": "user", "content": "private"})
    assert second["history"] == []


@pytest.mark.asyncio
async def test_gradio_process_api_initial_load_creates_distinct_concrete_sessions(tmp_path):
    """Exercise Gradio's preprocessing/state initialization, not just a Python factory."""
    import gradio as gr
    from gradio.state_holder import SessionState

    from agentic_kb.ui import _Conversations, create_app

    app = create_app(SimpleNamespace(data_dir=tmp_path, provider="offline-test"),
                     SimpleNamespace(list_documents=lambda: []), SimpleNamespace())
    load_id = next(key for key, fn in app.fns.items() if fn.fn and fn.fn.__name__ == "initialize_conversation")
    state_id = next(key for key, block in app.blocks.items() if isinstance(block, gr.State))
    first, second = SessionState(app), SessionState(app)
    assert first[state_id] is None and second[state_id] is None
    first_result = await app.process_api(load_id, [None], state=first, session_hash="browser-first")
    second_result = await app.process_api(load_id, [None], state=second, session_hash="browser-second")
    assert not first_result.get("error") and not second_result.get("error")
    assert isinstance(first[state_id], dict) and isinstance(second[state_id], dict)
    first_id, second_id = first[state_id]["thread_id"], second[state_id]["thread_id"]
    assert first_id and second_id and first_id != second_id
    assert first_result["data"][2] == first_id
    assert second_result["data"][2] == second_id
    store = _Conversations(tmp_path / "ui_sessions.sqlite")
    assert store.get(first_id)["thread_id"] == first_id
    assert store.get(second_id)["thread_id"] == second_id
    # Corrupt/legacy callable state is normalized without ever being executed.
    first[state_id] = lambda: (_ for _ in ()).throw(AssertionError("must not invoke callable"))
    repaired = await app.process_api(load_id, [None], state=first, session_hash="browser-first")
    assert isinstance(first[state_id], dict)
    assert repaired["data"][2] == first[state_id]["thread_id"]


@pytest.mark.asyncio
async def test_gradio_process_api_document_load_returns_concrete_outputs(tmp_path):
    from gradio.state_holder import SessionState

    from agentic_kb.ui import create_app

    app = create_app(SimpleNamespace(data_dir=tmp_path, provider="offline-test"),
                     SimpleNamespace(list_documents=lambda: []), SimpleNamespace())
    refresh_id = next(key for key, fn in app.fns.items() if fn.fn and fn.fn.__name__ == "refresh_documents")
    result = await app.process_api(refresh_id, [], state=SessionState(app), session_hash="document-load")
    assert not result.get("error")
    assert result["data"][0]["data"] == []


@pytest.mark.asyncio
async def test_gradio_chatbot_clear_rotates_thread_and_discards_active_context(tmp_path):
    import gradio as gr
    from gradio.state_holder import SessionState

    from agentic_kb.ui import _Conversations, create_app

    class CaptureEngine:
        def __init__(self):
            self.calls = []

        async def astream(self, question, thread_id, clarification=None, mode="agent"):
            self.calls.append((question, thread_id, clarification))
            yield {"type": "done", "answer": "fresh reply", "metrics": {}}

    engine = CaptureEngine()
    app = create_app(SimpleNamespace(data_dir=tmp_path, provider="offline-test"),
                     SimpleNamespace(list_documents=lambda: []), engine)
    state_id = next(key for key, block in app.blocks.items() if isinstance(block, gr.State))
    chatbot_id = next(key for key, block in app.blocks.items() if isinstance(block, gr.Chatbot))
    clear_dependency = next(dependency for dependency in app.get_config_file()["dependencies"]
                            if any(target[0] == chatbot_id and target[1] == "clear" for target in dependency["targets"]))
    clear_id = clear_dependency["id"]
    assert app.fns[clear_id].fn.__name__ == "new_conversation"
    session = SessionState(app)
    store = _Conversations(tmp_path / "ui_sessions.sqlite")
    old = store.get(None)
    old.update(history=[{"role": "user", "content": "remember v1"}], pending=True,
               clarification="which version?", sources="old sources", stages="old stages", metrics={"old": True})
    store.put(old)
    session[state_id] = old
    result = await app.process_api(clear_id, [], state=session, session_hash="clear-browser")
    fresh = session[state_id]
    assert fresh["thread_id"] != old["thread_id"]
    assert result["data"][2] == fresh["thread_id"]
    assert fresh["history"] == [] and fresh["pending"] is False
    assert fresh["clarification"] == fresh["sources"] == fresh["stages"] == ""
    assert fresh["metrics"] == {}
    respond_id = next(key for key, fn in app.fns.items() if fn.fn and fn.fn.__name__ == "respond")
    response = await app.process_api(respond_id, ["new question", None, "agent"], state=session, session_hash="clear-browser")
    for _ in range(6):
        if not response["is_generating"]:
            break
        response = await app.process_api(respond_id, ["new question", None, "agent"], state=session,
                                         session_hash="clear-browser", iterator=response["iterator"])
    assert not response["is_generating"]
    assert engine.calls == [("new question", fresh["thread_id"], None)]


@pytest.mark.asyncio
async def test_ui_resume_error_keeps_clarification_pending(tmp_path):
    from agentic_kb.ui import _Conversations, create_app

    class FailingResume:
        def __init__(self):
            self.calls = []

        async def astream(self, question, thread_id, clarification=None, mode="agent"):
            self.calls.append((question, clarification))
            yield {"type": "error", "text": "检查点暂时不可用"}

    engine = FailingResume()
    kb = SimpleNamespace(list_documents=lambda: [])
    app = create_app(SimpleNamespace(data_dir=tmp_path, provider="offline-test"), kb, engine)
    callback = next(fn.fn for fn in app.fns.values() if fn.fn and fn.fn.__name__ == "respond")
    store = _Conversations(tmp_path / "ui_sessions.sqlite")
    state = store.get(None)
    state.update(pending=True, clarification="哪个版本？")
    for answer in ("v2", "v2再试一次"):
        async for update in callback(answer, state, "agent"):
            state = update[1]
        assert state["pending"] is True
        assert store.get(state["thread_id"])["pending"] is True
    assert engine.calls == [(None, "v2"), (None, "v2再试一次")]


class ScriptedEngine:
    def __init__(self):
        self.pending = set()
        self.calls = []

    async def astream(self, question, thread_id, clarification=None, mode="agent"):
        self.calls.append((question, thread_id, clarification, mode))
        if question == "fail":
            raise RuntimeError("检索不可用")
        if question == "ambiguous":
            self.pending.add(thread_id)
            yield {"type": "clarification", "question": "哪个版本？"}
            return
        if question is None:
            assert thread_id in self.pending
            assert clarification == "v2"
        yield {"type": "stage", "text": "检索"}
        yield {"type": "sources", "sources": [{"document_name": "a.md", "text": "answer evidence"}]}
        yield {"type": "token", "text": "真实"}
        yield {"type": "token", "text": "流"}
        yield {"type": "done", "answer": "真实流", "metrics": {"context_tokens_before": 100, "context_tokens_after": 60}}


def settings():
    return SimpleNamespace(provider="offline-test", run_timeout_seconds=1, api_key="not-exported")


@pytest.mark.asyncio
async def test_evaluation_keeps_failure_rows_and_separates_ambiguous(tmp_path):
    path = write_dataset(tmp_path / "cases.jsonl", [
        {"id": "good", "question": "ok", "reference": "answer", "reference_documents": ["a.md"]},
        {"id": "bad", "question": "fail", "reference": "answer", "reference_documents": ["a.md"]},
        {"id": "pending", "question": "ambiguous", "kind": "ambiguous"},
        {"id": "resumed", "question": "ambiguous", "kind": "ambiguous", "clarification": "v2"},
    ])
    engine = ScriptedEngine()
    result = await evaluate_dataset(settings(), engine, path, tmp_path / "report")
    rows = [json.loads(line) for line in (tmp_path / "report/results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert result["statuses"] == {"success": 2, "clarification_required": 1, "incomplete": 0, "timeout": 0, "error": 1}
    assert result["retrieval"]["document_recall_at_5"]["mean"] == .5
    assert result["compression"]["token_weighted_ratio"] == .6
    assert result["quality_evaluation"] is False
    assert rows[1]["error"] == "检索不可用"
    assert rows[2]["ragas_eligible"] is False
    resume_calls = [call for call in engine.calls if call[0] is None]
    assert len(resume_calls) == 1
    assert resume_calls[0][1] == rows[3]["thread_id"]
    manifest = json.loads((tmp_path / "report/manifest.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["api_key"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_offline_cannot_report_ragas(tmp_path):
    path = write_dataset(tmp_path / "cases.jsonl", [{"id": "x", "question": "q"}])
    with pytest.raises(ValueError, match="禁止生成"):
        await evaluate_dataset(settings(), ScriptedEngine(), path, tmp_path / "report", use_ragas=True)


@pytest.mark.asyncio
async def test_output_is_not_overwritten(tmp_path):
    path = write_dataset(tmp_path / "cases.jsonl", [{"id": "x", "question": "ok"}])
    await evaluate_dataset(settings(), ScriptedEngine(), path, tmp_path / "report")
    with pytest.raises(ValueError, match="已有"):
        await evaluate_dataset(settings(), ScriptedEngine(), path, tmp_path / "report")


@pytest.mark.asyncio
async def test_warmup_turns_use_same_thread_and_default_test_split(tmp_path):
    path = write_dataset(tmp_path / "cases.jsonl", [
        {"id": "dev", "question": "fail", "split": "dev"},
        {"id": "long", "question": "followup", "turns": [{"question": "setup"}]},
    ])
    engine = ScriptedEngine()
    report = await evaluate_dataset(settings(), engine, path, tmp_path / "report")
    assert report["case_count"] == 1
    assert [call[0] for call in engine.calls] == ["setup", "followup"]
    assert engine.calls[0][1] == engine.calls[1][1]


@pytest.mark.asyncio
async def test_ragas_process_protocol_reuses_worker_and_closes(tmp_path, monkeypatch):
    """Transport-only test; the dummy worker does not assess answer quality."""
    from agentic_kb.evaluation import _RagasScorer

    worker = tmp_path / "protocol_worker.py"
    worker.write_text("import json,sys\nfor line in sys.stdin:\n request=json.loads(line)\n assert request['sample']['retrieved_contexts']==['actual compressed context']\n print(json.dumps({'scores':{'answer_correctness':0.5},'errors':{},'versions':{'transport':'test-only'}}),flush=True)\n", encoding="utf-8")
    monkeypatch.setenv("KB_RAGAS_PYTHON", sys.executable)
    monkeypatch.setenv("KB_RAGAS_WORKER_PATH", str(worker))
    embedder = SimpleNamespace(encode=lambda texts: [SimpleNamespace(dense=[1., 0.]) for _ in texts])
    engine = SimpleNamespace(retriever=SimpleNamespace(embedder=embedder))
    config = SimpleNamespace(chat_model="unused", ollama_base_url="http://localhost:11434", context_window_tokens=4096, output_reserve_tokens=512, request_timeout_seconds=1, run_timeout_seconds=1)
    scorer = _RagasScorer(config, engine)
    case = EvaluationCase(id="transport", question="q", reference="ref")
    try:
        scores, errors = await scorer.score(case, "answer", ["actual compressed context"], metric_names=["answer_correctness"])
        first_pid = scorer.process.pid
        assert scores == {"answer_correctness": .5} and errors == {}
        await scorer.score(case, "answer", ["actual compressed context"], metric_names=["answer_correctness"])
        assert scorer.process.pid == first_pid
    finally:
        await scorer.close()
    assert scorer.process is None and scorer.stderr_task is None


class DurableEngine:
    def __init__(self):
        self.calls = []

    async def astream(self, question, thread_id, clarification=None, mode="agent"):
        self.calls.append((question, thread_id))
        yield {"type": "sources", "sources": [{"document_name": "raw.md", "text": "full source not sent"}]}
        yield {"type": "done", "answer": "answer:" + question,
               "metrics": {"generation_contexts": ["actual compressed context"]}}


class DurableScorer:
    """Transport/resume fixture only; these constants are not quality scores."""
    def __init__(self, *, allow_encoding=False):
        self.calls = []
        self.encode_calls = []
        self.allow_encoding = allow_encoding
        self.closed = 0
        self.versions = {"transport": "test-only"}

    async def prepare_embeddings(self, reference, answer):
        assert self.allow_encoding, "score phase must never encode"
        self.encode_calls.append((reference, answer))
        return {reference: [1., 0.], answer: [0., 1.]}

    async def score(self, case, answer, contexts, *, metric_names, embeddings):
        assert contexts == ["actual compressed context"]
        assert embeddings[answer] == [0., 1.]
        self.calls.append((case.id, metric_names[0]))
        return {metric_names[0]: .5}, {}

    async def close(self):
        self.closed += 1


def durable_cases():
    return [EvaluationCase(id="a", question="alpha", reference="ref-a", kind="single_document"),
            EvaluationCase(id="b", question="beta", reference="ref-b", kind="cross_document")]


@pytest.mark.asyncio
async def test_benchmark_resume_freeze_and_three_repeats_without_regeneration(tmp_path):
    from agentic_kb.evaluation import BenchmarkRun

    environment = {"model_digest": "frozen-model", "code_sha": "frozen-code"}
    run = BenchmarkRun(tmp_path, durable_cases(), environment, stability_ids=["a"])
    engine = DurableEngine()
    await run.generate(engine, limit=1)
    assert len(engine.calls) == 1
    resumed = BenchmarkRun(tmp_path, durable_cases(), environment, stability_ids=["a"])
    await resumed.generate(engine)
    assert len(engine.calls) == 4  # two main cases + two extra executions of a
    await resumed.generate(engine)
    assert len(engine.calls) == 4
    assert len({thread for _, thread in engine.calls}) == 4
    with pytest.raises(ValueError, match="冻结清单不匹配"):
        BenchmarkRun(tmp_path, durable_cases(), {**environment, "model_digest": "changed"}, stability_ids=["a"])
    with pytest.raises(ValueError, match="冻结清单不匹配"):
        BenchmarkRun(tmp_path, durable_cases(), environment, stability_ids=["b"])
    changed = durable_cases()
    changed[0].reference = "changed answer"
    with pytest.raises(ValueError, match="冻结清单不匹配"):
        BenchmarkRun(tmp_path, changed, environment, stability_ids=["a"])


@pytest.mark.asyncio
async def test_benchmark_metric_commits_resume_and_precomputed_vector_gate(tmp_path):
    from agentic_kb.evaluation import BenchmarkRun

    run = BenchmarkRun(tmp_path, durable_cases(), {"version": 1}, stability_ids=["a"])
    await run.generate(DurableEngine())
    missing = DurableScorer()
    with pytest.raises(ValueError, match="先完成prepare-vectors"):
        await run.score(missing)
    assert missing.calls == [] and missing.closed == 1
    vectors = DurableScorer(allow_encoding=True)
    await run.prepare_vectors(vectors)
    assert len(vectors.encode_calls) == 2  # identical texts across repeats reuse deterministic vectors
    resumed = BenchmarkRun(tmp_path, durable_cases(), {"version": 1}, stability_ids=["a"])
    cached = DurableScorer(allow_encoding=True)
    await resumed.prepare_vectors(cached)
    assert cached.encode_calls == []
    first = DurableScorer()
    await resumed.score(first, limit=1)
    assert first.calls == [("a", "answer_correctness")]
    second = DurableScorer()
    report = await resumed.score(second)
    assert len(first.calls) + len(second.calls) == 12
    assert second.calls[:3] == [("b", "answer_correctness"), ("a", "answer_correctness"), ("a", "answer_correctness")]
    assert report["main"]["planned_generations"] == 2
    assert report["main"]["metrics"]["answer_correctness"]["eligible_count"] == 2
    assert report["stability"]["per_case"]["a"]["planned_generations"] == 3
    assert report["stability"]["per_case"]["a"]["metrics"]["faithfulness"]["valid_count"] == 3
    final = DurableScorer()
    await resumed.score(final)
    assert final.calls == []


@pytest.mark.asyncio
async def test_benchmark_partial_tail_recovery_and_durable_done_event(tmp_path):
    from agentic_kb.evaluation import BenchmarkRun, canonical_hash

    cases = durable_cases()[:1]
    run = BenchmarkRun(tmp_path, cases, {}, stability_repeats=1, stability_ids=[])
    item = run.item_id("a", 0)
    generation_key = f"generation:{item}:attempt1"
    turn_key = generation_key + ":turn0"
    run.journal.begin(generation_key)
    run.journal.begin(turn_key)
    events = [{"type": "done", "answer": "persisted answer",
               "metrics": {"generation_contexts": ["actual compressed context"]}}]
    event_path = tmp_path / "artifacts" / (canonical_hash(turn_key) + ".events.jsonl")
    event_path.write_text(json.dumps(events[0]) + "\n", encoding="utf-8")
    with (tmp_path / "journal.jsonl").open("ab") as stream:
        stream.write(b'{"key":"not_committed')
    recovered = BenchmarkRun(tmp_path, cases, {}, stability_repeats=1, stability_ids=[])
    engine = DurableEngine()
    report = await recovered.generate(engine)
    assert engine.calls == []
    assert report["main"]["statuses"] == {"success": 1}
    assert recovered._latest("generation", item)["answer"] == "persisted answer"
    assert list(tmp_path.glob("journal.jsonl.uncommitted-tail-*"))


@pytest.mark.asyncio
async def test_benchmark_inflight_metric_is_retained_and_not_silently_repeated(tmp_path):
    from agentic_kb.evaluation import BenchmarkRun

    run = BenchmarkRun(tmp_path, durable_cases()[:1], {}, stability_repeats=1, stability_ids=[])
    await run.generate(DurableEngine())
    await run.prepare_vectors(DurableScorer(allow_encoding=True))
    item = run.item_id("a", 0)
    generation_hash = run._latest("generation", item)["generation_hash"]
    run.journal.begin(f"score:{item}:{generation_hash}:answer_correctness:attempt1")
    scorer = DurableScorer()
    report = await run.score(scorer)
    assert scorer.calls == [("a", "context_recall"), ("a", "faithfulness")]
    metric = report["main"]["metrics"]["answer_correctness"]
    assert metric["eligible_count"] == 1 and metric["failed_or_blocked_count"] == 1
    assert metric["valid_count"] == 0
    retry = DurableScorer()
    report = await run.score(retry, retry_failed=True)
    assert retry.calls == [("a", "answer_correctness")]
    assert report["main"]["metrics"]["answer_correctness"]["valid_count"] == 1
    attempts = [row for row in run.journal.completed.values() if row.get("phase") == "score" and row.get("metric") == "answer_correctness"]
    assert len(attempts) == 2 and {row["status"] for row in attempts} == {"interrupted", "success"}
    assert report["main"]["metrics"]["answer_correctness"]["historical_failed_attempt_count"] == 1


@pytest.mark.asyncio
async def test_benchmark_generation_failure_stays_in_eligible_denominator(tmp_path):
    from agentic_kb.evaluation import BenchmarkRun

    class FailedEngine:
        async def astream(self, *args, **kwargs):
            yield {"type": "error", "text": "expected provider failure"}

    run = BenchmarkRun(tmp_path, durable_cases()[:1], {}, stability_repeats=1, stability_ids=[])
    await run.generate(FailedEngine())
    scorer = DurableScorer()
    report = await run.score(scorer)
    assert scorer.calls == []
    assert report["main"]["statuses"] == {"error": 1}
    for metric in report["main"]["metrics"].values():
        assert metric["eligible_count"] == metric["failed_or_blocked_count"] == 1
        assert metric["mean_valid_only"] is None


@pytest.mark.asyncio
async def test_worker_metric_selection_does_not_call_other_metrics():
    import runpy
    from pathlib import Path

    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "ragas_worker.py"))
    calls = []
    client_options = {}
    initialized = {}

    def client(**kwargs):
        client_options.update(kwargs)
        return SimpleNamespace(**kwargs)

    def wrapper(value, **kwargs):
        return SimpleNamespace(value=value, **kwargs)

    class Metric:
        def __init__(self, name, **kwargs):
            self.name = name
            self.options = kwargs

        def init(self, run_config):
            initialized[self.name] = {"options": self.options, "run_config": run_config}

        async def single_turn_ascore(self, sample):
            calls.append(self.name)
            return .5

    function = namespace["score"]
    loaded = (type("Embeddings", (), {}), client, lambda **kwargs: kwargs, wrapper, wrapper,
              lambda **kwargs: Metric("answer_correctness", **kwargs),
              lambda **kwargs: Metric("faithfulness", **kwargs),
              lambda **kwargs: Metric("context_recall", **kwargs), lambda **kwargs: SimpleNamespace(**kwargs))
    config = {"model": "test-only", "base_url": "http://127.0.0.1:11435", "context_window_tokens": 4096,
              "output_tokens": 2048, "request_timeout_seconds": 30, "metric_timeout_seconds": 1,
              "score_keep_alive": "10m"}
    result = await function({"config": config, "embeddings": {},
                             "sample": {}, "metrics": ["context_recall"]}, loaded)
    assert calls == ["context_recall"]
    assert result["scores"] == {"context_recall": .5}
    assert client_options["client_kwargs"] == {"timeout": 30, "trust_env": False}
    assert client_options["keep_alive"] == "10m" and client_options["temperature"] == 0
    assert client_options["num_ctx"] == 4096 and client_options["num_predict"] == 2048
    assert initialized["answer_correctness"]["options"]["weights"] == [.75, .25]
    assert set(initialized) == {"answer_correctness", "context_recall", "faithfulness"}
    for entry in initialized.values():
        assert vars(entry["run_config"]) == {"timeout": 30, "max_retries": 1, "max_workers": 1}


def test_benchmark_exclusive_os_lock_blocks_second_writer(tmp_path):
    import runpy
    from pathlib import Path

    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"))
    lock = module["exclusive_run"]
    with lock(tmp_path):
        with pytest.raises(RuntimeError, match="拒绝并发计分"):
            with lock(tmp_path):
                raise AssertionError("second writer must not enter")
    with lock(tmp_path):
        pass


@pytest.mark.asyncio
async def test_ablation_uses_real_qdrant_modes_top5_and_expands_parent_sources(tmp_path):
    """Actual KB/Qdrant integration with explicit test vectors, no quality claim."""
    from agentic_kb.config import Settings
    from agentic_kb.evaluation import BenchmarkRun
    from agentic_kb.providers import HashingTestEmbedder, LexicalTestReranker
    from agentic_kb.retrieval import KnowledgeBase

    settings = Settings(data_dir=tmp_path / "kb", provider="offline-test", parent_chunk_size=400,
                        parent_chunk_overlap=30, parent_min_size=80, child_chunk_size=80,
                        child_chunk_overlap=10, rerank_top_k=2, dense_top_k=64, sparse_top_k=64,
                        evidence_token_budget=5000)
    paths = []
    for index in range(7):
        path = tmp_path / f"deployment-{index}.md"
        path.write_text(f"# Deployment {index}\n\nRelease configuration app{index}: port 8088; rollout requires migration signed. "
                        f"Rollback app{index} requires snapshot verification before deployment. "
                        f"Deployment app{index} readiness checks database and redis. "
                        f"This final context identifies owner team{index} and generation g{index}.", encoding="utf-8")
        paths.append(path)
    kb = KnowledgeBase(settings, HashingTestEmbedder(), LexicalTestReranker())
    try:
        documents = kb.ingest(paths)
        assert len(documents) == 7 and all(row.parent_count == 1 and row.child_count > 1 for row in documents)
        case = EvaluationCase(id="real-kb-contract", question="deployment rollback readiness", reference_documents=[paths[0].name])
        run = BenchmarkRun(tmp_path / "benchmark", [case], {"protocol_only": "offline-test"},
                           stability_repeats=1, stability_ids=[])
        report = await run.retrieval_ablation(kb)
        rows = {row["mode"]: row for row in run.journal.completed.values() if row.get("phase") == "retrieval"}
        assert set(rows) == {"dense", "hybrid", "rerank", "parent"}
        for label, row in rows.items():
            assert row["status"] == "success" and row["error"] is None
            assert row["requested_k"] == 5 and len(row["sources"]) == 5  # configured default is only two
            assert row["search_mode"] == ("full" if label == "parent" else label)
            assert report["retrieval_ablation"][label]["attempted_count"] == 1
            assert report["retrieval_ablation"][label]["failed_count"] == 0
        assert kb.last_trace["mode"] == "full" and not kb.last_trace["reranker_fallback"]
        for label in ("dense", "hybrid", "rerank"):
            for source in rows[label]["sources"]:
                parent = kb.store.get_parent(source["parent_id"])
                assert source["evidence_id"] != parent.parent_id
                assert source["child_ids"] == [source["evidence_id"]]
                assert source["text"] in parent.text and len(source["text"]) < len(parent.text)
        for source in rows["parent"]["sources"]:
            parent = kb.store.get_parent(source["parent_id"])
            assert source["evidence_id"] == parent.parent_id
            assert source["text"] == source["raw_text"] == parent.text
            assert len(source["text"]) > settings.child_chunk_size
            assert len(source["child_ids"]) > 1
        expanded = rows["parent"]["sources"][0]
        child = next(source for source in rows["rerank"]["sources"] if source["parent_id"] == expanded["parent_id"])
        assert child["text"] in expanded["text"] and child["text"] != expanded["text"]
    finally:
        kb.close()


@pytest.mark.parametrize("selected,scope,label,count", [
    (["a", "b"], "full_dataset", "锁定完整题集", 6),
    (["a"], "subset", "锁定小集", 3),
    ([], "disabled", "未选择题目", 0),
])
def test_benchmark_report_describes_actual_stability_scope(tmp_path, selected, scope, label, count):
    from agentic_kb.evaluation import BenchmarkRun

    run = BenchmarkRun(tmp_path, durable_cases(), {}, stability_ids=selected)
    report = run.write_report()
    assert report["stability"]["scope"] == scope
    assert report["stability"]["selected_case_count"] == len(selected)
    assert report["stability"]["dataset_case_count"] == 2
    assert report["stability"]["aggregate"]["planned_generations"] == count
    assert report["main"]["planned_generations"] == 2
    assert label in report["notice"] and "计划不等于已完成" in report["notice"]
    assert report["stability"]["repeat0_reused_from_main"] is True
    assert run.manifest["frozen"]["stability_selection"].startswith("explicit case IDs")


def test_benchmark_freezes_helper_digest_tokenizer_revision_and_routing(tmp_path, monkeypatch):
    """Metadata-only stubs: no Ollama request or model load occurs in this test."""
    import runpy
    import sqlite3
    from pathlib import Path

    from agentic_kb.config import Settings
    from agentic_kb.evaluation import canonical_hash

    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"))
    snapshot = module["environment_snapshot"]
    settings = Settings(data_dir=tmp_path / "data", models_cache_dir=tmp_path / "models",
                        structured_model="qwen2.5:3b", structured_tokenizer_model="Qwen/Qwen2.5-3B-Instruct",
                        structured_model_schemas=" ConversationSummary, EvidenceDigest ")
    settings.data_dir.mkdir()
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("frozen dataset", encoding="utf-8")
    default_worker = tmp_path / "scripts" / "ragas_worker.py"
    default_worker.parent.mkdir()
    default_worker.write_text("raise AssertionError('metadata freeze must never execute worker')", encoding="utf-8")
    document = tmp_path / "document.md"
    document.write_text("frozen document", encoding="utf-8")
    with sqlite3.connect(settings.data_dir / "documents.sqlite") as connection:
        connection.execute("CREATE TABLE documents (doc_id TEXT, data TEXT)")
        connection.execute("INSERT INTO documents VALUES (?, ?)", ("doc", json.dumps({
            "path": str(document), "name": document.name, "sha256": module["file_sha"](document)})))
    hf_names = [settings.embedding_model, settings.reranker_model, settings.tokenizer_model,
                settings.structured_tokenizer_model]
    for index, name in enumerate(hf_names):
        cache = settings.models_dir / ("models--" + name.replace("/", "--"))
        revision = "revision-" + str(index)
        (cache / "snapshots" / revision).mkdir(parents=True)
        (cache / "snapshots" / revision / "config.json").write_text("{}", encoding="utf-8")
        (cache / "refs").mkdir()
        (cache / "refs" / "main").write_text(revision, encoding="utf-8")
    tags = [{"name": settings.chat_model, "digest": "main-digest", "size": 7},
            {"name": settings.structured_model, "digest": "helper-digest", "size": 3}]
    calls = []
    server_versions = {settings.ollama_base_url: "0.13.0-test"}

    def get_metadata(url, **kwargs):
        calls.append(url)
        if url.endswith("/api/version"):
            payload = {"version": server_versions[url.removesuffix("/api/version")]}
        else:
            assert url.endswith("/api/tags")
            payload = {"models": tags}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    monkeypatch.delenv("KB_EVAL_MODEL", raising=False)
    monkeypatch.delenv("KB_EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("KB_RAGAS_WORKER_PATH", raising=False)
    monkeypatch.delenv("KB_RAGAS_WORKER", raising=False)
    monkeypatch.setattr(module["httpx"], "get", get_metadata)
    monkeypatch.setattr(module["subprocess"], "run", lambda *a, **kw: SimpleNamespace(stdout='{"ragas":"0.3.3"}'))
    monkeypatch.setattr(module["importlib"].metadata, "distributions", lambda: [])
    first = snapshot(settings, tmp_path, dataset)
    helper_key = first["ollama_model_roles"]["structured"]
    assert first["ollama_models"][helper_key]["digest"] == "helper-digest"
    assert first["hf_models"][settings.structured_tokenizer_model]["snapshot_revisions"] == ["revision-3"]
    assert first["hf_models"][settings.structured_tokenizer_model]["refs"] == {"main": "revision-3"}
    assert first["structured_routing"]["schema_allowlist"] == ["ConversationSummary", "EvidenceDigest"]
    assert first["judge"]["worker"] == {"path": str(default_worker.resolve()), "sha256": module["file_sha"](default_worker)}
    assert first["ollama_servers"] == {settings.ollama_base_url: {"version": "0.13.0-test"}}
    assert len(calls) == 3  # one version; chat and judge share identity, helper is distinct.
    legacy_worker = tmp_path / "legacy_worker.py"
    legacy_worker.write_text("legacy worker, never executed", encoding="utf-8")
    override_worker = tmp_path / "override_worker.py"
    override_worker.write_text("explicit worker, never executed", encoding="utf-8")
    monkeypatch.setenv("KB_RAGAS_WORKER", str(legacy_worker))
    assert snapshot(settings, tmp_path, dataset)["judge"]["worker"]["path"] == str(legacy_worker.resolve())
    monkeypatch.setenv("KB_RAGAS_WORKER_PATH", str(override_worker))
    override_snapshot = snapshot(settings, tmp_path, dataset)
    assert override_snapshot["judge"]["worker"] == {"path": str(override_worker.resolve()), "sha256": module["file_sha"](override_worker)}
    override_worker.write_text("changed actual worker", encoding="utf-8")
    assert canonical_hash(snapshot(settings, tmp_path, dataset)) != canonical_hash(override_snapshot)
    monkeypatch.setenv("KB_RAGAS_WORKER_PATH", str(tmp_path / "missing.py"))
    with pytest.raises(ValueError, match="worker文件不存在"):
        snapshot(settings, tmp_path, dataset)
    monkeypatch.delenv("KB_RAGAS_WORKER_PATH")
    monkeypatch.delenv("KB_RAGAS_WORKER")
    server_versions[settings.ollama_base_url] = "0.14.0-test"
    assert canonical_hash(snapshot(settings, tmp_path, dataset)) != canonical_hash(first)
    monkeypatch.setenv("KB_EVAL_BASE_URL", "http://judge.example.test")
    server_versions["http://judge.example.test"] = "0.15.0-test"
    assert snapshot(settings, tmp_path, dataset)["ollama_servers"] == {
        settings.ollama_base_url: {"version": "0.14.0-test"}, "http://judge.example.test": {"version": "0.15.0-test"}}
    monkeypatch.delenv("KB_EVAL_BASE_URL")
    tags[1]["digest"] = "changed-helper"
    assert canonical_hash(snapshot(settings, tmp_path, dataset)) != canonical_hash(first)
    tags.pop()
    with pytest.raises(ValueError, match="实际digest"):
        snapshot(settings, tmp_path, dataset)
    tags.append({"name": settings.structured_model, "digest": "helper-digest", "size": 3})
    tokenizer_cache = settings.models_dir / ("models--" + settings.structured_tokenizer_model.replace("/", "--"))
    (tokenizer_cache / "snapshots").rename(tokenizer_cache / "not-prepared")
    with pytest.raises(ValueError, match="模型缓存尚未准备"):
        snapshot(settings, tmp_path, dataset)
