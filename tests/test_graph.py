"""Behavioral tests execute real LangGraph graphs and SQLite checkpoints.

The scripted provider and retriever make no claims about model answer quality.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agentic_kb.config import Settings
from agentic_kb.graph import EvidenceReview, GraphEngine, merge_by_task_id
from agentic_kb.memory import BudgetMemory, ConversationSummary
from agentic_kb.models import Evidence, RetrievalTask


def decode_evidence_request(prompt):
    if "<retrieved_sources>" not in prompt:
        return json.loads(prompt)
    request = json.loads(prompt.split("\n", 1)[0])
    sources = [{**json.loads(metadata), "text": body} for metadata, body in re.findall(
        r"<source>\n([^\n]+)\n<text>\n(.*?)\n</text>\n</source>", prompt, flags=re.DOTALL)]
    request["evidence"] = json.dumps(sources, ensure_ascii=False)
    request["query"] = request["question"]
    return request


class ScriptedLLM:
    token_count_estimated = True

    def __init__(self, tasks=None, *, clarify=False, sufficient=True, long_answer=False):
        self.tasks = tasks
        self.clarify = clarify
        self.sufficient = sufficient
        self.long_answer = long_answer
        self.reviews = []
        self.summary_inputs = []
        self.structured_calls = []
        self.text_prompts = []
        self.stream_prompts = []

    def count_tokens(self, text):
        return len(text)

    async def structured(self, system, prompt, schema):
        self.structured_calls.append((schema.__name__, prompt))
        if schema.__name__ == "QueryDecision":
            data = json.loads(prompt)
            return schema(missing_fields=["版本"] if self.clarify else [],
                          rewritten_query=data["question"], needs_clarification=self.clarify,
                          needs_decomposition=bool(self.tasks),
                          clarification_question="请指定版本", options=["v1", "v2"],
                          scope_constraints={"product": "AtlasDesk", "version": "v2"}
                          if "AtlasDesk v2" in data["question"] else {})
        if schema.__name__ == "QueryPlan":
            data = json.loads(prompt)
            tasks = self.tasks or [RetrievalTask(task_id="t1", query=data["question"])]
            return schema(tasks=tasks)
        if schema.__name__ == "EvidenceReview":
            data = decode_evidence_request(prompt)
            self.reviews.append(data)
            sources = json.loads(data["evidence"])
            return schema(supporting_quotes=[item["text"][:80] for item in sources[:3]],
                          sufficient=self.sufficient, missing=[] if self.sufficient else [data["question"]],
                          next_query=data["query"] + " 补充版本")
        if schema.__name__ == "ConversationSummary":
            data = json.loads(prompt[prompt.index("{"):])
            self.summary_inputs.append(data)
            previous = data["previous_summary"]
            return schema(goals=previous.get("goals", []) or ["检索技术文档"],
                constraints=previous.get("constraints", []) or ["version=2"],
                entities=previous.get("entities", []), facts=["已读取文档"],
                open_questions=previous.get("open_questions", []))
        if schema.__name__ == "EvidenceDigest":
            data = json.loads(prompt[prompt.index("["):])
            return schema(extracts=[{"evidence_id": item["evidence_id"],
                                     "quotes": [item["text"][:24]]} for item in data])
        raise AssertionError(schema.__name__)

    async def text(self, system, prompt):
        self.text_prompts.append(prompt)
        data = decode_evidence_request(prompt)
        evidence = json.loads(data["evidence"])
        return " ".join(f"{item['text']} [{item['evidence_id']}]" for item in evidence)

    async def astream(self, system, prompt):
        self.stream_prompts.append(prompt)
        data = decode_evidence_request(prompt)
        evidence = json.loads(data["evidence"])
        answer = "综合证据 " + " ".join(f"[{item['evidence_id']}]" for item in evidence)
        if self.long_answer:
            answer += " 文档内容" * 200
        yield answer[:5]
        await asyncio.sleep(0)
        yield answer[5:]


class ScriptedRetriever:
    def __init__(self, *, barrier=None, empty=False, blocked=None):
        self.barrier = barrier
        self.empty = empty
        self.blocked = blocked
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()

    def search(self, query, *, filters=None, mode="full", top_k=None):
        with self.guard:
            self.calls.append({"query": query, "filters": filters, "mode": mode})
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.barrier:
                self.barrier.wait(timeout=5)
            if self.blocked:
                self.blocked[0].set()
                assert self.blocked[1].wait(timeout=5)
            if self.empty:
                return []
            name = "alpha" if query.startswith("alpha") else "beta"
            return [Evidence(evidence_id="e_" + name, parent_id="p_" + name,
                doc_id="d_" + name, document_name=name + ".md", text=name + " 版本2配置说明",
                raw_text=name + " 版本2配置说明", source_path="docs/" + name + ".md")]
        finally:
            with self.guard:
                self.active -= 1


async def collect(engine, question, thread="session", clarification=None, mode="agent"):
    return [event async for event in engine.astream(question, thread,
        clarification=clarification, mode=mode)]


def completed(events):
    errors = [event for event in events if event["type"] == "error"]
    assert not errors, errors
    done = [event for event in events if event["type"] == "done"]
    assert len(done) == 1, events
    return done[0]


def test_task_reducer_is_idempotent_and_resettable():
    assert merge_by_task_id({"a": {"answer": "old"}}, {"a": {"answer": "new"}}) == {
        "a": {"answer": "new"}}
    assert merge_by_task_id({"a": {}}, {"__reset__": {}}) == {}


@pytest.mark.asyncio
async def test_real_send_parallelism_and_private_child_context(tmp_path):
    tasks = [RetrievalTask(task_id="a", query="alpha"), RetrievalTask(task_id="b", query="beta")]
    llm = ScriptedLLM(tasks)
    retriever = ScriptedRetriever(barrier=threading.Barrier(2))
    settings = Settings(data_dir=tmp_path, max_parallel_tasks=2)
    result = completed(await collect(GraphEngine(settings, llm, retriever), "比较两个组件"))
    assert retriever.max_active == 2
    assert result["metrics"]["waves"] == 1
    assert result["metrics"]["tasks"] == 2
    assert result["metrics"]["failed_tasks"] == 0
    for review in llm.reviews:
        evidence = json.loads(review["evidence"])
        assert len(evidence) == 1
        assert review["query"] in evidence[0]["text"]
        assert "recent_messages" not in review
    assert settings.checkpoint_path.exists()


@pytest.mark.asyncio
async def test_dependent_tasks_run_in_second_wave_with_explicit_result(tmp_path):
    tasks = [RetrievalTask(task_id="a", query="alpha"),
             RetrievalTask(task_id="b", query="beta", depends_on=["a"])]
    retriever = ScriptedRetriever()
    llm = ScriptedLLM(tasks)
    engine = GraphEngine(Settings(data_dir=tmp_path), llm, retriever)
    result = completed(await collect(engine, "先确定实体再检索"))
    assert result["metrics"]["waves"] == 2
    assert retriever.calls[0]["query"].startswith("alpha")
    assert "前置任务证据摘录" in retriever.calls[1]["query"]
    assert "alpha 版本2配置说明" in retriever.calls[1]["query"]
    packet = json.loads(result["metrics"]["task_results"][0]["answer"])
    assert packet["kind"] == "evidence_packet"
    assert packet["quotes"][0]["evidence_id"] == "e_alpha"
    assert not llm.text_prompts


def test_static_interrupt_checkpoint_resumes_across_engine_and_event_loops(tmp_path):
    settings = Settings(data_dir=tmp_path)
    retriever = ScriptedRetriever()
    initial = asyncio.run(collect(GraphEngine(settings, ScriptedLLM(clarify=True), retriever), "怎么配置"))
    assert initial[-1] == {"type": "clarification", "question": "请指定版本", "options": ["v1", "v2"]}
    assert not retriever.calls
    # A fresh engine and a fresh event loop read the original durable checkpoint.
    engine = GraphEngine(settings, ScriptedLLM(), retriever)
    result = completed(asyncio.run(collect(engine, None, clarification="版本2")))
    assert result["thread_id"] == "session"
    assert "用户补充：版本2" in retriever.calls[0]["query"]
    assert len(retriever.calls) == 1


def test_session_status_reads_durable_interrupt_without_running_nodes(tmp_path):
    settings = Settings(data_dir=tmp_path)
    llm, retriever = ScriptedLLM(clarify=True), ScriptedRetriever()
    engine = GraphEngine(settings, llm, retriever)
    assert asyncio.run(engine.session_status("session")) == {
        "pending_clarification": False, "question": "", "options": []}
    assert not settings.checkpoint_path.exists()
    asyncio.run(collect(engine, "怎么配置"))
    calls_before = len(llm.structured_calls)
    restored = GraphEngine(settings, llm, retriever)
    assert asyncio.run(restored.session_status("session")) == {
        "pending_clarification": True, "question": "请指定版本", "options": ["v1", "v2"]}
    assert len(llm.structured_calls) == calls_before
    assert not retriever.calls
    assert not asyncio.run(restored.session_status("unknown"))["pending_clarification"]
    completed(asyncio.run(collect(restored, None, clarification="版本2")))
    assert not asyncio.run(restored.session_status("session"))["pending_clarification"]


@pytest.mark.asyncio
async def test_empty_retrieval_does_not_repeat_identical_search_and_refuses(tmp_path):
    retriever = ScriptedRetriever(empty=True)
    engine = GraphEngine(Settings(data_dir=tmp_path, max_retrieval_attempts=5), ScriptedLLM(), retriever)
    result = completed(await collect(engine, "没有答案的问题"))
    assert len(retriever.calls) == 1
    assert result["metrics"]["retrieval_attempts"] == 1
    assert result["metrics"]["failed_tasks"] == 1
    assert "没有找到" in result["answer"]


@pytest.mark.asyncio
async def test_transient_backend_error_retries_with_business_limit_not_five_attempts(tmp_path):
    class BrokenRetriever(ScriptedRetriever):
        def search(self, query, **kwargs):
            super().search(query, **kwargs)
            raise ConnectionError("scripted backend failure")

    retriever = BrokenRetriever()
    engine = GraphEngine(Settings(data_dir=tmp_path, max_retrieval_attempts=5), ScriptedLLM(), retriever)
    result = completed(await collect(engine, "alpha"))
    assert len(retriever.calls) == 2
    assert result["metrics"]["retrieval_attempts"] == 2
    assert result["metrics"]["failed_tasks"] == 1
    assert not result["metrics"]["execution_succeeded"]


@pytest.mark.asyncio
async def test_insufficient_evidence_rewrites_and_returns_partial_at_limit(tmp_path):
    retriever = ScriptedRetriever()
    settings = Settings(data_dir=tmp_path).model_copy(update={"evidence_review_policy": "always"})
    engine = GraphEngine(settings, ScriptedLLM(sufficient=False), retriever)
    question = "alpha 版本条件"
    result = completed(await collect(engine, question))
    assert len(retriever.calls) == 2
    assert "补充版本" not in retriever.calls[1]["query"]
    assert retriever.calls[1]["query"] == question + " " + question
    assert result["metrics"]["partial_tasks"] == 1
    assert result["metrics"]["task_results"][0]["missing"] == [question]


@pytest.mark.asyncio
async def test_three_wave_or_cyclic_plan_falls_back_to_one_task(tmp_path):
    tasks = [RetrievalTask(task_id="a", query="alpha"),
             RetrievalTask(task_id="b", query="beta", depends_on=["a"]),
             RetrievalTask(task_id="c", query="gamma", depends_on=["b"])]
    retriever = ScriptedRetriever()
    engine = GraphEngine(Settings(data_dir=tmp_path), ScriptedLLM(tasks), retriever)
    events = await collect(engine, "原始问题")
    result = completed(events)
    assert result["metrics"]["tasks"] == 1
    assert retriever.calls[0]["query"] == "原始问题"
    assert any("回退" in event.get("text", "") for event in events)


@pytest.mark.asyncio
async def test_new_turn_resets_results_but_keeps_conversation(tmp_path):
    settings = Settings(data_dir=tmp_path)
    engine = GraphEngine(settings, ScriptedLLM(), ScriptedRetriever())
    completed(await collect(engine, "alpha"))
    second = completed(await collect(engine, "beta"))
    assert [item["evidence_id"] for item in second["metrics"]["task_results"][0]["evidence"]] == ["e_beta"]
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_path)) as saver:
        snapshot = await engine._build_graph(saver).aget_state({"configurable": {"thread_id": "session"}})
    assert [message["content"] for message in snapshot.values["messages"] if message["role"] == "user"] == [
        "alpha", "beta"]


@pytest.mark.asyncio
async def test_same_thread_rejects_concurrent_writes(tmp_path):
    entered, release = threading.Event(), threading.Event()
    retriever = ScriptedRetriever(blocked=(entered, release))
    settings = Settings(data_dir=tmp_path)
    engine = GraphEngine(settings, ScriptedLLM(), retriever)
    active = asyncio.create_task(collect(engine, "alpha"))
    assert await asyncio.to_thread(entered.wait, 4)
    try:
        duplicate = await collect(GraphEngine(settings, ScriptedLLM(), retriever), "beta")
        assert duplicate[0]["type"] == "error"
        assert "正在执行" in duplicate[0]["text"]
    finally:
        release.set()
    completed(await active)


@pytest.mark.asyncio
async def test_long_history_produces_cumulative_summary_and_token_records(tmp_path):
    settings = Settings(data_dir=tmp_path, context_window_tokens=1024,
        output_reserve_tokens=128, recent_messages=2, evidence_token_budget=200)
    llm = ScriptedLLM(long_answer=True)
    engine = GraphEngine(settings, llm, ScriptedRetriever())
    completed(await collect(engine, "alpha version=2"))
    second = completed(await collect(engine, "继续解释"))
    third = completed(await collect(engine, "保留版本限制"))
    records = second["metrics"]["compression_records"]
    history = next(record for record in records if record["stage"] == "global_history")
    assert history["triggered"]
    assert history["after_tokens"] < history["before_tokens"]
    assert second["metrics"]["token_count_estimated"]
    assert llm.summary_inputs
    assert any(item["previous_summary"].get("constraints") == ["version=2"] for item in llm.summary_inputs)
    assert third["metrics"]["quality_metrics_computed"] is False


@pytest.mark.asyncio
async def test_evidence_compression_preserves_original_and_rejects_invented_quotes(tmp_path):
    class BadExtractLLM(ScriptedLLM):
        async def structured(self, system, prompt, schema):
            if schema.__name__ == "EvidenceDigest":
                return schema(extracts=[{"evidence_id": "e1", "quotes": ["invented fact"]}])
            return await super().structured(system, prompt, schema)

    memory = BudgetMemory(Settings(data_dir=tmp_path, evidence_token_budget=100,
        evidence_compression_strategy="llm"), BadExtractLLM())
    original = {"evidence_id": "e1", "document_name": "source.md", "text": "real source " * 100}
    view, record = await memory.evidence([original], "query", stage="local_answer")
    assert "invented fact" not in view
    assert original["text"] == "real source " * 100
    assert record["fallback"] and record["triggered"]
    assert record["after_tokens"] <= record["budget_tokens"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["dense", "hybrid", "rerank", "parent"])
async def test_ablation_mode_uses_single_pass_without_planning(tmp_path, mode):
    llm, retriever = ScriptedLLM(clarify=True, sufficient=False), ScriptedRetriever()
    result = completed(await collect(GraphEngine(Settings(data_dir=tmp_path), llm, retriever),
                                     "alpha", mode=mode))
    assert len(retriever.calls) == 1 and retriever.calls[0]["mode"] == ("full" if mode == "parent" else mode)
    assert not any(name in {"QueryPlan", "QueryDecision", "EvidenceReview"}
                   for name, _ in llm.structured_calls)
    assert result["metrics"]["mode"] == mode


@pytest.mark.asyncio
async def test_summary_failure_is_bounded_and_explicit(tmp_path):
    class FailingSummaryLLM(ScriptedLLM):
        async def structured(self, system, prompt, schema):
            if schema is ConversationSummary:
                raise RuntimeError("offline summarizer unavailable")
            return await super().structured(system, prompt, schema)

    memory = BudgetMemory(Settings(data_dir=tmp_path, context_window_tokens=512,
        output_reserve_tokens=128, recent_messages=1), FailingSummaryLLM())
    messages = [{"role": "user", "content": "version=2 " * 100}, {"role": "user", "content": "继续"}]
    _, summary, record = await memory.history(messages, {"constraints": ["version=2"]})
    assert record["fallback"] and record["lossy"]
    assert record["after_tokens"] <= record["budget_tokens"]
    assert "version=2" in summary["constraints"]


@pytest.mark.asyncio
async def test_generation_contexts_match_compressed_prompt_and_reject_invisible_source(tmp_path):
    first_id = "11111111-1111-1111-1111-111111111111"
    second_id = "22222222-2222-2222-2222-222222222222"

    class TwoSourceRetriever:
        def search(self, query, **kwargs):
            return [Evidence(evidence_id=eid, parent_id=eid, doc_id=eid,
                document_name="doc.md", text=("alpha " if eid == first_id else "beta ") * 500)
                for eid in (first_id, second_id)]

    class SelectiveLLM(ScriptedLLM):
        async def structured(self, system, prompt, schema):
            if schema.__name__ == "EvidenceDigest":
                items = json.loads(prompt[prompt.index("["):])
                return schema(extracts=[{"evidence_id": item["evidence_id"], "quotes": [item["text"][:24]]}
                    for item in items if item["evidence_id"] == first_id])
            return await super().structured(system, prompt, schema)

        async def astream(self, system, prompt):
            yield f"这个ID虽在原始来源中，但没有进入最终上下文 [{second_id}]"

    settings = Settings(data_dir=tmp_path, context_window_tokens=1024,
        output_reserve_tokens=128, evidence_token_budget=200)
    events = await collect(GraphEngine(settings, SelectiveLLM(), TwoSourceRetriever()), "alpha")
    result = completed(events)
    original = next(event["sources"] for event in events if event["type"] == "sources")
    metrics = result["metrics"]
    assert len(original) == 2
    assert metrics["generation_context_evidence_ids"] == [first_id]
    assert metrics["generation_contexts"] != [item["text"] for item in original]
    assert metrics["generation_context_kind"] == "model_prompt"
    assert metrics["unknown_citation_ids"] == [second_id]
    assert not metrics["citation_ids_valid"]


@pytest.mark.asyncio
async def test_model_answer_without_citations_is_not_marked_valid(tmp_path):
    class UncitedLLM(ScriptedLLM):
        async def astream(self, system, prompt):
            yield "没有引用的回答"

    result = completed(await collect(GraphEngine(Settings(data_dir=tmp_path),
        UncitedLLM(), ScriptedRetriever()), "alpha"))
    assert result["metrics"]["cited_evidence_ids"] == []
    assert not result["metrics"]["citation_ids_valid"]


@pytest.mark.asyncio
@pytest.mark.parametrize("sufficient", [True, False])
async def test_document_brackets_remain_exact_in_model_and_partial_answers(tmp_path, sufficient):
    class BracketRetriever(ScriptedRetriever):
        def search(self, query, **kwargs):
            values = super().search(query, **kwargs)
            values[0].text = "params[timeout]=30"
            return values

    class QuoteLLM(ScriptedLLM):
        async def astream(self, system, prompt):
            yield "参数原文 params[timeout]=30 [e_alpha]"

    result = completed(await collect(GraphEngine(Settings(data_dir=tmp_path),
        QuoteLLM(sufficient=sufficient), BracketRetriever()), "alpha"))
    assert "params[timeout]=30" in result["answer"]
    assert "来源未验证" not in result["answer"]
    assert result["metrics"]["citation_ids_valid"]
    if not sufficient:
        assert result["metrics"]["generation_context_kind"] == "model_prompt"
        assert result["metrics"]["generation_contexts"] == ["params[timeout]=30"]


@pytest.mark.asyncio
async def test_oversized_clarification_preserves_waiting_checkpoint_and_can_retry(tmp_path):
    settings = Settings(data_dir=tmp_path, context_window_tokens=1024, output_reserve_tokens=128)
    retriever = ScriptedRetriever()
    engine = GraphEngine(settings, ScriptedLLM(clarify=True), retriever)
    initial = await collect(engine, "怎么配置")
    assert initial[-1]["type"] == "clarification"
    rejected = await collect(engine, None, clarification="x" * 705)
    assert rejected[-1]["type"] == "error"
    assert "超过模型输入预算" in rejected[-1]["text"]
    assert (await engine.session_status("session"))["pending_clarification"]
    assert not retriever.calls
    completed(await collect(engine, None, clarification="版本2"))
    assert not (await engine.session_status("session"))["pending_clarification"]
    assert len(retriever.calls) == 1


@pytest.mark.asyncio
async def test_deterministic_summary_trimming_is_explicitly_lossy(tmp_path):
    class OversizedSummaryLLM(ScriptedLLM):
        async def structured(self, system, prompt, schema):
            if schema is ConversationSummary:
                return schema(constraints=["version=2"], facts=["long fact " * 100] * 3)
            return await super().structured(system, prompt, schema)

    memory = BudgetMemory(Settings(data_dir=tmp_path, context_window_tokens=1024,
        output_reserve_tokens=128, recent_messages=1), OversizedSummaryLLM())
    messages = [{"role": "user", "content": "version=2 " * 150}, {"role": "user", "content": "继续"}]
    _, summary, record = await memory.history(messages, {})
    assert record["triggered"] and record["lossy"] and record["hard_truncated"]
    assert record["after_tokens"] <= record["budget_tokens"]
    assert summary["constraints"] == ["version=2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("compress_history", [False, True])
async def test_prior_entity_and_version_drive_followup_planner_and_actual_search(tmp_path, compress_history):
    class ContextualLLM(ScriptedLLM):
        async def structured(self, system, prompt, schema):
            if schema.__name__ == "QueryDecision":
                data = json.loads(prompt)
                if data["question"] == "它的回滚步骤呢？":
                    memory_text = json.dumps(data["summary"] if compress_history else data["recent_messages"],
                                             ensure_ascii=False)
                    assert "AtlasDesk" in memory_text and "v2" in memory_text
                    if compress_history:
                        # Only the current pronoun question remains verbatim;
                        # the standalone query therefore depends on the summary.
                        assert "AtlasDesk" not in json.dumps(data["recent_messages"], ensure_ascii=False)
                    return schema(missing_fields=[], needs_clarification=False, needs_decomposition=False,
                                  rewritten_query="AtlasDesk v2 的回滚步骤")
            if schema is ConversationSummary:
                return schema(goals=["查询部署和回滚"], entities=["AtlasDesk"],
                              constraints=["版本=v2"], facts=["用户明确使用AtlasDesk v2"])
            return await super().structured(system, prompt, schema)

    settings = Settings(data_dir=tmp_path, context_window_tokens=1024 if compress_history else 8192,
        output_reserve_tokens=128, recent_messages=1 if compress_history else 4)
    retriever = ScriptedRetriever()
    engine = GraphEngine(settings, ContextualLLM(long_answer=compress_history), retriever)
    completed(await collect(engine, "AtlasDesk v2 的部署步骤"))
    result = completed(await collect(engine, "它的回滚步骤呢？"))
    assert retriever.calls[-1]["query"] == "AtlasDesk v2 的回滚步骤"
    assert result["metrics"]["task_results"][0]["query"] == "AtlasDesk v2 的回滚步骤"
    history_record = next(record for record in result["metrics"]["compression_records"]
                          if record["stage"] == "global_history")
    assert history_record["triggered"] is compress_history


@pytest.mark.asyncio
async def test_review_and_aggregate_receive_literal_sources_without_leaf_generation(tmp_path):
    literal = 'AtlasDesk v2 默认端口为 8088。\n"/healthz" 是存活探针；"/readyz" 检查数据库和 Redis。'

    class LiteralRetriever(ScriptedRetriever):
        def search(self, query, **kwargs):
            evidence = super().search(query, **kwargs)
            evidence[0].text = literal
            evidence[0].heading_path = ["AtlasDesk v2 部署指南", "探针与发布"]
            return evidence

    llm = ScriptedLLM()
    settings = Settings(data_dir=tmp_path).model_copy(update={"evidence_review_policy": "always"})
    result = completed(await collect(GraphEngine(settings, llm, LiteralRetriever()),
                                     "AtlasDesk v2 的端口和两种探针用途是什么"))
    review_prompts = [prompt for name, prompt in llm.structured_calls if name == "EvidenceReview"]
    assert review_prompts and llm.stream_prompts and not llm.text_prompts
    for prompt in [*review_prompts, *llm.stream_prompts]:
        assert literal in prompt
        assert "<retrieved_sources>" in prompt and "<text>" in prompt
        assert '"evidence":"[' not in prompt
        assert '\\"/healthz\\"' not in prompt
        assert "AtlasDesk v2 部署指南 / 探针与发布" in prompt
    assert result["metrics"]["generation_contexts"] == [literal]


@pytest.mark.asyncio
async def test_planner_omission_cannot_drop_parent_product_and_version_from_children(tmp_path):
    question = "AtlasDesk v2 的默认端口及存活和就绪探针定义是什么"
    tasks = [RetrievalTask(task_id="port", query="默认监听端口"),
             RetrievalTask(task_id="probes", query="存活探针和就绪探针的定义")]
    llm, retriever = ScriptedLLM(tasks), ScriptedRetriever()
    result = completed(await collect(GraphEngine(Settings(data_dir=tmp_path), llm, retriever), question))
    assert result["metrics"]["tasks"] == 2
    assert all("AtlasDesk v2" in call["query"] for call in retriever.calls)
    assert all(review["constraints"] == {"product": "AtlasDesk", "version": "v2"}
               for review in llm.reviews)
    # The reviewer still assesses each child, rather than treating all parent
    # facts as mandatory requirements of every retrieval task.
    assert {review["question"] for review in llm.reviews} == {task.query for task in tasks}


@pytest.mark.asyncio
async def test_parent_extra_objectives_never_reach_child_review_answer_or_search(tmp_path):
    parent = "AtlasDesk v2 默认监听什么端口，存活和就绪探针如何定义"

    class ObjectiveAwareLLM(ScriptedLLM):
        async def structured(self, system, prompt, schema):
            if schema.__name__ == "EvidenceReview":
                self.structured_calls.append((schema.__name__, prompt))
                # A script deliberately reproduces the observed failure if any
                # extra parent objective leaks into this child's request.
                leaked = "存活" in prompt or "就绪" in prompt
                return schema(supporting_quotes=["AtlasDesk v2 默认监听端口为8088。"],
                              sufficient="8088" in prompt and not leaked,
                              missing=["存活和就绪探针的定义"] if leaked else [], next_query="")
            return await super().structured(system, prompt, schema)

    llm, retriever = ObjectiveAwareLLM(), ScriptedRetriever()
    engine = GraphEngine(Settings(data_dir=tmp_path), llm, retriever)
    task = RetrievalTask(task_id="port", query="默认监听的端口", constraints={
        "product": "AtlasDesk", "version": "v2", "parent_question": parent,
        "root_question": parent, "objective": "同时说明存活和就绪探针"})
    evidence = Evidence(evidence_id="e_port", parent_id="p_port", doc_id="d_port",
                        document_name="deploy.md", text="AtlasDesk v2 默认监听端口为8088。")
    state = {"task": task.model_dump(), "query": "AtlasDesk v2 默认监听的端口", "mode": "agent",
             "attempts": 1, "evidence": [evidence.model_dump()], "compression_records": []}
    review = await engine._review(state)
    assert review["sufficient"] and review["missing"] == []
    answered = await engine._subanswer({**state, **review})
    assert answered["result"]["status"] == "succeeded"
    for prompt in [*[prompt for name, prompt in llm.structured_calls if name == "EvidenceReview"],
                   *llm.text_prompts]:
        assert "存活" not in prompt and "就绪" not in prompt
        request = decode_evidence_request(prompt)
        assert request["constraints"] == {"product": "AtlasDesk", "version": "v2"}
    await engine._retrieve({**state, "query": "具体监听端口"})
    query = retriever.calls[-1]["query"]
    assert "AtlasDesk" in query and "v2" in query
    assert "存活" not in query and "就绪" not in query


@pytest.mark.asyncio
@pytest.mark.parametrize("sufficient", [True, False])
async def test_quote_first_review_validates_quotes_without_overriding_decision(tmp_path, sufficient):
    assert list(EvidenceReview.model_fields) == ["supporting_quotes", "missing", "sufficient", "next_query"]
    text = "API 默认监听 8088 端口。"

    class InconsistentReviewLLM(ScriptedLLM):
        async def structured(self, system, prompt, schema):
            if schema is EvidenceReview:
                return schema(supporting_quotes=[text, "这是模型虚构的句子"],
                    missing=["尚未确认的另一必要条件"], sufficient=sufficient, next_query="补充查询")
            return await super().structured(system, prompt, schema)

    engine = GraphEngine(Settings(data_dir=tmp_path), InconsistentReviewLLM(), ScriptedRetriever())
    evidence = Evidence(evidence_id="e_port", parent_id="p_port", doc_id="d_port",
                        document_name="deploy.md", text=text)
    review = await engine._review({"task": RetrievalTask(task_id="port", query="默认端口是多少").model_dump(),
        "query": "默认端口是多少", "mode": "agent", "attempts": 1, "evidence": [evidence.model_dump()]})
    assert review["supporting_quotes"] == [text]
    assert review["sufficient"] is sufficient
    assert review["compression_records"][-1]["invalid_quote_count"] == 1
    assert review["missing"] == []
    assert review["next_query"] == ""
    assert review["compression_records"][-1]["discarded_expanded_requirements"] == ["尚未确认的另一必要条件"]


@pytest.mark.asyncio
async def test_unreviewed_partial_tasks_aggregate_without_publishing_audit_opinions(tmp_path):
    class PartialAnswerLLM(ScriptedLLM):
        async def astream(self, system, prompt):
            self.stream_prompts.append(prompt)
            yield "原文可确认 alpha 版本2配置说明 [e_alpha]"

    llm = PartialAnswerLLM(sufficient=False)
    result = completed(await collect(GraphEngine(Settings(data_dir=tmp_path), llm, ScriptedRetriever()), "alpha"))
    assert llm.stream_prompts
    assert "原文可确认 alpha" in result["answer"]
    assert "完整性审核未确认" not in result["answer"]
    assert "无法给出完整结论" not in result["answer"]
    assert result["metrics"]["partial_tasks"] == 1
    assert not result["metrics"]["completeness_confirmed"]
    assert result["metrics"]["task_execution"]["t1"]["review_status"] == "skipped"
    assert result["metrics"]["generation_context_kind"] == "model_prompt"


@pytest.mark.asyncio
async def test_all_partial_generation_failure_keeps_actual_original_excerpt_context(tmp_path):
    class BrokenStreamLLM(ScriptedLLM):
        async def astream(self, system, prompt):
            yield "尚未完成的输出"
            raise RuntimeError("offline stream failed")

    result = completed(await collect(GraphEngine(Settings(data_dir=tmp_path),
        BrokenStreamLLM(sufficient=False), ScriptedRetriever()), "alpha"))
    assert "回答生成未完成" in result["answer"]
    assert "alpha 版本2配置说明" in result["answer"]
    assert "尚未完成的输出" not in result["answer"]
    assert result["metrics"]["generation_context_kind"] == "deterministic_excerpt"
    assert result["metrics"]["generation_contexts"] == ["alpha 版本2配置说明"]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_unknown", [False, True])
async def test_real_qwen_prefixed_multi_uuid_citations_are_normalized_and_audited(tmp_path, include_unknown):
    port_id = "00b2b4d7-dc55-5b7a-974a-de998ee2ae13"
    probe_id = "9fcdfdc2-539f-5f44-9d39-a9c2cc4bd4c4"
    unknown_id = "99999999-9999-5999-8999-999999999999"
    second_citation = unknown_id if include_unknown else probe_id
    raw_answer = ("默认端口8088，存活路径/healthz，就绪路径/readyz。params[timeout]=30。"
                  f"[evidence_id: {port_id}, {second_citation}]")

    class RealSourceRetriever:
        def search(self, query, **kwargs):
            return [Evidence(evidence_id=eid, parent_id=eid, doc_id=eid,
                             document_name="deploy.md", text=text) for eid, text in (
                (port_id, "AtlasDesk v2 默认监听8088端口。"),
                (probe_id, "/healthz是存活探针，/readyz是就绪探针。"))]

    class RealFormatLLM(ScriptedLLM):
        async def astream(self, system, prompt):
            yield raw_answer

    result = completed(await collect(GraphEngine(Settings(data_dir=tmp_path),
        RealFormatLLM(), RealSourceRetriever()), "AtlasDesk v2 端口和探针是什么"))
    metrics = result["metrics"]
    assert metrics["raw_answer"] == raw_answer
    assert "params[timeout]=30" in result["answer"]
    assert f"[{port_id}]" in result["answer"]
    assert "[evidence_id:" not in result["answer"]
    if include_unknown:
        assert "[来源未验证]" in result["answer"]
        assert metrics["unknown_citation_ids"] == [unknown_id]
        assert not metrics["citation_ids_valid"]
        assert metrics["cited_evidence_ids"] == [port_id]
    else:
        assert f"[{port_id}][{probe_id}]" in result["answer"]
        assert metrics["citation_ids_valid"]
        assert metrics["cited_evidence_ids"] == sorted([port_id, probe_id])


@pytest.mark.asyncio
async def test_missing_scope_fields_pause_without_losing_original_intent(tmp_path):
    class MissingScopeLLM(ScriptedLLM):
        async def structured(self, system, prompt, schema):
            if schema.__name__ == "QueryDecision":
                # Reproduce an inconsistent local-model response: it finds
                # missing scope but says false and rewrites the action away.
                return schema(missing_fields=["产品", "目标版本"], needs_clarification=False,
                              needs_decomposition=False,
                              rewritten_query="这个版本号是多少？")
            return await super().structured(system, prompt, schema)

    retriever = ScriptedRetriever()
    engine = GraphEngine(Settings(data_dir=tmp_path), MissingScopeLLM(), retriever)
    events = await collect(engine, "这个版本怎么升级？", thread="missing-scope")
    clarification = next(event for event in events if event["type"] == "clarification")
    assert "产品" in clarification["question"]
    assert not retriever.calls
    status = await engine.session_status("missing-scope")
    assert status["pending_clarification"]
    async with AsyncSqliteSaver.from_conn_string(str(engine.settings.checkpoint_path)) as saver:
        graph = engine._build_graph(saver)
        snapshot = await graph.aget_state({"configurable": {"thread_id": "missing-scope"}})
    assert snapshot.values["query"] == "这个版本怎么升级？"


def test_empty_model_query_decision_is_not_silently_treated_as_clear():
    from pydantic import ValidationError

    from agentic_kb.graph import QueryDecision

    with pytest.raises(ValidationError):
        QueryDecision.model_validate({})
    with pytest.raises(ValidationError, match="needs_decomposition"):
        QueryDecision.model_validate({"missing_fields": [], "needs_clarification": False,
                                      "rewritten_query": "同一主体的多个事实"})


@pytest.mark.asyncio
async def test_structured_simple_decision_uses_one_search_and_no_leaf_model_calls(tmp_path):
    llm, retriever = ScriptedLLM(), ScriptedRetriever()
    settings = Settings(data_dir=tmp_path).model_copy(update={"evidence_review_policy": "complex"})
    result = completed(await collect(GraphEngine(settings, llm, retriever), "alpha的端口和两个状态端点"))
    assert [name for name, _ in llm.structured_calls] == ["QueryDecision"]
    assert len(retriever.calls) == 1
    assert not llm.text_prompts and len(llm.stream_prompts) == 1
    metrics = result["metrics"]
    assert not metrics["planner_called"] and not metrics["decomposition_used"]
    assert metrics["subanswer_generation_calls"] == 0
    assert metrics["task_execution"]["t1"]["review_status"] == "skipped"
    assert metrics["task_execution"]["t1"]["review_performed"] is False
    assert metrics["execution_succeeded"] and not metrics["completeness_confirmed"]
    assert metrics["task_results"][0]["answer"] == ""


@pytest.mark.asyncio
async def test_expanded_review_goals_are_audit_only_and_do_not_drive_retrieval_or_answer(tmp_path):
    expanded = ["端口是否允许配置", "探针的完整工作流程"]

    class GoalExpandingLLM(ScriptedLLM):
        async def structured(self, system, prompt, schema):
            if schema is EvidenceReview:
                self.structured_calls.append((schema.__name__, prompt))
                source = json.loads(decode_evidence_request(prompt)["evidence"])[0]
                return schema(supporting_quotes=[source["text"]], missing=expanded,
                              sufficient=False, next_query="进一步查询端口配置方式和探针完整工作流程")
            return await super().structured(system, prompt, schema)

        async def astream(self, system, prompt):
            self.stream_prompts.append(prompt)
            assert all(item not in prompt for item in expanded)
            request = decode_evidence_request(prompt)
            assert not {"task_results", "review_unconfirmed", "summary"}.intersection(request)
            source = json.loads(request["evidence"])[0]
            yield source["text"] + f" [{source['evidence_id']}]"

    class PortRetriever(ScriptedRetriever):
        def search(self, query, **kwargs):
            values = super().search(query, **kwargs)
            values[0].text = "组件L默认监听8123端口。"
            return values

    llm, retriever = GoalExpandingLLM(), PortRetriever()
    settings = Settings(data_dir=tmp_path).model_copy(update={"evidence_review_policy": "always"})
    result = completed(await collect(GraphEngine(settings, llm, retriever), "组件L默认监听端口是什么"))
    assert len(retriever.calls) == 1
    assert all(item not in retriever.calls[0]["query"] for item in expanded)
    assert "8123" in result["answer"] and not any(item in result["answer"] for item in expanded)
    assert "审核未确认" not in result["answer"] and "待核查项" not in result["answer"]
    assert result["metrics"]["task_results"][0]["status"] == "partial"
    assert result["metrics"]["task_results"][0]["missing"] == []
    record = next(item for item in result["metrics"]["compression_records"] if item["stage"] == "local_review")
    assert record["discarded_expanded_requirements"] == expanded
    assert record["proposed_next_query"].startswith("进一步查询")


@pytest.mark.asyncio
async def test_new_simple_turn_clears_previous_complex_execution_audit(tmp_path):
    llm = ScriptedLLM([RetrievalTask(task_id="a", query="alpha"), RetrievalTask(task_id="b", query="beta")])
    engine = GraphEngine(Settings(data_dir=tmp_path), llm, ScriptedRetriever())
    first = completed(await collect(engine, "比较两个组件"))
    assert set(first["metrics"]["task_execution"]) == {"a", "b"}
    llm.tasks = None
    second = completed(await collect(engine, "alpha"))
    assert set(second["metrics"]["task_execution"]) == {"t1"}
    assert second["metrics"]["task_execution"]["t1"]["review_status"] == "skipped"


@pytest.mark.asyncio
async def test_analysis_receives_configured_knowledge_base_domain_metadata(tmp_path):
    domain = "组件R研发技术文档，包含v1/v2版本"
    settings = Settings(data_dir=tmp_path).model_copy(update={"knowledge_base_scope": domain})
    llm = ScriptedLLM()
    completed(await collect(GraphEngine(settings, llm, ScriptedRetriever()), "v1为什么不能水平扩容"))
    request = json.loads(next(prompt for name, prompt in llm.structured_calls if name == "QueryDecision"))
    assert request["knowledge_base_scope"] == domain
    assert request["question"] == "v1为什么不能水平扩容"


@pytest.mark.asyncio
async def test_first_turn_cannot_be_replaced_by_inconsistent_analysis_followup(tmp_path):
    class InconsistentRewriteLLM(ScriptedLLM):
        async def structured(self, system, prompt, schema):
            if schema.__name__ == "QueryDecision":
                return schema(missing_fields=[], needs_clarification=False,
                    needs_decomposition=False, rewritten_query="请确认您指的是哪个版本和组件范围。")
            return await super().structured(system, prompt, schema)

    question = "组件R包含哪些主要组件？"
    llm, retriever = InconsistentRewriteLLM(), ScriptedRetriever()
    result = completed(await collect(GraphEngine(Settings(data_dir=tmp_path), llm, retriever), question))
    assert retriever.calls[0]["query"] == question
    assert decode_evidence_request(llm.stream_prompts[0])["question"] == question
    assert result["metrics"]["planner_called"] is False


def test_clarification_schema_cannot_ask_user_to_supply_the_business_answer():
    from pydantic import ValidationError

    from agentic_kb.graph import QueryDecision

    with pytest.raises(ValidationError):
        QueryDecision(missing_fields=["两个版本的具体差异"], needs_clarification=True,
                      needs_decomposition=True, rewritten_query="比较两个版本")


@pytest.mark.asyncio
@pytest.mark.parametrize("question,missing,expected_pause", [
    ("为什么v1不能扩容，而v2可以多副本？", "version", False),
    ("组件R v3的默认端口是什么？", "版本", False),
    ("组件R v3如何升级？", "目标版本", True),
    ("这个版本如何升级？", "版本", True),
])
async def test_clarification_checks_scope_against_explicit_question_versions(tmp_path, question, missing, expected_pause):
    class RedundantVersionLLM(ScriptedLLM):
        async def structured(self, system, prompt, schema):
            if schema.__name__ == "QueryDecision":
                return schema(missing_fields=[missing], needs_clarification=True,
                    needs_decomposition=False, rewritten_query=question)
            return await super().structured(system, prompt, schema)

    events = await collect(GraphEngine(Settings(data_dir=tmp_path), RedundantVersionLLM(), ScriptedRetriever()), question)
    assert any(event["type"] == "clarification" for event in events) is expected_pause
    assert any(event["type"] == "done" for event in events) is not expected_pause


@pytest.mark.asyncio
async def test_explicit_version_comparison_reaches_planner_despite_missed_flag(tmp_path):
    class VersionPlannerLLM(ScriptedLLM):
        async def structured(self, system, prompt, schema):
            if schema.__name__ == "QueryPlan":
                self.structured_calls.append((schema.__name__, prompt))
                return schema(tasks=[RetrievalTask(task_id="old", query="组件R v4的会话状态"),
                                     RetrievalTask(task_id="new", query="组件R v5的会话状态")])
            return await super().structured(system, prompt, schema)

    llm, retriever = VersionPlannerLLM(), ScriptedRetriever()
    result = completed(await collect(GraphEngine(Settings(data_dir=tmp_path), llm, retriever),
                                    "比较组件R v4和v5的会话状态"))
    assert result["metrics"]["planner_called"] is True
    assert result["metrics"]["decomposition_used"] is True
    assert result["metrics"]["analysis_audit"]["reported_needs_decomposition"] is False
    assert {call["query"] for call in retriever.calls} == {"组件R v4的会话状态", "组件R v5的会话状态"}


def test_child_query_keeps_own_literal_terms_without_sibling_objective():
    from agentic_kb.graph import preserve_task_query_terms

    question = "为什么组件R v4不能直接扩容，而v5可以多个副本？"
    old = preserve_task_query_terms(question, "组件R v4的架构限制")
    new = preserve_task_query_terms(question, "组件R v5的分布式架构")
    assert "v4不能直接扩容" in old and "v5" not in old
    assert "v5可以多个副本" in new and "v4" not in new
    shared = "比较组件R v4和v5的部署步骤"
    assert preserve_task_query_terms(shared, "组件R v4部署") == "组件R v4部署"
    assert preserve_task_query_terms("比较组件R v4，和组件S v5的配置", "组件R v4配置") == "组件R v4配置"
    assert preserve_task_query_terms("组件R v4需要哪些配置", "组件R v4数据库配置") == "组件R v4数据库配置"
