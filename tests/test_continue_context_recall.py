"""Continuation safety checks; no real models or external calls."""
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location("continue_cr", Path(__file__).parents[1] / "scripts/continue_context_recall.py")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class Journal:
    def __init__(self):
        self.started = set()
        self.completed = {}

    def begin(self, key):
        self.started.add(key)

    def commit(self, key, row):
        self.completed[key] = row
        self.started.discard(key)


def make_run():
    return SimpleNamespace(journal=Journal(), cases={"a": object()},
                           item_id=lambda c, r: f"agent:{c}:repeat{r}", _latest=lambda *a, **k: None)


def test_failures_keep_denominator_and_null():
    rows = [{"question_id": c, "sample_id": c, "score": score, "status": status}
            for c, score, status in [("a", 0.0, "success"), ("b", 1.0, "success"),
                                      ("c", None, "generation_failed"), ("d", None, "unscored")]]
    summary = driver.summarize(rows)
    assert (summary["total"], summary["valid_count"], summary["failed_count"], summary["unscored_count"]) == (4, 2, 1, 1)
    assert summary["mean_valid_only"] == summary["median_valid_only"] == .5
    assert summary["stddev_population"] == .5
    assert summary["zero_question_ids"] == ["a"]


def test_generation_failed_never_calls_judge():
    scorer = SimpleNamespace(versions={})
    generation = {"generation_hash": "h", "status": "clarification_required", "error": None}
    row = asyncio.run(driver.score_one(make_run(), scorer, "a", 0, generation))
    assert row["status"] == "generation_failed" and row["score"] is None
    assert row["errors"]["generation"] == "clarification_required"


def test_only_context_recall_is_requested():
    calls = []

    async def score(case, answer, contexts, **kwargs):
        calls.append(kwargs)
        return {"context_recall": .75}, {}

    scorer = SimpleNamespace(versions={"ragas": "0.3.3"}, score=score)
    generation = {"generation_hash": "h", "status": "success", "context_scope": "final_generation_contexts",
                  "retrieved_contexts": ["actual final context"], "answer": "answer"}
    row = asyncio.run(driver.score_one(make_run(), scorer, "a", 0, generation))
    assert calls == [{"metric_names": ["context_recall"], "embeddings": {}}]
    assert row["status"] == "success" and row["score"] == .75


def test_inflight_score_is_logged_without_second_external_call():
    run = make_run()
    run.journal.started.add("score:agent:a:repeat0:h:context_recall:attempt1")
    row = asyncio.run(driver.score_one(run, SimpleNamespace(versions={}), "a", 0,
                                     {"generation_hash": "h", "status": "success"}))
    assert row["status"] == "scoring_failed" and "interrupted" in row["errors"]


def test_question_id_and_repeat_remain_distinct():
    run = make_run()
    run.journal.completed = {str(i): {"phase": "score", "item": f"agent:a:repeat{i}", "metric": "context_recall",
                                          "status": "success", "score": float(i)} for i in range(2)}
    assert driver.latest_valid(run, "agent:a:repeat0")["score"] == 0.0
    assert driver.latest_valid(run, "agent:a:repeat1")["score"] == 1.0


def test_inherited_uncommitted_case_gets_one_explicit_fresh_attempt():
    run = make_run()
    records = [{"key": "generation:agent:a:repeat0:attempt1", "state": "started"},
               {"key": "generation:agent:a:repeat0:attempt1:turn0", "state": "started"}]
    run._latest = lambda *a, **k: {"status": "interrupted", "attempt": 1}
    assert driver.inherited_inflight_repairs(run, records) == {"agent:a:repeat0"}
    run._latest = lambda *a, **k: {"status": "interrupted", "attempt": 2}
    assert driver.inherited_inflight_repairs(run, records) == set()
    run._latest = lambda *a, **k: {"status": "clarification_required", "attempt": 2}
    assert driver.inherited_inflight_repairs(run, records) == set()


def test_failure_detail_uses_current_generation_not_old_attempt_with_equal_number():
    run = make_run()

    def latest(phase, item, **filters):
        if phase == "generation":
            return {"generation_hash": "new", "status": "clarification_required", "retrieved_contexts": []}
        assert filters == {"metric": "context_recall", "generation_hash": "new"}
        return {"status": "generation_failed", "score": None, "errors": {"generation": "new failure"}}

    run._latest = latest
    row = driver.detail(run, [("a", 0)])[0]
    assert row["status"] == "generation_failed"
    assert row["errors"] == {"generation": "new failure"}
