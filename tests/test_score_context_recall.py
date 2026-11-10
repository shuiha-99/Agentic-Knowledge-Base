"""ContextRecall replay protocol only; no real judge or embedding model runs."""
from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from agentic_kb.evaluation import BenchmarkRun, EvaluationCase, canonical_hash

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("context_recall_replay_protocol", ROOT / "scripts/score_context_recall.py")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class ProtocolScorer:
    """Transport fixture values are not model-quality measurements."""
    versions = {"fixture": "protocol-only"}

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
        self.closed = 0

    async def score(self, case, answer, contexts, *, metric_names, embeddings):
        assert metric_names == ["context_recall"] and embeddings == {}
        assert contexts == ["actual generation context"]
        self.calls.append(case.id)
        if self.fail:
            return {"context_recall": None}, {"context_recall": "protocol fixture failure"}
        return {"context_recall": .5}, {}

    async def prepare_embeddings(self, *args, **kwargs):
        raise AssertionError("ContextRecall must never prepare embeddings")

    async def close(self):
        self.closed += 1


def commit_generation(run, case_id, *, status="success", repeat=0, attempt=1, context_scope="final_generation_contexts"):
    item = run.item_id(case_id, repeat)
    row = {"phase": "generation", "item": item, "case_id": case_id, "repeat": repeat,
           "attempt": attempt, "status": status, "answer": "recorded answer", "error": None,
           "context_scope": context_scope, "retrieved_contexts": ["actual generation context"]}
    row["generation_hash"] = canonical_hash(row)
    run.journal.commit(f"generation:{item}:attempt{attempt}", row)
    return row


def make_run(path, ids=("a",), *, repeats=1):
    cases = [EvaluationCase(id=case_id, question="question-" + case_id, reference="reference-" + case_id) for case_id in ids]
    return BenchmarkRun(path, cases, {"identity": "frozen"}, stability_repeats=repeats, stability_ids=list(ids))


@pytest.mark.asyncio
async def test_partial_existing_only_replay_preserves_all_denominators_and_resumes(tmp_path):
    cases = [EvaluationCase(id=case_id, question="question-" + case_id, reference="reference-" + case_id)
             for case_id in ("a", "b", "c", "missing", "excluded")]
    cases[-1].kind = "ambiguous"
    ids = [case.id for case in cases]
    run = BenchmarkRun(tmp_path, cases, {"identity": "frozen"}, stability_repeats=3, stability_ids=ids)
    original_manifest = (tmp_path / "manifest.json").read_bytes()
    commit_generation(run, "a")
    commit_generation(run, "b", status="clarification_required")
    commit_generation(run, "c", status="interrupted")
    commit_generation(run, "excluded")
    scorer = ProtocolScorer()
    report = await driver.score_existing(run, scorer)
    assert scorer.calls == ["a"] and scorer.closed == 1
    assert (tmp_path / "manifest.json").read_bytes() == original_manifest
    assert not list((tmp_path / "artifacts").glob("*.embeddings.json"))
    selection = json.loads((tmp_path / "context_recall_selection.json").read_text(encoding="utf-8"))
    assert selection["metric"] == "context_recall" and selection["expected_count"] == 3
    assert len(selection["generation_snapshot"]) == 4
    assert selection["frozen_hash"] == run.manifest["frozen_hash"]
    assert selection["driver_sha256"] == driver.file_sha(Path(driver.__file__))
    assert {entry["item"] for entry in selection["items"]} == {run.item_id(case_id, 0) for case_id in ("a", "b", "c")}
    metrics = report["stability"]["aggregate"]["metrics"]
    assert metrics["context_recall"]["eligible_count"] == 12
    assert metrics["context_recall"]["valid_count"] == 1
    assert metrics["context_recall"]["failed_or_blocked_count"] == 2
    assert metrics["context_recall"]["not_started_count"] == 9
    assert report["stability"]["aggregate"]["statuses"]["not_started"] == 11
    for metric in ("answer_correctness", "faithfulness"):
        assert metrics[metric]["eligible_count"] == metrics[metric]["not_started_count"] == 12
        assert metrics[metric]["total_attempt_count"] == 0
    resumed = BenchmarkRun(tmp_path, cases, {"identity": "frozen"}, stability_repeats=3, stability_ids=ids)
    skipped = ProtocolScorer()
    await driver.score_existing(resumed, skipped)
    assert skipped.calls == [] and skipped.closed == 1
    records = [row for row in resumed.journal.completed.values() if row.get("phase") == "score"]
    assert len(records) == 3 and {row["metric"] for row in records} == {"context_recall"}


@pytest.mark.asyncio
async def test_failed_metric_requires_explicit_retry_and_keeps_attempt_history(tmp_path):
    run = make_run(tmp_path)
    commit_generation(run, "a")
    first = ProtocolScorer(fail=True)
    await driver.score_existing(run, first)
    skipped = ProtocolScorer()
    await driver.score_existing(run, skipped)
    assert skipped.calls == []
    retry = ProtocolScorer()
    report = await driver.score_existing(run, retry, retry_failed=True)
    assert retry.calls == ["a"]
    stats = report["main"]["metrics"]["context_recall"]
    assert stats["eligible_count"] == stats["valid_count"] == 1
    assert stats["total_attempt_count"] == 2 and stats["historical_failed_attempt_count"] == 1


@pytest.mark.asyncio
async def test_inflight_metric_is_not_silently_repeated_and_invalid_context_does_not_judge(tmp_path):
    run = make_run(tmp_path, ("a", "missing-context"))
    generation = commit_generation(run, "a")
    commit_generation(run, "missing-context", context_scope="not_recorded")
    item = run.item_id("a", 0)
    run.journal.begin(f"score:{item}:{generation['generation_hash']}:context_recall:attempt1")
    scorer = ProtocolScorer()
    report = await driver.score_existing(run, scorer)
    assert scorer.calls == []
    assert report["main"]["metrics"]["context_recall"]["failed_or_blocked_count"] == 2
    rows = [row for row in run.journal.completed.values() if row.get("phase") == "score"]
    assert {row["status"] for row in rows} == {"interrupted", "error"}


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["new_generation", "new_attempt", "excluded_generation", "driver", "scope"])
async def test_selection_rejects_generation_scope_or_driver_drift_before_any_judge(tmp_path, change):
    run = make_run(tmp_path, ("a", "b", "excluded"))
    run.cases["excluded"].kind = "ambiguous"
    commit_generation(run, "a")
    await driver.score_existing(run, ProtocolScorer(), limit=0)
    scope = "all"
    if change == "new_generation":
        commit_generation(run, "b")
    elif change == "new_attempt":
        commit_generation(run, "a", attempt=2)
    elif change == "excluded_generation":
        commit_generation(run, "excluded")
    elif change == "driver":
        path = tmp_path / "context_recall_selection.json"
        selection = json.loads(path.read_text(encoding="utf-8"))
        selection["driver_sha256"] = "changed-driver"
        path.write_text(json.dumps(selection), encoding="utf-8")
    else:
        scope = "main"
    scorer = ProtocolScorer()
    with pytest.raises(ValueError, match="选择已冻结"):
        await driver.score_existing(run, scorer, scope=scope)
    assert scorer.calls == [] and scorer.closed == 1


@pytest.mark.asyncio
async def test_execute_rejects_original_environment_drift_without_creating_scorer(tmp_path, monkeypatch):
    run = make_run(tmp_path / "run")
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text("\n".join(json.dumps(asdict(case)) for case in run.cases.values()), encoding="utf-8")
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(driver.Settings, "from_env", lambda: type("Settings", (), {"provider": "ollama"})())
    monkeypatch.setattr(driver, "environment_snapshot", lambda *a, **kw: {"identity": "changed"})

    def forbidden_scorer(*args, **kwargs):
        raise AssertionError("environment drift must be rejected before constructing a worker")

    monkeypatch.setattr(driver, "make_scorer", forbidden_scorer)
    args = driver.parser().parse_args(["--run-dir", str(run.directory), "--dataset", str(dataset),
                                      "--stability-repeats", "1", "--stability-ids", "a"])
    with pytest.raises(ValueError, match="冻结清单不匹配"):
        await driver.execute(args)
    assert not (run.directory / "context_recall_selection.json").exists()


def test_cli_cannot_request_generation_or_a_different_metric():
    for forbidden in (["generate"], ["--metrics", "answer_correctness"], ["--metric", "faithfulness"]):
        with pytest.raises(SystemExit):
            driver.parser().parse_args(["--run-dir", "existing-run", *forbidden])
