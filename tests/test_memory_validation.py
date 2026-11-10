"""Protocol tests only: no Ollama, tokenizer download or model inference."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from agentic_kb.config import Settings
from agentic_kb.memory import ConversationSummary

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("memory_validation_protocol", ROOT / "scripts/validate_memory.py")
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


class CountingSummary:
    """A deliberately imperfect test provider, never a quality measurement."""
    token_count_estimated = True

    def __init__(self):
        self.calls = 0
        self.inputs = []

    def count_tokens(self, text):
        return max(1, (len(text) + 3) // 4)

    async def structured(self, system, prompt, schema):
        assert schema is ConversationSummary
        self.calls += 1
        self.inputs.append((system, prompt))
        assert "gold_patterns" not in prompt and "answer_checks" not in prompt
        return schema(goals=["继续交接"], constraints=[f"independent_summary_call={self.calls}"],
                      entities=["测试对象"], facts=[], open_questions=[])


class CountingAnswer:
    token_count_estimated = True

    def __init__(self, fail_calls=()):
        self.calls = 0
        self.inputs = []
        self.fail_calls = set(fail_calls)

    def count_tokens(self, text):
        return max(1, (len(text) + 3) // 4)

    async def structured(self, system, prompt, schema):
        self.calls += 1
        self.inputs.append(prompt)
        assert "gold_patterns" not in prompt and "answer_checks" not in prompt
        if self.calls in self.fail_calls:
            raise RuntimeError("scripted final answer failure")
        # Wrong answers are intentional: the harness must report their failures,
        # not interpret successful plumbing as factual or semantic correctness.
        return schema(**{key: f"unknown-{self.calls}" for key in schema.model_fields})


@pytest.fixture
def setup_protocol(tmp_path):
    settings = Settings(data_dir=tmp_path / "runtime", provider="offline-test",
                        context_window_tokens=2048, output_reserve_tokens=256)
    specs = validation.load_specs(ROOT / "data/eval/memory_stress.jsonl")
    summary, answer = CountingSummary(), CountingAnswer()
    workloads = [validation.materialize_local(row, ROOT / "data/demo_docs") if row["kind"] == "local"
                 else validation.materialize_history(row, settings, summary.count_tokens) for row in specs]
    return settings, specs, workloads, summary, answer


def test_frozen_cases_are_calibrated_meaningful_and_gold_is_not_an_input(setup_protocol):
    settings, specs, workloads, summary, _ = setup_protocol
    assert {kind: sum(row["kind"] == kind for row in specs) for kind in ("normal", "overload", "local")} == {
        "normal": 4, "overload": 2, "local": 2}
    assert sum(row.get("semantic_check", False) for row in specs if row["kind"] == "normal") == 2
    for spec, workload in zip(specs, workloads, strict=True):
        encoded = json.dumps(workload, ensure_ascii=False)
        assert "gold_patterns" not in encoded and "answer_checks" not in encoded
        if spec["kind"] == "normal":
            assert workload["calibration"]["normal_above_trigger"]
            assert workload["calibration"]["baseline_answer_fits"]
            context = validation.history_json(workload["messages"])
            assert validation.pattern_score(context, spec["gold_patterns"])["fraction"] == 1
            again = validation.materialize_history(spec, settings, summary.count_tokens)
            assert workload == again
            requests = [row["content"] for row in workload["messages"] if "WK-" in row["content"] and row["role"] == "user"]
            assert len(requests) == len(set(requests))
        elif spec["kind"] == "overload":
            assert workload["calibration"]["full_context_tokens"] > settings.context_window_tokens
            assert not workload["calibration"]["baseline_answer_fits"]
        else:
            assert spec["evidence_budget_tokens"] == workload["evidence_budget_tokens"] == 768


def test_correction_and_association_use_answer_slots_not_just_context_patterns(setup_protocol):
    _, specs, workloads, _, _ = setup_protocol
    spec, workload = specs[0], workloads[0]
    context = validation.history_json(workload["messages"])
    assert "8317" in context and "8429" in context
    assert validation.pattern_score(context, spec["gold_patterns"])["fraction"] == 1
    wrong = {"version": "2.4.1", "port": "8317", "write_state": "只读",
             "write_conditions": "演练事务回滚且业务负责人签字"}
    score = validation.answer_score(wrong, spec["answer_checks"])
    assert not score["all_passed"]
    assert next(row for row in score["slots"] if row["slot"] == "port")["passed"] is False
    swap = {"north_queue": "Q-SOUTH", "north_batch": "61", "south_queue": "Q-NORTH", "south_batch": "37",
            "north_state": "只读，尚未签字", "south_state": "可写"}
    assert not validation.answer_score(swap, specs[1]["answer_checks"])["all_passed"]


def test_provenance_allows_ordered_noncontiguous_quotes_and_rejects_fabrication():
    sources = [{"evidence_id": "a", "text": "第一句。无关内容。第三句。"}]
    assert validation.provenance([{"evidence_id": "a", "text": "第一句。第三句。"}], sources)["valid"]
    assert not validation.provenance([{"evidence_id": "a", "text": "第三句。第一句。"}], sources)["valid"]
    assert not validation.provenance([{"evidence_id": "a", "text": "第一句。虚构句子。"}], sources)["valid"]
    assert not validation.provenance([{"evidence_id": "unknown", "text": "第一句。"}], sources)["valid"]


@pytest.mark.asyncio
async def test_three_repeats_do_not_reuse_model_outputs_and_resume_skips_only_committed_jobs(tmp_path, setup_protocol):
    settings, specs, workloads, summary, answer = setup_protocol
    output = tmp_path / "experiment"
    args = (settings, specs[:1], workloads[:1], output, summary, answer,
            {"source": "frozen-test-code", "models": {"summary": "fake-a", "answer": "fake-b"}})
    report = await validation.run_validation(*args, allow_estimated=True)
    assert report["complete"] and not report["real_model_validation"]
    assert report["expected_jobs"] == 6 and report["completed_jobs"] == 6
    assert report["semantic_summary"]["completed_answer_jobs"] == 6
    assert answer.calls == 6 and summary.calls >= 3
    assert report["semantic_summary"]["before_all_slots_passed"] == 0
    assert report["semantic_summary"]["after_all_slots_passed"] == 0
    assert report["estimated_token_counts"]
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    on = [row for row in rows if row["condition"] == "on"]
    assert {row["repeat"] for row in on} == {1, 2, 3}
    assert len({row["context_sha256"] for row in on}) == 3
    old_counts = (summary.calls, answer.calls)
    await validation.run_validation(*args[:-1], {"models": {"answer": "fake-b", "summary": "fake-a"},
        "source": "frozen-test-code"}, allow_estimated=True, resume=True)
    assert (summary.calls, answer.calls) == old_counts
    with pytest.raises(ValueError, match="冻结"):
        await validation.run_validation(*args[:-1], {"source": "changed-code"}, allow_estimated=True, resume=True)
    assert (summary.calls, answer.calls) == old_counts


@pytest.mark.asyncio
async def test_quick_schedule_separates_normal_overload_and_local(tmp_path, setup_protocol):
    settings, specs, workloads, summary, answer = setup_protocol
    report = await validation.run_validation(settings, specs, workloads, tmp_path / "quick", summary, answer,
                                              {"test": True}, quick=True, allow_estimated=True)
    assert report["complete"] and report["quick"]
    assert report["expected_jobs"] == 16
    assert report["groups"]["normal"]["ratio"]["count"] == 4
    assert report["groups"]["overload"]["ratio"]["count"] == 2
    assert report["groups"]["local"]["ratio"]["count"] == 2
    assert report["groups"]["overload"]["target_band"] is None
    assert report["semantic_summary"]["expected_answer_jobs"] == 4
    assert answer.calls == 4
    rows = [json.loads(line) for line in (tmp_path / "quick/results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["source_provenance"]["valid"] for row in rows if row["kind"] == "local")
    assert all(row["answer_status"] == "not_requested" for row in rows if row["kind"] == "overload")
    for row in rows:
        if row["kind"] == "local" and row["condition"] == "off":
            assert row["context_sha256"] == row["full_context_sha256"]


@pytest.mark.asyncio
async def test_retry_preserves_failed_attempt_and_reexecutes_only_that_condition(tmp_path, setup_protocol):
    settings, specs, workloads, summary, _ = setup_protocol
    answer = CountingAnswer(fail_calls={2})
    output = tmp_path / "retry"
    args = (settings, specs[:1], workloads[:1], output, summary, answer, {"test": True})
    first = await validation.run_validation(*args, allow_estimated=True)
    assert not first["complete"] and len(first["errors"]) == 1
    counts = summary.calls, answer.calls
    await validation.run_validation(*args, allow_estimated=True, resume=True)
    assert (summary.calls, answer.calls) == counts
    final = await validation.run_validation(*args, allow_estimated=True, resume=True, retry_errors=True)
    assert final["complete"] and answer.calls == counts[1] + 1
    assert summary.calls > counts[0]
    attempts = [json.loads(line) for line in (output / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(row["status"] == "error" for row in attempts) == 1
    assert len(attempts) == 7


@pytest.mark.asyncio
async def test_estimated_counts_require_explicit_protocol_opt_in(tmp_path, setup_protocol):
    settings, specs, workloads, summary, answer = setup_protocol
    with pytest.raises(ValueError, match="估算"):
        await validation.run_validation(settings, specs[:1], workloads[:1], tmp_path / "blocked", summary, answer, {})
    assert not summary.calls and not answer.calls
    assert not (tmp_path / "blocked").exists()


@pytest.mark.asyncio
async def test_local_cap_is_frozen_applies_to_both_conditions_and_requires_real_trigger(tmp_path, setup_protocol):
    settings, specs, _, summary, answer = setup_protocol
    # Deliberately synthetic counting only: guarantee a protocol pressure case
    # independently of the real cached tokenizer used by the separate preflight.
    summary.count_tokens = lambda text: max(1, len(text))
    local_specs = [row for row in specs if row["kind"] == "local"]
    workloads = [validation.materialize_local(row, ROOT / "data/demo_docs") for row in local_specs]
    output = tmp_path / "local"
    original_budget = settings.evidence_token_budget
    report = await validation.run_validation(settings, local_specs, workloads, output, summary, answer,
                                              {"test_only": True}, allow_estimated=True)
    assert settings.evidence_token_budget == original_budget
    assert summary.calls == answer.calls == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"]["version"] == 2
    assert manifest["protocol"]["local_evidence_budget_tokens"] == {row["case_id"]: 768 for row in local_specs}
    stored_workloads = [json.loads(line) for line in (output / "workloads.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["evidence_budget_tokens"] == 768 for row in stored_workloads)
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        assert row["budget_tokens"] == row["evidence_budget_tokens"] == 768
        assert row["source_provenance"]["valid"] and row["source_provenance"]["has_context"]
        if row["condition"] == "off":
            assert row["context_sha256"] == row["full_context_sha256"] and row["ratio"] == 1
            assert row["before_tokens"] > row["budget_tokens"] and not row["triggered"]
        else:
            assert row["triggered"] and row["ratio"] < 1 and row["within_budget"]
    assert report["local_compression"] == {"expected_on_conditions": 6, "completed_ok_on_conditions": 6,
        "triggered_on_conditions": 6, "all_expected_triggered": True, "all_sources_literal_and_nonempty": True}
    assert "local_compression_not_triggered" not in report["target_failures"]
    on_row = next(row for row in rows if row["condition"] == "on")
    on_row["triggered"] = False
    untriggered = validation.summarize(rows, manifest)
    assert not untriggered["quality_targets_met"]
    assert "local_compression_not_triggered" in untriggered["target_failures"]
    assert not untriggered["local_compression"]["all_expected_triggered"]
    on_row["triggered"] = True
    on_row["source_provenance"]["valid"] = False
    assert "local_source_provenance_failed" in validation.summarize(rows, manifest)["target_failures"]


@pytest.mark.parametrize("cap", [None, 0, -1, True, 768.5])
def test_local_spec_requires_explicit_valid_budget(tmp_path, cap):
    rows = [json.loads(line) for line in (ROOT / "data/eval/memory_stress.jsonl").read_text(encoding="utf-8").splitlines()]
    next(row for row in rows if row["kind"] == "local")["evidence_budget_tokens"] = cap
    path = tmp_path / "specs.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence_budget_tokens"):
        validation.load_specs(path)


def test_environment_freezes_python_dependencies_sources_and_corpus_without_inference(tmp_path, monkeypatch):
    def forbidden_network(*args, **kwargs):
        raise AssertionError("offline protocol metadata must not contact a model server")

    monkeypatch.setattr(validation.httpx, "get", forbidden_network)
    snapshot = validation.environment_snapshot(Settings(data_dir=tmp_path, provider="offline-test"),
        ROOT / "data/eval/memory_stress.jsonl", ROOT / "data/demo_docs", estimated=True)
    assert snapshot["python"]["version"] and snapshot["versions"]["pydantic"]
    assert snapshot["source_sha256"]["scripts/validate_memory.py"]
    assert snapshot["source_sha256"]["src/agentic_kb/memory.py"]
    assert snapshot["corpus_sha256"]["02_deploy_v2.md"]
    assert snapshot["models"] == {"protocol_only": "offline-test"}
