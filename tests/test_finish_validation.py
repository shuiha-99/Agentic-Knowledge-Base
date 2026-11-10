"""Pure coordinator protocol tests: no models, real subprocesses or port 7860."""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("finish_validation_protocol", ROOT / "scripts/finish_validation.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    (root / "data/eval").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "scripts/score_context_recall.py").write_text("# Protocol-only score driver fixture.\n", encoding="utf-8")
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("protocol-only executable placeholder", encoding="utf-8")
    (root / ".env").write_text("KB_DATA_DIR=runtime\nKB_PORT=7860\nKB_PROVIDER=ollama\n", encoding="utf-8")
    rows = [{"id": f"test-{index:03d}", "split": "test", "kind": "single_document" if index <= 77 else "unanswerable",
             "reference": "fixture reference" if index <= 77 else ""} for index in range(1, 89)]
    (root / "data/eval/demo_cases.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    profile = root / "profile.json"
    profile.write_text(json.dumps({"KB_DATA_DIR": "runtime/phase2_corpus", "KB_CHAT_MODEL": "test-main",
                                   "KB_STRUCTURED_MODEL": "test-helper", "KB_OLLAMA_BASE_URL": "http://fixture-chat:11434",
                                   "KB_EVAL_BASE_URL": "http://fixture-judge:11434"}), encoding="utf-8")
    return root, profile, tmp_path / "pipeline"


class FakePage:
    def __init__(self, progress):
        self.progress = progress

    def start(self):
        pass

    def close(self):
        pass


def write_memory_artifacts(output, *, failed_condition=False, ledger_count=40):
    workloads = [{"case_id": f"{kind}-{index}", "kind": kind,
                  "semantic_check": kind == "normal" and index <= 2}
                 for kind, size in (("normal", 4), ("overload", 2), ("local", 2)) for index in range(1, size + 1)]
    workload_text = "".join(pipeline.canonical(row) + "\n" for row in workloads)
    repeats = {row["case_id"]: 1 if row["kind"] == "overload" else 3 for row in workloads}
    protocol = {"quick": False, "conditions": ["off", "on"], "repeats": repeats,
                "expected_jobs": 40, "semantic_answer_jobs": 12}
    locked = {"environment": {"token_count_estimated": False, "models": {"test": "protocol-only"}},
              "protocol": protocol, "workloads_sha256": pipeline.digest(workload_text)}
    fingerprint = pipeline.digest(pipeline.canonical(locked))
    manifest = {**locked, "real_model_validation": True, "fingerprint": fingerprint}
    pipeline.atomic_write(output / "memory/workloads.jsonl", workload_text)
    pipeline.atomic_write(output / "memory/manifest.json", json.dumps(manifest))
    rows = []
    for workload in workloads:
        for repeat in range(1, repeats[workload["case_id"]] + 1):
            for condition in ("off", "on"):
                failed = failed_condition and workload["case_id"] == "normal-1" and repeat == 1 and condition == "on"
                row = {"case_id": workload["case_id"], "kind": workload["kind"], "repeat": repeat,
                       "condition": condition, "status": "error" if failed else "ok", "token_count_estimated": False,
                       "answer_status": "completed" if workload["semantic_check"] and not failed else "not_requested"}
                rows.append((f"{workload['case_id']}/{repeat}/{condition}", row))
    rows = rows[:ledger_count]
    with sqlite3.connect(output / "memory/ledger.sqlite") as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS results (job TEXT PRIMARY KEY, body TEXT)")
        connection.executemany("INSERT OR REPLACE INTO results VALUES (?,?)",
                               [(key, json.dumps(row)) for key, row in rows])
    semantic = sum(row["answer_status"] == "completed" for _, row in rows)
    report = {"complete": not failed_condition and ledger_count == 40, "real_model_validation": True,
              "estimated_token_counts": False, "fingerprint": fingerprint, "expected_jobs": 40,
              "completed_jobs": 40, "semantic_summary": {"expected_answer_jobs": 12, "completed_answer_jobs": semantic},
              "target_failures": ["fixture_model_failure" if failed_condition else "fixture_quality_target_not_met"],
              "errors": [{"error": "fixture model timeout"}] if failed_condition else []}
    pipeline.atomic_write(output / "memory/summary.json", json.dumps(report))


def write_artifacts(name, output, schedule):
    path = output / "benchmark/summary.json"
    report = pipeline.read_json(path, {"stability": {"ids": schedule["ids"], "repeats": 3,
        "aggregate": {"planned_generations": 264, "generation_attempted_all": False,
                      "statuses": {"not_started": 264}, "metrics": {}}}})
    if name == "memory":
        write_memory_artifacts(output)
        return
    elif name == "retrieval":
        report["retrieval_ablation"] = {mode: {"planned_count": 88, "attempted_count": 88, "failed_count": 1}
                                       for mode in ("dense", "hybrid", "rerank", "parent")}
    elif name == "generate":
        report["stability"]["aggregate"].update(generation_attempted_all=True,
                                                statuses={"success": 263, "clarification_required": 1})
    elif name == "prepare_vectors":
        report["vector_preparation"] = {"successful_eligible_generations": 231, "logical_items_with_vector_files": 231}
    elif name == "score":
        report["stability"]["aggregate"]["metrics"] = {
            metric: {"eligible_count": 231, "valid_count": 230 if index else 0,
                     "failed_or_blocked_count": 1 if index else 231, "not_started_count": 0,
                     "mean_valid_only": 0.61 if index else None}
            for index, metric in enumerate(pipeline.METRICS)}
    pipeline.atomic_write(path, json.dumps(report))


def coordinator(project, calls, *, resume=False, fail_stage=None, cleanup=True, page=FakePage, executor=None,
                disable_scoring=False, context_recall_only=False):
    root, profile, output = project
    schedule = pipeline.dataset_schedule(root)

    def execute(stage, command, environment, owner):
        calls.append((stage["name"], command, environment))
        stage["pid"] = 999999991  # Never an actual child process.
        if stage["name"] == fail_stage:
            if fail_stage == "memory":
                pipeline.atomic_write(output / "memory/manifest.json", "{}")
            return 7
        if stage["name"] == "score" and "score_context_recall.py" in Path(command[2]).name:
            write_context_recall_artifacts(root, output)
        else:
            write_artifacts(stage["name"], output, schedule)
        return 0

    def restore(owner):
        calls.append(("restore_ui", None, None))
        return {"status": "protocol_fake_only"}

    return pipeline.Coordinator(root, output, profile, resume=resume, executor=executor or execute,
        status_factory=page, model_unloader=lambda p, r: {"confirmed_empty": cleanup, "services": []},
        ui_restorer=restore, awake_guard=lambda warn: nullcontext(), disable_scoring=disable_scoring,
        context_recall_only=context_recall_only)


def seed_partial_generation_results(output):
    frozen_hash = pipeline.digest("protocol-only original benchmark freeze")
    pipeline.atomic_write(output / "benchmark/manifest.json", json.dumps({"frozen_hash": frozen_hash}))
    generations = [{"phase": "generation", "item": f"agent:test-{index:03d}:repeat0", "case_id": f"test-{index:03d}",
                    "repeat": 0, "generation_hash": pipeline.digest(f"retained-generation-{index}"),
                    "status": "success" if index == 1 else "clarification_required", "answer": f"retained answer {index}",
                    "context_scope": "final_generation_contexts", "retrieved_contexts": [f"retained source {index}"]}
                   for index in (1, 2)]
    records = []
    for generation in generations:
        key = "generation:" + generation["item"]
        records.extend([{"key": key, "state": "started", "payload": None},
                        {"key": key, "state": "completed", "payload": generation}])
    journal_bytes = "".join(pipeline.canonical(row) + "\n" for row in records).encode("utf-8")
    (output / "benchmark/journal.jsonl").write_bytes(journal_bytes)
    report = pipeline.read_json(output / "benchmark/summary.json")
    report["stability"]["aggregate"].update(generation_attempted_all=False,
        statuses={"success": 1, "clarification_required": 1, "not_started": 262})
    pipeline.atomic_write(output / "benchmark/summary.json", json.dumps(report))
    return generations, journal_bytes


def write_context_recall_artifacts(root, output, *, omit_last_score=False):
    path = output / "benchmark/journal.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    generations = [row["payload"] for row in records if row["state"] == "completed"
                   and row["payload"].get("phase") == "generation"]
    items = [{"item": row["item"], "generation_hash": row["generation_hash"]} for row in generations]
    selection = {"metric": "context_recall", "driver_sha256": pipeline.digest(
        (root / "scripts/score_context_recall.py").read_bytes()), "scope": "all",
        "frozen_hash": pipeline.read_json(output / "benchmark/manifest.json")["frozen_hash"],
        "items": items, "generation_snapshot": items, "expected_count": len(items)}
    pipeline.atomic_write(output / "benchmark/context_recall_selection.json", json.dumps(selection))
    scores = generations[:-1] if omit_last_score else generations
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for generation in scores:
            key = f"score:{generation['item']}:{generation['generation_hash']}:context_recall:attempt1"
            payload = {"phase": "score", "item": generation["item"], "case_id": generation["case_id"],
                       "repeat": generation["repeat"], "generation_hash": generation["generation_hash"],
                       "metric": "context_recall", "attempt": 1,
                       "status": "success" if generation["status"] == "success" else "blocked_generation",
                       "score": 0.82 if generation["status"] == "success" else None,
                       "errors": {} if generation["status"] == "success" else {"generation": "clarification required"}}
            for record in ({"key": key, "state": "started", "payload": None},
                           {"key": key, "state": "completed", "payload": payload}):
                stream.write(pipeline.canonical(record) + "\n")
    report = pipeline.read_json(output / "benchmark/summary.json")
    report["stability"]["aggregate"]["metrics"] = {"context_recall": {
        "eligible_count": 231, "valid_count": 1, "failed_or_blocked_count": len(scores) - 1,
        "not_started_count": 231 - len(scores), "mean_valid_only": 0.82}}
    pipeline.atomic_write(output / "benchmark/summary.json", json.dumps(report))


def legacy_controller_definition(output):
    manifest_path = output / "pipeline_manifest.json"
    manifest = pipeline.read_json(manifest_path)
    source = b"# Previous supervisor source: protocol fixture only.\n"
    manifest["frozen"]["coordinator_sha256"] = pipeline.digest(source)
    manifest["fingerprint"] = pipeline.digest(pipeline.canonical(manifest["frozen"]))
    original_bytes = json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8")
    manifest_path.write_bytes(original_bytes)
    (output / "frozen_supervisor.py").write_bytes(source)
    state = pipeline.read_json(output / "pipeline_state.json")
    state["fingerprint"] = manifest["fingerprint"]
    pipeline.atomic_write(output / "pipeline_state.json", json.dumps(state))
    return manifest, original_bytes, source


def test_disable_scoring_initial_run_persists_and_never_spawns_scoring(project):
    calls = []
    assert coordinator(project, calls, disable_scoring=True).run() == 0
    assert [entry[0] for entry in calls] == ["prepare", "memory", "retrieval", "generate", "restore_ui"]
    output = project[2]
    state = pipeline.read_json(output / "pipeline_state.json")
    for stage in state["stages"]:
        if stage["name"] in pipeline.SCORING_STAGES:
            assert stage["status"] == "skipped_by_user" and stage["exitcode"] is None
            assert stage["attempts"] == [] and stage["pid"] is None
    controls = pipeline.read_json(output / "user_controls.json")
    assert controls["scoring_disabled"] and len(controls["changes"]) == 1
    before = (output / "user_plan_changes.jsonl").read_bytes()
    progress = pipeline.Progress(output)
    progress.update(state)
    score = progress.public()["score"]
    assert score == {"completed": 0, "planned": 693, "valid": 0, "failed_or_blocked": 0,
                     "not_started": 693, "disabled": True}
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "停用" in report and "不计为成功或实际失败" in report and "null" in report
    calls.clear()
    assert coordinator(project, calls, resume=True).run() == 0
    assert calls == [] and (output / "user_plan_changes.jsonl").read_bytes() == before


def test_disable_scoring_resumes_existing_run_with_only_controller_migration(project):
    calls = []
    assert coordinator(project, calls, fail_stage="generate").run() == 2
    output = project[2]
    old, manifest_bytes, source = legacy_controller_definition(output)
    preserved = {"manifest.json": b'{"frozen":"benchmark fixture unchanged"}',
                 "answers.jsonl": b'{"id":"retained-answer","answer":"keep actual output"}\n'}
    for name, data in preserved.items():
        (output / "benchmark" / name).write_bytes(data)
    calls.clear()
    assert coordinator(project, calls, resume=True, disable_scoring=True).run() == 0
    assert [entry[0] for entry in calls] == ["generate", "restore_ui"]
    archive = output / "controller_archive"
    assert (archive / (old["fingerprint"] + ".manifest.json")).read_bytes() == manifest_bytes
    assert (archive / (old["frozen"]["coordinator_sha256"] + ".py")).read_bytes() == source
    new = pipeline.read_json(output / "pipeline_manifest.json")
    assert new["fingerprint"] != old["fingerprint"] and new["frozen"]["stages"] == old["frozen"]["stages"]
    for name, data in preserved.items():
        assert (output / "benchmark" / name).read_bytes() == data
    state = pipeline.read_json(output / "pipeline_state.json")
    assert state["fingerprint"] == new["fingerprint"]
    assert all(stage["status"] == "skipped_by_user" for stage in state["stages"]
               if stage["name"] in pipeline.SCORING_STAGES)


@pytest.mark.parametrize("drift", ["profile", "dataset", "command"])
def test_disable_scoring_does_not_authorize_other_frozen_drift(project, drift):
    calls = []
    assert coordinator(project, calls, fail_stage="generate").run() == 2
    output = project[2]
    legacy_controller_definition(output)
    if drift == "profile":
        profile = json.loads(project[1].read_text(encoding="utf-8"))
        profile["KB_CHAT_MODEL"] = "different-model"
        project[1].write_text(json.dumps(profile), encoding="utf-8")
    elif drift == "dataset":
        with (project[0] / "data/eval/demo_cases.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    else:
        manifest = pipeline.read_json(output / "pipeline_manifest.json")
        manifest["frozen"]["stages"][3]["command"].append("--changed-command")
        manifest["fingerprint"] = pipeline.digest(pipeline.canonical(manifest["frozen"]))
        pipeline.atomic_write(output / "pipeline_manifest.json", json.dumps(manifest))
        state = pipeline.read_json(output / "pipeline_state.json")
        state["fingerprint"] = manifest["fingerprint"]
        pipeline.atomic_write(output / "pipeline_state.json", json.dumps(state))
    calls.clear()
    with pytest.raises(ValueError, match="profile|固定计划"):
        coordinator(project, calls, resume=True, disable_scoring=True).run()
    assert calls == [] and not (output / "user_controls.json").exists()


def test_disabled_scoring_is_not_permission_for_a_different_controller(project):
    calls = []
    assert coordinator(project, calls, fail_stage="generate", disable_scoring=True).run() == 2
    legacy_controller_definition(project[2])
    calls.clear()
    with pytest.raises(ValueError, match="coordinator自身hash"):
        coordinator(project, calls, resume=True).run()
    assert calls == []


def test_disable_scoring_requires_original_source_archive(project):
    calls = []
    assert coordinator(project, calls, fail_stage="generate").run() == 2
    legacy_controller_definition(project[2])
    (project[2] / "frozen_supervisor.py").write_bytes(b"wrong source")
    calls.clear()
    with pytest.raises(ValueError, match="缺少.*原manifest"):
        coordinator(project, calls, resume=True, disable_scoring=True).run()
    assert calls == [] and not (project[2] / "user_controls.json").exists()


def test_committed_disable_migration_recovers_after_interruption_without_reauthorizing(project, monkeypatch):
    calls = []
    assert coordinator(project, calls, fail_stage="generate").run() == 2
    legacy_controller_definition(project[2])
    original = pipeline.atomic_write

    def interrupted(path, text):
        if path.name == "pipeline_manifest.json":
            raise OSError("protocol interrupted after atomic control commit")
        return original(path, text)

    monkeypatch.setattr(pipeline, "atomic_write", interrupted)
    calls.clear()
    with pytest.raises(OSError, match="after atomic control commit"):
        coordinator(project, calls, resume=True, disable_scoring=True).run()
    assert pipeline.read_json(project[2] / "user_controls.json")["scoring_disabled"] and calls == []
    monkeypatch.setattr(pipeline, "atomic_write", original)
    assert coordinator(project, calls, resume=True).run() == 0
    assert [entry[0] for entry in calls] == ["generate", "restore_ui"]
    log = (project[2] / "user_plan_changes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 1


@pytest.mark.parametrize("legacy_source", [False, True])
def test_context_recall_switch_stops_generation_unskips_only_score_and_persists(project, legacy_source):
    calls = []
    assert coordinator(project, calls, fail_stage="generate", disable_scoring=True).run() == 2
    root, _profile, output = project
    generations, journal_bytes = seed_partial_generation_results(output)
    benchmark_manifest = (output / "benchmark/manifest.json").read_bytes()
    if legacy_source:
        old, archived_manifest, source = legacy_controller_definition(output)
    else:
        old = pipeline.read_json(output / "pipeline_manifest.json")
        archived_manifest = (output / "pipeline_manifest.json").read_bytes()
        source = Path(pipeline.__file__).read_bytes()
    frozen_commands = old["frozen"]["stages"]
    calls.clear()
    assert coordinator(project, calls, resume=True, context_recall_only=True).run() == 0
    assert [entry[0] for entry in calls] == ["score", "restore_ui"]
    command = calls[0][1]
    old_score = next(stage["command"] for stage in frozen_commands if stage["name"] == "score")
    assert command == [*old_score[:2], str(root / "scripts/score_context_recall.py"), *old_score[4:]]
    assert "score" not in command and "prepare-vectors" not in command
    controls = pipeline.read_json(output / "user_controls.json")
    assert controls["scoring_disabled"] is False and controls["generation_disabled"] is True
    assert controls["score_metrics"] == ["context_recall"]
    assert controls["score_driver_sha256"] == pipeline.digest((root / "scripts/score_context_recall.py").read_bytes())
    state = pipeline.read_json(output / "pipeline_state.json")
    stages = {stage["name"]: stage for stage in state["stages"]}
    assert stages["generate"]["status"] == stages["prepare_vectors"]["status"] == "skipped_by_user"
    assert len(stages["generate"]["attempts"]) == 1 and stages["prepare_vectors"]["attempts"] == []
    assert stages["score"]["status"] == "completed" and len(stages["score"]["attempts"]) == 1
    assert stages["score"]["actual_command"] == command
    current = pipeline.read_json(output / "pipeline_manifest.json")
    assert current["frozen"]["stages"] == frozen_commands
    archive = output / "controller_archive"
    assert (archive / (old["fingerprint"] + ".manifest.json")).read_bytes() == archived_manifest
    assert (archive / (old["frozen"]["coordinator_sha256"] + ".py")).read_bytes() == source
    assert (output / "benchmark/manifest.json").read_bytes() == benchmark_manifest
    assert (output / "benchmark/journal.jsonl").read_bytes().startswith(journal_bytes)
    selection = pipeline.read_json(output / "benchmark/context_recall_selection.json")
    assert selection["expected_count"] == 2 and len(selection["items"]) == 2
    records = [json.loads(line) for line in (output / "benchmark/journal.jsonl").read_text(encoding="utf-8").splitlines()]
    scores = [row["payload"] for row in records if row["state"] == "completed" and row["payload"]["phase"] == "score"]
    assert {score["metric"] for score in scores} == {"context_recall"}
    assert {score["status"] for score in scores} == {"success", "blocked_generation"}
    assert {(score["item"], score["generation_hash"]) for score in scores} == {
        (row["item"], row["generation_hash"]) for row in generations}
    aggregate = pipeline.read_json(output / "benchmark/summary.json")["stability"]["aggregate"]
    assert aggregate["statuses"]["not_started"] == 262 and not aggregate["generation_attempted_all"]
    assert aggregate["metrics"]["context_recall"]["not_started_count"] == 229
    before = (output / "user_plan_changes.jsonl").read_bytes()
    calls.clear()
    assert coordinator(project, calls, resume=True).run() == 0
    assert calls == [] and (output / "user_plan_changes.jsonl").read_bytes() == before


@pytest.mark.parametrize("drift", ["profile", "dataset", "command"])
def test_context_recall_does_not_authorize_unrelated_frozen_drift(project, drift):
    calls = []
    assert coordinator(project, calls, fail_stage="generate", disable_scoring=True).run() == 2
    seed_partial_generation_results(project[2])
    if drift == "profile":
        profile = json.loads(project[1].read_text(encoding="utf-8"))
        profile["KB_CHAT_MODEL"] = "different-model"
        project[1].write_text(json.dumps(profile), encoding="utf-8")
    elif drift == "dataset":
        with (project[0] / "data/eval/demo_cases.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    else:
        manifest = pipeline.read_json(project[2] / "pipeline_manifest.json")
        manifest["frozen"]["stages"][3]["command"].append("--unrelated-change")
        manifest["fingerprint"] = pipeline.digest(pipeline.canonical(manifest["frozen"]))
        pipeline.atomic_write(project[2] / "pipeline_manifest.json", json.dumps(manifest))
        state = pipeline.read_json(project[2] / "pipeline_state.json")
        state["fingerprint"] = manifest["fingerprint"]
        pipeline.atomic_write(project[2] / "pipeline_state.json", json.dumps(state))
    calls.clear()
    before = (project[2] / "user_controls.json").read_bytes()
    with pytest.raises(ValueError, match="profile|固定计划|冻结"):
        coordinator(project, calls, resume=True, context_recall_only=True).run()
    assert calls == [] and (project[2] / "user_controls.json").read_bytes() == before


@pytest.mark.parametrize("explicit_mode", [False, True])
def test_context_recall_driver_drift_is_rejected_on_persistent_resume(project, explicit_mode):
    calls = []
    assert coordinator(project, calls, fail_stage="generate", disable_scoring=True).run() == 2
    seed_partial_generation_results(project[2])
    assert coordinator(project, calls, resume=True, context_recall_only=True).run() == 0
    (project[0] / "scripts/score_context_recall.py").write_text("# Different driver source.\n", encoding="utf-8")
    calls.clear()
    with pytest.raises(ValueError, match="driver|hash|脚本|哈希"):
        coordinator(project, calls, resume=True, context_recall_only=explicit_mode).run()
    assert calls == []


def test_context_recall_needs_each_selected_item_terminal_but_not_all_231(project):
    calls = []
    assert coordinator(project, calls, fail_stage="generate", disable_scoring=True).run() == 2
    seed_partial_generation_results(project[2])
    calls.clear()

    def incomplete(stage, command, environment, owner):
        calls.append((stage["name"], command, environment))
        assert stage["name"] == "score"
        write_context_recall_artifacts(project[0], project[2], omit_last_score=True)
        return 0

    assert coordinator(project, calls, resume=True, context_recall_only=True, executor=incomplete).run() == 2
    state = pipeline.read_json(project[2] / "pipeline_state.json")
    assert next(stage for stage in state["stages"] if stage["name"] == "score")["status"] == "failed"
    assert [entry[0] for entry in calls] == ["score", "restore_ui"]


def test_context_recall_cli_passes_explicit_user_control(project, monkeypatch):
    captured = []

    def constructor(*args, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(run=lambda: 0)

    monkeypatch.setattr(pipeline, "Coordinator", constructor)
    assert pipeline.main(["--profile", str(project[1]), "--output", str(project[2]),
                          "--resume", "--context-recall-only"]) == 0
    assert captured[0]["resume"] and captured[0]["context_recall_only"]
    assert captured[0]["disable_scoring"] is False


def test_commands_freeze_all_test_ids_three_rounds_and_never_use_old_replay(project):
    root, profile, output = project
    schedule = pipeline.dataset_schedule(root)
    stages = pipeline.build_stages(root, output, pipeline.project_python(root), schedule)
    assert [stage["name"] for stage in stages] == list(pipeline.STAGES)
    for stage in stages:
        command = stage["command"]
        assert command[1] == "-u" and "work/finish_delivery.py" not in " ".join(command)
        assert "--retry-failed" not in command and "--retry-errors" not in command and "--limit" not in command
        if stage["name"] != "memory":
            assert command[command.index("--scope") + 1] == "all"
            assert len(command[command.index("--stability-ids") + 1].split(",")) == 88
            assert command[command.index("--stability-repeats") + 1] == "3"
    assert schedule["generation_slots"] == 264 and schedule["metric_slots"] == 693
    assert pipeline.load_profile(profile, root)["KB_DATA_DIR"] == "runtime/phase2_corpus"


def test_success_is_serial_and_report_keeps_null_failures_and_denominators(project):
    calls = []
    owner = coordinator(project, calls)
    assert owner.run() == 0
    assert [entry[0] for entry in calls] == [*pipeline.STAGES, "restore_ui"]
    state = pipeline.read_json(project[2] / "pipeline_state.json")
    assert state["status"] == "completed"
    assert all(stage["status"] == "completed" and stage["exitcode"] == 0 for stage in state["stages"])
    report = (project[2] / "report.md").read_text(encoding="utf-8")
    assert "0/231" in report and "null" in report and "231" in report
    assert "fixture_quality_target_not_met" in report
    assert "质量目标达成判断：未作出" in report
    calls.clear()
    assert coordinator(project, calls, resume=True).run() == 0
    assert calls == []


def test_failed_stage_stops_and_resume_skips_only_completed_phases(project):
    calls = []
    assert coordinator(project, calls, fail_stage="memory").run() == 2
    assert [entry[0] for entry in calls] == ["prepare", "memory", "restore_ui"]
    calls.clear()
    assert coordinator(project, calls, resume=True).run() == 0
    assert [entry[0] for entry in calls] == ["memory", "retrieval", "generate", "prepare_vectors", "score", "restore_ui"]
    assert "--resume" in calls[0][1]
    state = pipeline.read_json(project[2] / "pipeline_state.json")
    assert len(state["stages"][1]["attempts"]) == 2
    assert state["stages"][1]["attempts"][0]["exitcode"] == 7


@pytest.mark.parametrize("scenario", ["terminal_model_failure", "missing_slot", "fatal_exit"])
def test_memory_terminal_failures_continue_but_missing_or_fatal_execution_stops(project, scenario):
    calls = []
    schedule = pipeline.dataset_schedule(project[0])

    def execute(stage, command, environment, owner):
        calls.append((stage["name"], command, environment))
        if stage["name"] == "memory":
            write_memory_artifacts(project[2], failed_condition=True,
                                   ledger_count=39 if scenario == "missing_slot" else 40)
            return 1 if scenario == "fatal_exit" else 2
        write_artifacts(stage["name"], project[2], schedule)
        return 0

    result = coordinator(project, calls, executor=execute).run()
    state = pipeline.read_json(project[2] / "pipeline_state.json")
    memory = state["stages"][1]
    if scenario == "terminal_model_failure":
        assert result == 0 and state["status"] == "completed"
        assert [entry[0] for entry in calls] == [*pipeline.STAGES, "restore_ui"]
        assert memory["status"] == "completed" and memory["exitcode"] == 2
        assert memory["attempts"][-1]["exitcode"] == 2
        assert memory["measurement_failures"]
        assert memory["measurement_summary"]["semantic_planned"] == 12
        assert memory["measurement_summary"]["semantic_completed"] == 11
        assert any(row["code"] == "memory_complete_with_measurement_failures" for row in state["warnings"])
        assert "fixture model timeout" in (project[2] / "report.md").read_text(encoding="utf-8")
    else:
        assert result == 2 and state["status"] == "failed" and memory["status"] == "failed"
        assert memory["exitcode"] == (1 if scenario == "fatal_exit" else 2)
        assert [entry[0] for entry in calls] == ["prepare", "memory", "restore_ui"]


def test_memory_fingerprint_mismatch_stops_even_with_all_terminal_slots(project):
    calls = []
    schedule = pipeline.dataset_schedule(project[0])

    def execute(stage, command, environment, owner):
        calls.append((stage["name"], command, environment))
        write_artifacts(stage["name"], project[2], schedule)
        if stage["name"] == "memory":
            path = project[2] / "memory/summary.json"
            report = pipeline.read_json(path)
            report["fingerprint"] = "mismatched"
            pipeline.atomic_write(path, json.dumps(report))
        return 0

    assert coordinator(project, calls, executor=execute).run() == 2
    assert [entry[0] for entry in calls] == ["prepare", "memory", "restore_ui"]


def test_profile_drift_and_live_orphan_child_refuse_without_new_processes(project, monkeypatch):
    calls = []
    assert coordinator(project, calls, fail_stage="memory").run() == 2
    state_path = project[2] / "pipeline_state.json"
    state = pipeline.read_json(state_path)
    state["stages"][1].update(status="running", pid=424242)
    pipeline.atomic_write(state_path, json.dumps(state))
    monkeypatch.setattr(pipeline, "pid_alive", lambda pid: pid == 424242)
    calls.clear()
    with pytest.raises(RuntimeError, match="存活"):
        coordinator(project, calls, resume=True).run()
    assert calls == []
    monkeypatch.setattr(pipeline, "pid_alive", lambda pid: False)
    profile = json.loads(project[1].read_text(encoding="utf-8"))
    profile["KB_CHAT_MODEL"] = "changed-profile"
    project[1].write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="hash|profile"):
        coordinator(project, calls, resume=True).run()
    assert calls == []


def test_exclusive_lock_rejects_second_coordinator(project):
    path = project[2] / ".coordinator.lock"
    with pipeline.exclusive_lock(path):
        with pytest.raises(RuntimeError, match="并行"):
            with pipeline.exclusive_lock(path):
                pytest.fail("second lock acquired")


def test_cleanup_not_confirmed_blocks_ui_and_resume_only_retries_cleanup(project):
    calls = []
    assert coordinator(project, calls, cleanup=False).run() == 2
    assert [entry[0] for entry in calls] == list(pipeline.STAGES)
    state = pipeline.read_json(project[2] / "pipeline_state.json")
    assert state["status"] == "completed_cleanup_failed"
    assert state["ui_restore"]["status"] == "not_started_unload_unconfirmed"
    calls.clear()
    assert coordinator(project, calls, resume=True, cleanup=True).run() == 0
    assert [entry[0] for entry in calls] == ["restore_ui"]


@pytest.mark.parametrize("confirmed", [True, False])
def test_cancel_stops_owned_tree_before_unload_or_ui(project, confirmed):
    calls = []

    class Tree:
        def stop_and_confirm(self):
            calls.append(("stop_owned_tree", None, None))
            return confirmed

        def close(self):
            calls.append(("close_tree_handle", None, None))

    def cancel(stage, command, environment, owner):
        owner.active_process = SimpleNamespace(pid=31337)
        owner.active_tree = Tree()
        raise KeyboardInterrupt

    owner = coordinator(project, calls, executor=cancel)

    def unload(profile, root):
        calls.append(("unload_profile_models", None, None))
        return {"confirmed_empty": True}

    owner.model_unloader = unload
    assert owner.run() == 130
    names = [item[0] for item in calls]
    assert names[:2] == ["stop_owned_tree", "close_tree_handle"]
    if confirmed:
        assert names[2:] == ["unload_profile_models", "restore_ui"]
    else:
        assert "unload_profile_models" not in names and "restore_ui" not in names
        state = pipeline.read_json(project[2] / "pipeline_state.json")
        assert not state["resource_cleanup"]["child_processes_confirmed_stopped"]
        with pytest.raises(RuntimeError, match="worker树退出未确认"):
            coordinator(project, [], resume=True).run()


@pytest.mark.parametrize("confirmed", [True, False])
def test_only_profile_services_are_unloaded_without_prompts(project, monkeypatch, confirmed):
    calls, seen = [], {}

    def get(url, **kwargs):
        assert kwargs == {"timeout": 10, "trust_env": False}
        calls.append(("get", url, None))
        seen[url] = seen.get(url, 0) + 1
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "models": [{"name": "resident-test"}] if seen[url] == 1 or not confirmed else []})

    def post(url, **kwargs):
        assert kwargs["json"] == {"model": "resident-test", "keep_alive": 0}
        assert "prompt" not in kwargs["json"] and kwargs["trust_env"] is False and kwargs["timeout"] == 60
        calls.append(("post", url, kwargs["json"]))
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(pipeline.httpx, "get", get)
    monkeypatch.setattr(pipeline.httpx, "post", post)
    profile = pipeline.load_profile(project[1], project[0])
    result = pipeline.unload_profile_models(profile, project[0])
    assert result["confirmed_empty"] is confirmed
    assert len(calls) == 6
    assert {entry[1].split("/api/")[0] for entry in calls} == {"http://fixture-chat:11434", "http://fixture-judge:11434"}


def test_awake_request_is_system_only_and_always_restored():
    flags, warnings = [], []

    def setter(value):
        flags.append(value)
        return 1

    with pytest.raises(RuntimeError):
        with pipeline.system_awake(lambda *args: warnings.append(args), setter=setter):
            raise RuntimeError("protocol cancellation")
    assert flags == [0x80000001, 0x80000000]
    assert not warnings and all(not value & 0x2 for value in flags)


def test_awake_failure_warns_and_continues_without_power_plan_changes():
    flags, warnings = [], []

    def setter(value):
        flags.append(value)
        return 0

    with pipeline.system_awake(lambda *args: warnings.append(args), setter=setter):
        reached = True
    assert reached and flags == [0x80000001, 0x80000000]
    assert [row[0] for row in warnings] == ["system_awake_request_failed", "system_awake_restore_failed"]


def test_execute_exception_closes_owned_process_tree_before_unload(project, monkeypatch):
    events = []

    class Process:
        pid = 515151
        returncode = None

        def poll(self):
            raise RuntimeError("scripted process polling failure")

    class Tree:
        def attach_suspended(self, process):
            events.append("attach")

        def stop_and_confirm(self):
            events.append("stop_tree")
            return True

        def close(self):
            events.append("close_tree")

    def fake_popen(*args, **kwargs):
        events.append("popen")
        if os.name == "nt":
            assert kwargs["creationflags"] & 0x00000004
        return Process()

    monkeypatch.setattr(pipeline.subprocess, "Popen", fake_popen)
    owner = coordinator(project, [])
    owner.executor = None
    owner.process_tree_factory = Tree
    owner.model_unloader = lambda p, r: events.append("unload") or {"confirmed_empty": True}
    owner.ui_restorer = lambda p: events.append("ui") or {"status": "protocol_fake"}
    assert owner.run() == 2
    assert events == ["popen", "attach", "stop_tree", "close_tree", "unload", "ui"]


def test_status_page_is_get_only_counts_not_config_or_files(tmp_path):
    progress = pipeline.Progress(tmp_path)
    journal = tmp_path / "benchmark/journal.jsonl"
    journal.parent.mkdir()
    records = [{"state": "completed", "payload": {"phase": "generation", "item": "case1", "status": "success",
                                                    "answer": "PRIVATE-CONFIG-MARKER"}},
               {"state": "completed", "payload": {"phase": "score", "item": "case1", "metric": "faithfulness", "status": "error"}}]
    journal.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    memory = tmp_path / "memory/ledger.sqlite"
    memory.parent.mkdir()
    with sqlite3.connect(memory) as connection:
        connection.execute("CREATE TABLE results (body TEXT)")
        connection.execute("INSERT INTO results VALUES (?)", (json.dumps({"condition": "on", "status": "ok",
            "triggered": True, "ratio": 0.63, "answer_status": "completed", "context": "PRIVATE-CONFIG-MARKER"}),))
    progress.update({"status": "running", "current_stage": "score", "profile": "PRIVATE-CONFIG-MARKER"})
    page = pipeline.StatusPage(progress, port=0)
    page.start()
    try:
        base = f"http://127.0.0.1:{page.server.server_address[1]}"
        response = httpx.get(base + "/api/progress", trust_env=False)
        assert response.json()["generation"]["completed"] == 1
        assert response.json()["score"]["failed_or_blocked"] == 1
        assert response.json()["memory"]["semantic_completed"] == 1
        assert "PRIVATE-CONFIG-MARKER" not in response.text
        html = httpx.get(base + "/", trust_env=False).text
        assert "处理完成不代表答案正确" in html and "有效完成" in html
        assert "JSON.stringify" not in html
        assert httpx.get(base + "/logs", trust_env=False).status_code == 404
        assert httpx.get(base + "/api/progress?file=.env", trust_env=False).status_code == 404
        assert httpx.post(base + "/api/progress", trust_env=False).status_code != 200
    finally:
        page.close()


def test_launch_confirms_actual_pid_with_launch_id_and_ui_env_clears_profile(project, monkeypatch, capsys):
    root, profile, output = project
    captured = []
    monkeypatch.setattr(pipeline, "ROOT", root)
    monkeypatch.setenv("KB_CHAT_MODEL", "ambient-wrong-model")

    def popen(command, **kwargs):
        captured.append((command, kwargs))
        launch_id = command[command.index("--launch-id") + 1]
        pipeline.atomic_write(output / "pipeline_state.json", json.dumps({
            "coordinator_pid": 88124, "launch_id": launch_id, "status": "running"}))
        return SimpleNamespace(pid=88123, poll=lambda: None)

    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    args = SimpleNamespace(profile=profile, output=output, resume=False)
    assert pipeline.launch(args) == 0
    command, kwargs = captured[0]
    assert command[1] == "-u" and "--launch" not in command
    assert not any(key.startswith("KB_") for key in kwargs["env"])
    if os.name == "nt":
        assert kwargs["creationflags"] == pipeline.subprocess.CREATE_NO_WINDOW
    confirmed = json.loads(capsys.readouterr().out)
    assert confirmed["coordinator_pid"] == 88124 and confirmed["launcher_pid"] == 88123
    assert confirmed["status"] == "running"


def test_launch_rejects_old_state_even_when_redirector_pid_matches(project, monkeypatch):
    root, profile, output = project
    monkeypatch.setattr(pipeline, "ROOT", root)
    polls = iter([None, 1])

    def popen(command, **kwargs):
        pipeline.atomic_write(output / "pipeline_state.json", json.dumps({
            "coordinator_pid": 88123, "launch_id": "previous-start", "status": "running"}))
        return SimpleNamespace(pid=88123, poll=lambda: next(polls), returncode=1)

    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    monkeypatch.setattr(pipeline.time, "sleep", lambda value: None)
    with pytest.raises(RuntimeError, match="启动失败"):
        pipeline.launch(SimpleNamespace(profile=profile, output=output, resume=True))


@pytest.mark.skipif(os.name != "nt", reason="Windows Job protocol")
@pytest.mark.parametrize("resume_count", [1, 0, 2, 0xFFFFFFFF])
def test_suspended_process_is_bound_before_resume_and_thread_handle_always_closed(resume_count):
    events = []
    tree = pipeline.OwnedProcessTree.__new__(pipeline.OwnedProcessTree)
    tree.process, tree.handle, tree.attached = None, 71, False
    tree._suspended_primary_thread = lambda process: events.append("open_own_primary_thread") or 83
    tree.kernel = SimpleNamespace(
        AssignProcessToJobObject=lambda job, handle: events.append(("assign_job", job, handle)) or True,
        ResumeThread=lambda thread: events.append(("resume_thread", thread)) or resume_count,
        CloseHandle=lambda handle: events.append(("close_thread", handle)) or True,
    )
    process = SimpleNamespace(_handle=91)
    if resume_count == 1:
        tree.attach_suspended(process)
    else:
        with pytest.raises(OSError, match="挂起计数异常"):
            tree.attach_suspended(process)
    assert tree.attached and tree.process is process
    assert events == ["open_own_primary_thread", ("assign_job", 71, 91),
                      ("resume_thread", 83), ("close_thread", 83)]


@pytest.mark.skipif(os.name != "nt", reason="Windows Job protocol")
def test_job_empty_waits_for_held_parent_handle_instead_of_polling_once():
    events = []
    tree = pipeline.OwnedProcessTree.__new__(pipeline.OwnedProcessTree)
    tree.handle, tree.attached = 71, True
    tree.process = SimpleNamespace(poll=lambda: None, wait=lambda **kwargs: events.append(kwargs) or 1)
    tree.Accounting = lambda: SimpleNamespace(active=0)
    tree.ctypes = SimpleNamespace(byref=lambda value: value, sizeof=lambda value: 0)
    tree.kernel = SimpleNamespace(QueryInformationJobObject=lambda *args: True)
    assert tree.stop_and_confirm()
    assert len(events) == 1 and 0 < events[0]["timeout"] <= 10


@pytest.mark.skipif(os.name != "nt", reason="Windows Job protocol")
def test_parent_handle_wait_timeout_is_unconfirmed():
    tree = pipeline.OwnedProcessTree.__new__(pipeline.OwnedProcessTree)
    tree.handle, tree.attached = 71, True

    def wait(**kwargs):
        raise pipeline.subprocess.TimeoutExpired("own-process", kwargs["timeout"])

    tree.process = SimpleNamespace(poll=lambda: None, wait=wait)
    tree.Accounting = lambda: SimpleNamespace(active=0)
    tree.ctypes = SimpleNamespace(byref=lambda value: value, sizeof=lambda value: 0)
    tree.kernel = SimpleNamespace(QueryInformationJobObject=lambda *args: True)
    assert not tree.stop_and_confirm()


def test_ui_restore_only_on_free_port_uses_original_env(project, monkeypatch):
    owner = coordinator(project, [])
    owner.ui_restorer = None
    owner.state = {"status": "completed", "current_stage": None, "warnings": [], "stages": []}
    (project[2] / "logs").mkdir(parents=True)
    captured = []
    monkeypatch.setenv("KB_DATA_DIR", "ambient-benchmark-dir")
    monkeypatch.setattr(pipeline, "port_busy", lambda: True)
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **k: pytest.fail("port busy must not spawn"))
    owner.restore_ui(pipeline.project_python(project[0]))
    assert owner.state["ui_restore"]["status"] == "skipped_port_busy"
    values = iter([False, True])
    monkeypatch.setattr(pipeline, "port_busy", lambda: next(values))

    def spawn(command, **kwargs):
        captured.append((command, kwargs))
        return SimpleNamespace(pid=11773, poll=lambda: None)

    monkeypatch.setattr(pipeline.subprocess, "Popen", spawn)
    owner.restore_ui(pipeline.project_python(project[0]))
    assert owner.state["ui_restore"]["status"] == "listening"
    assert captured[0][0][-3:] == ["-m", "agentic_kb", "serve"]
    assert not any(key.startswith("KB_") for key in captured[0][1]["env"])
