"""Guard the demo dataset against known semantic leakage and history-free probes."""
import json
import runpy
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from agentic_kb.evaluation import load_dataset


@pytest.fixture(scope="module")
def demo_cases():
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / "scripts" / "build_demo_eval.py"))
    return root, namespace["build"](root)


def test_generated_dataset_matches_builder_and_remains_source_checked(demo_cases):
    root, cases = demo_cases
    assert len(cases) == 120
    assert Counter(case["kind"] for case in cases) == {
        "single_document": 84, "cross_document": 20, "ambiguous": 6,
        "unanswerable": 6, "long_context": 4,
    }
    for split in (None, "dev", "test"):
        path = root / "data" / "eval" / ("demo_cases.jsonl" if split is None else f"demo_{split}.jsonl")
        saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert saved == [case for case in cases if split is None or case["split"] == split]
        assert len(load_dataset(path)) == len(saved)
    assert all(case["review_status"].startswith("ai_assisted_") for case in cases)


def test_known_semantic_variants_share_family_and_split(demo_cases):
    _, cases = demo_cases
    by_id = {case["id"]: case for case in cases}
    for group in (
        (28, 33, 89, 108, 120),  # Writes have resumed: direct old-version rollback is forbidden.
        (26, 91, 109),          # A v1 session secret is not an OAuth client secret.
        (36, 47, 92),           # Read scope cannot authorize creation.
        (19, 55, 57, 95, 117),  # Liveness does not imply dependency readiness.
        (63, 93),               # Create idempotency does not make notifications exactly once.
        (94, 119),              # API capacity is bounded by per-replica connection peaks.
        (25, 87),               # INI-to-environment mapping.
        (14, 118),              # v1 attachment and pagination limits.
    ):
        rows = [by_id[f"demo-{number:03d}"] for number in group]
        assert len({row["family"] for row in rows}) == 1
        assert len({row["split"] for row in rows}) == 1
    splits = defaultdict(set)
    for case in cases:
        splits[case["family"]].add(case["split"])
    assert all(len(values) == 1 for values in splits.values())
    assert any(count > 1 for count in Counter(case["family"] for case in cases).values())
    assert {case["split"] for case in cases} == {"dev", "test"}


def test_long_followups_require_prior_version_state_or_custom_parameters(demo_cases):
    _, cases = demo_cases
    long_cases = {case["id"]: case for case in cases if case["kind"] == "long_context"}
    assert len(long_cases) == 4
    for case in long_cases.values():
        question = case["question"].casefold()
        assert "v1" not in question and "v2" not in question
        assert "恢复写入" not in question and "新增工单" not in question
        assert all(str(number) not in question for number in (37, 80, 15))
        assert len(case["turns"]) >= 2
    version_one = long_cases["demo-118"]
    turns = "\n".join(turn["question"] for turn in version_one["turns"])
    assert "v1" in turns and "v2" not in turns
    assert "request_id" not in turns
    assert "日志存放位置" in turns and "健康检查" in turns
    budget = long_cases["demo-119"]
    setup = budget["turns"][0]["question"]
    assert "37" in setup and "pool_size已调整为4" in setup and "max_overflow为2" in setup
    assert "floor(37/6)=6" in budget["reference"] and "剩余1" in budget["reference"]
