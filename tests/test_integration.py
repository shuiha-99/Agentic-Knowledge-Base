"""Exercise the real storage/graph/UI contracts together, without model downloads."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from agentic_kb import cli
from agentic_kb.config import Settings
from agentic_kb.graph import GraphEngine
from agentic_kb.providers import build_providers
from agentic_kb.retrieval import KnowledgeBase


@pytest.mark.asyncio
async def test_modules_integrate_and_resume(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "runtime", provider="offline-test")
    llm, embedder, reranker = build_providers(settings)
    kb = KnowledgeBase(settings, embedder, reranker)
    source = tmp_path / "deploy.md"
    source.write_text("# AtlasDesk v2\n\n## 部署\nAtlasDesk v2 默认端口8088，使用环境变量配置。", encoding="utf-8")
    try:
        kb.ingest([source])
        engine = GraphEngine(settings, llm, kb)
        events = [event async for event in engine.astream("AtlasDesk v2默认端口是多少？", "integration")]
        assert not [event for event in events if event["type"] == "error"]
        result = next(event for event in events if event["type"] == "done")
        assert "离线流程测试" in result["answer"]
        assert result["metrics"]["tasks"] == 1
        assert any(event["type"] == "sources" and event["sources"] for event in events)
        assert result["metrics"]["quality_metrics_computed"] is False
        assert kb.last_trace["used_sparse"]
        paused = [event async for event in engine.astream("这个版本怎么升级", "clarify-integration")]
        assert any(event["type"] == "clarification" for event in paused)
        # A fresh engine uses the same durable SQLite checkpoint.
        resumed = [event async for event in GraphEngine(settings, llm, kb).astream(
            None, "clarify-integration", clarification="AtlasDesk v1.8到v2.0")]
        assert any(event["type"] == "done" for event in resumed)
        assert not any(event["type"] == "error" for event in resumed)
    finally:
        kb.close()


def test_settings_reject_invalid_budget():
    with pytest.raises(ValueError, match="overlap"):
        Settings(child_chunk_size=100, child_chunk_overlap=100)
    with pytest.raises(ValueError, match="output reserve"):
        Settings(context_window_tokens=1024, output_reserve_tokens=1024)


def test_doctor_is_readonly_and_separates_isolated_ragas_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "not-created"
    python = tmp_path / "isolated-python.exe"
    python.write_bytes(b"fixture interpreter; never executed")
    monkeypatch.setenv("KB_DATA_DIR", str(target))
    monkeypatch.setenv("KB_RAGAS_PYTHON", str(python))
    monkeypatch.setenv("KB_STRUCTURED_MODEL", "helper")
    monkeypatch.setenv("KB_STRUCTURED_TOKENIZER_MODEL", "helper-tokenizer")
    monkeypatch.setenv("KB_STRUCTURED_MODEL_SCHEMAS", "ConversationSummary,EvidenceDigest")
    import agentic_kb.runtime as runtime
    monkeypatch.setattr(runtime, "ollama_status", lambda settings: {"available": True, "models": []})
    calls = []

    def query_versions(arguments, **kwargs):
        calls.append(arguments)
        assert "from importlib.metadata" in arguments[-1]
        assert "import ragas" not in arguments[-1]
        assert "-I" in arguments and kwargs["timeout"] == 10
        return SimpleNamespace(returncode=0, stdout=json.dumps({"python_version": "3.11.9", "packages": {
            "ragas": "0.3.3", "langchain-core": "0.3.83", "langchain-community": "0.3.31", "langchain-ollama": "0.3.10"}}), stderr="")

    monkeypatch.setattr(cli.subprocess, "run", query_versions)
    result = CliRunner().invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "ragas" not in payload["main_environment"]["dependencies"]
    isolated = payload["isolated_ragas_environment"]
    assert isolated["environment_exists"] and isolated["ragas_version_matches"]
    assert isolated["packages"]["ragas"] == "0.3.3"
    assert isolated["runtime_import_checked"] is False
    assert payload["structured_model"] == "helper"
    assert payload["structured_tokenizer_model"] == "helper-tokenizer"
    assert payload["structured_model_schemas"] == "ConversationSummary,EvidenceDigest"
    assert len(calls) == 1
    assert not target.exists()


def test_doctor_reports_missing_environment_and_metadata_timeout(tmp_path, monkeypatch):
    missing = tmp_path / "missing-python.exe"
    monkeypatch.setenv("KB_RAGAS_PYTHON", str(missing))
    assert cli._isolated_ragas_metadata()["status"] == "environment_missing"
    missing.write_bytes(b"fixture; never executed")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("metadata-only-query", 10)

    monkeypatch.setattr(cli.subprocess, "run", timeout)
    result = cli._isolated_ragas_metadata()
    assert result["environment_exists"]
    assert result["status"] == "metadata_query_timeout"
    assert result["runtime_import_checked"] is False


def test_prepare_models_explicitly_prepares_helper_tokenizer_without_inference(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, structured_model="helper", structured_tokenizer_model="helper-tokenizer")
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: settings))
    calls = []
    llm = SimpleNamespace(count_tokens=lambda text: calls.append("main-tokenizer"),
                          count_structured_tokens=lambda text: calls.append("helper-tokenizer"))

    def encode(texts):
        calls.append("embedding-fixture")
        return [SimpleNamespace(dense=[1.0, 0.0], sparse_indices=[1])]

    embedder = SimpleNamespace(encode=encode, close=lambda: calls.append("embedding-close"))
    reranker = SimpleNamespace(score=lambda *args: calls.append("reranker-fixture"), close=lambda: calls.append("reranker-close"))
    import agentic_kb.providers as providers
    monkeypatch.setattr(providers, "build_providers", lambda settings: (llm, embedder, reranker))
    result = CliRunner().invoke(cli.app, ["prepare-models"])
    assert result.exit_code == 0, result.output
    assert calls == ["main-tokenizer", "helper-tokenizer", "embedding-fixture", "reranker-fixture", "embedding-close", "reranker-close"]
    assert "helper-tokenizer" in result.output


def test_prepare_models_rejects_missing_helper_tokenizer_before_model_creation(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, structured_model="helper", structured_tokenizer_model=None)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: settings))
    import agentic_kb.providers as providers

    def forbidden(settings):
        raise AssertionError("A model must not be created for incomplete configuration")

    monkeypatch.setattr(providers, "build_providers", forbidden)
    result = CliRunner().invoke(cli.app, ["prepare-models"])
    assert result.exit_code == 2
    assert "KB_STRUCTURED_TOKENIZER_MODEL" in result.output
