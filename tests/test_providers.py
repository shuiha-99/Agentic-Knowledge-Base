"""Budget, cancellation and low-memory failure boundaries without model downloads."""
import asyncio
import sqlite3
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agentic_kb.config import Settings
from agentic_kb.providers import (
    _HEAVY_MODEL_LOCK,
    BGEEmbedder,
    CrossEncoderReranker,
    OllamaLLM,
    ProviderUnavailable,
    _async_model_slot,
    _decompress_cache,
    _inference_key,
    _local_model_source,
)


class TinySchema(BaseModel):
    value: int


def client(**kwargs):
    llm = OllamaLLM(Settings(context_window_tokens=2048, output_reserve_tokens=128, **kwargs))
    llm._tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text))
    return llm


def test_main_and_helper_clients_bypass_environment_proxies(tmp_path, monkeypatch):
    import langchain_ollama.chat_models as chat_models

    created = []

    def fake_client(kind):
        def construct(*, host, **options):
            created.append((kind, host, options))
            return SimpleNamespace()
        return construct

    # Keep ChatOllama's real constructor and intercept both transports before
    # any HTTP client or model request can be created.
    monkeypatch.setattr(chat_models, "Client", fake_client("sync"))
    monkeypatch.setattr(chat_models, "AsyncClient", fake_client("async"))
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9999")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:9999")
    settings = Settings(data_dir=tmp_path, ollama_base_url="http://127.0.0.1:11435",
                        request_timeout_seconds=37, structured_model="helper",
                        structured_tokenizer_model="helper-tokenizer")

    llm = OllamaLLM(settings)

    expected = {"timeout": 37, "trust_env": False}
    assert llm._chat.client_kwargs == expected
    assert llm._structured_chat.client_kwargs == expected
    assert created == [(kind, settings.ollama_base_url, expected)
                       for kind in ("sync", "async", "sync", "async")]


@pytest.mark.asyncio
async def test_retries_do_not_accumulate_prompt():
    llm = client(structured_retries=2)
    seen = []

    async def invoke(messages):
        seen.append(messages[-1].content)
        if len(seen) < 3:
            raise ValueError("invalid output")
        return TinySchema(value=7)

    llm._chat = SimpleNamespace(with_structured_output=lambda *a, **kw: SimpleNamespace(ainvoke=invoke))
    result = await llm.structured("system", "question", TinySchema)
    assert result.value == 7
    assert len(seen) == 3
    assert seen[1] == seen[2]


@pytest.mark.asyncio
async def test_waiting_for_heavy_slot_is_timed_and_cancellable():
    llm = client(request_timeout_seconds=0.03)
    _HEAVY_MODEL_LOCK.acquire()
    try:
        with pytest.raises(ProviderUnavailable):
            await llm.text("system", "question")
    finally:
        _HEAVY_MODEL_LOCK.release()
    async with _async_model_slot(True):
        assert _HEAVY_MODEL_LOCK.locked()
    assert not _HEAVY_MODEL_LOCK.locked()


@pytest.mark.asyncio
async def test_cancelling_active_stream_releases_slot():
    llm = client()
    started = asyncio.Event()

    async def stream(messages):
        started.set()
        await asyncio.Event().wait()
        yield SimpleNamespace(content="unused")

    llm._chat = SimpleNamespace(astream=stream)

    async def consume():
        async for _ in llm.astream("system", "question"):
            pass

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not _HEAVY_MODEL_LOCK.locked()


def test_embedding_failure_unloads_and_releases_slot(tmp_path):
    settings = Settings(data_dir=tmp_path)
    embedder = BGEEmbedder(settings)

    def broken(*args, **kwargs):
        raise RuntimeError("model inference failed")

    embedder._model = SimpleNamespace(encode=broken)
    with pytest.raises(RuntimeError, match="inference failed"):
        embedder.encode(["test"])
    assert embedder._model is None
    assert not _HEAVY_MODEL_LOCK.locked()


@pytest.fixture
def fake_models(monkeypatch):
    """Explicit model doubles measure plumbing calls, never semantic quality."""
    record = SimpleNamespace(embedding_loads=[], encoding_calls=[], reranker_loads=[], scoring_calls=[])

    def embedding_factory(source, **options):
        record.embedding_loads.append((source, options))

        def encode(texts, **kwargs):
            record.encoding_calls.append((list(texts), kwargs))
            return {"dense_vecs": [[float(len(text)), *([0.0] * 1023)] for text in texts],
                    "lexical_weights": [{1: float(len(text))} for text in texts]}

        return SimpleNamespace(encode=encode)

    def reranker_factory(source, **options):
        record.reranker_loads.append((source, options))

        def compute_score(pairs, **kwargs):
            record.scoring_calls.append((list(pairs), kwargs))
            return [len(pair[1]) / 100 for pair in pairs]

        return SimpleNamespace(compute_score=compute_score)

    monkeypatch.setitem(sys.modules, "FlagEmbedding", SimpleNamespace(
        BGEM3FlagModel=embedding_factory, FlagReranker=reranker_factory))
    return record


def test_embedding_cache_deduplicates_misses_and_returns_independent_vectors(tmp_path, fake_models):
    embedder = BGEEmbedder(Settings(data_dir=tmp_path))
    try:
        first = embedder.encode(["a", "a", "bb"])
        assert fake_models.encoding_calls[0][0] == ["a", "bb"]
        assert embedder.last_call_stats["cache_misses"] == 2
        assert embedder.last_call_stats["batch_duplicates"] == 1
        assert embedder.last_call_stats["computed"] == 2
        assert embedder._model is None
        first[0].dense[0] = 999
        assert first[1].dense[0] == 1
        second = embedder.encode(["a", "ccc"])
        assert second[0].dense[0] == 1
        assert fake_models.encoding_calls[-1][0] == ["ccc"]
        assert embedder.last_call_stats["cache_hits"] == 1
        assert embedder.last_call_stats["cache_misses"] == 1
        assert len(fake_models.embedding_loads) == 2
    finally:
        embedder.close()


def test_persistent_embedding_cache_isolated_by_model_and_precision(tmp_path, fake_models):
    configuration = Settings(data_dir=tmp_path)
    embedder = BGEEmbedder(configuration)
    embedder.encode(["same"])
    embedder.close()
    restarted = BGEEmbedder(configuration)
    try:
        assert restarted.encode(["same"])[0].dense[0] == 4
        assert restarted.last_call_stats["disk_hits"] == 1
        assert restarted.last_call_stats["model_loads"] == 0
    finally:
        restarted.close()
    other = BGEEmbedder(configuration.model_copy(update={"embedding_model": "BAAI/different"}))
    other.encode(["same"])
    assert other.last_call_stats["cache_misses"] == 1
    other.close()
    precision = BGEEmbedder(configuration.model_copy(update={"device": "cuda:0"}))
    precision.encode(["same"])
    assert precision.last_call_stats["cache_misses"] == 1
    precision.close()
    assert len(fake_models.embedding_loads) == 3


def test_pair_cache_deduplicates_and_is_query_specific(tmp_path, fake_models):
    settings = Settings(data_dir=tmp_path)
    reranker = CrossEncoderReranker(settings)
    try:
        assert reranker.score("query", ["a", "a", "bb"]) == [0.01, 0.01, 0.02]
        assert fake_models.scoring_calls[0][0] == [["query", "a"], ["query", "bb"]]
        assert fake_models.scoring_calls[0][1]["batch_size"] == 2
        assert reranker.last_call_stats["batch_duplicates"] == 1
        assert reranker.score("query", ["bb", "a"]) == [0.02, 0.01]
        assert reranker.last_call_stats["cache_hits"] == 2
        assert reranker.last_call_stats["model_loads"] == 0
        reranker.score("different query", ["a"])
        assert reranker.last_call_stats["cache_misses"] == 1
        assert reranker._model is None
    finally:
        reranker.close()
    restarted = CrossEncoderReranker(settings)
    try:
        assert restarted.score("query", ["a"]) == [0.01]
        assert restarted.last_call_stats["disk_hits"] == 1
        assert len(fake_models.reranker_loads) == 2
    finally:
        restarted.close()


def test_cache_bounds_apply_to_disk_and_hot_entries(tmp_path, fake_models):
    embedder = BGEEmbedder(Settings(data_dir=tmp_path), cache_capacity=2, memory_cache_size=1)
    try:
        assert len(embedder.encode(["a", "bb", "ccc"])) == 3
        assert len(embedder._cache._memory) == 1
        count = embedder._cache._database().execute(
            "SELECT COUNT(*) FROM inference_cache WHERE kind='embedding'").fetchone()[0]
        assert count == 2
        embedder.encode(["a"])
        assert embedder.last_call_stats["cache_misses"] == 1
    finally:
        embedder.close()


def test_invalid_cache_recomputes_real_output(tmp_path, fake_models):
    embedder = BGEEmbedder(Settings(data_dir=tmp_path))
    try:
        key = _inference_key(embedder._cache_fingerprint, "a")
        embedder._cache.put(key, zlib.compress(b"{}"))
        assert embedder.encode(["a"])[0].dense[0] == 1
        assert embedder.last_call_stats["cache_hits"] == 0
        assert embedder.last_call_stats["computed"] == 1
        with pytest.raises(ValueError, match="大小限制"):
            _decompress_cache(zlib.compress(b"x" * (3 * 1024 * 1024)))
    finally:
        embedder.close()


def test_unavailable_cache_does_not_replace_actual_inference(tmp_path, fake_models, monkeypatch):
    embedder = BGEEmbedder(Settings(data_dir=tmp_path))

    def broken_database():
        raise sqlite3.OperationalError("fixture cache unavailable")

    monkeypatch.setattr(embedder._cache, "_database", broken_database)
    try:
        assert embedder.encode(["a"])[0].dense[0] == 1
        assert embedder.last_call_stats["computed"] == 1
        assert embedder.last_call_stats["cache_hits"] == 0
    finally:
        embedder.close()


def test_bulk_session_reuses_one_load_then_unloads(tmp_path, fake_models):
    embedder = BGEEmbedder(Settings(data_dir=tmp_path))
    try:
        with embedder.batch_session():
            embedder.encode(["first"])
            assert _HEAVY_MODEL_LOCK.locked()
            assert embedder._model is not None
            with embedder.batch_session():
                embedder.encode(["second"])
        assert len(fake_models.embedding_loads) == 1
        assert embedder._model is None
        assert not _HEAVY_MODEL_LOCK.locked()
        with pytest.raises(RuntimeError, match="fixture bulk failure"):
            with embedder.batch_session():
                embedder.encode(["third"])
                raise RuntimeError("fixture bulk failure")
        assert embedder._model is None
        assert not _HEAVY_MODEL_LOCK.locked()
    finally:
        embedder.close()


def test_same_query_concurrently_computes_once_and_has_thread_local_stats(tmp_path, fake_models):
    embedder = BGEEmbedder(Settings(data_dir=tmp_path))

    def encode():
        result = embedder.encode(["same"])
        return result[0].dense[0], embedder.last_call_stats

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: encode(), range(2)))
        assert all(value == 4 for value, _ in results)
        assert sorted(stats["cache_hits"] for _, stats in results) == [0, 1]
        assert len(fake_models.encoding_calls) == 1
    finally:
        embedder.close()


def test_sync_model_wait_has_a_bound_and_does_not_release_other_owner(tmp_path, fake_models):
    embedder = BGEEmbedder(Settings(data_dir=tmp_path, request_timeout_seconds=0.02))
    _HEAVY_MODEL_LOCK.acquire()
    try:
        with pytest.raises(ProviderUnavailable, match="内存槽超时"):
            embedder.encode(["a"])
        assert _HEAVY_MODEL_LOCK.locked()
        assert not fake_models.embedding_loads
    finally:
        _HEAVY_MODEL_LOCK.release()
        embedder.close()


def test_complete_snapshot_avoids_repository_load_and_isolates_revisions(tmp_path, fake_models):
    settings = Settings(data_dir=tmp_path)
    repository = settings.models_dir / "models--BAAI--bge-m3"

    def snapshot(revision):
        path = repository / "snapshots" / revision
        path.mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "pytorch_model.bin", "sparse_linear.pt"):
            (path / name).write_text("{}", encoding="utf-8")
        (repository / "refs").mkdir(exist_ok=True)
        (repository / "refs" / "main").write_text(revision, encoding="ascii")
        return path.resolve()

    first_path = snapshot("a" * 40)
    assert _local_model_source(settings, settings.embedding_model) == str(first_path)
    embedder = BGEEmbedder(settings)
    embedder.encode(["same"])
    embedder.close()
    second_path = snapshot("b" * 40)
    restarted = BGEEmbedder(settings)
    try:
        restarted.encode(["same"])
        assert restarted.last_call_stats["cache_misses"] == 1
        assert fake_models.embedding_loads[-1][0] == str(second_path)
        assert fake_models.embedding_loads[0][0] == str(first_path)
    finally:
        restarted.close()


@pytest.mark.asyncio
async def test_structured_role_uses_auxiliary_chat_and_its_tokenizer(tmp_path):
    settings = Settings(data_dir=tmp_path, context_window_tokens=2048, output_reserve_tokens=128,
                        structured_retries=0).model_copy(update={
                            "structured_model": "test-helper", "structured_tokenizer_model": "test-helper-tokenizer"})
    llm = OllamaLLM(settings)
    llm._tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
    llm._structured_tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text) * 3)
    calls = []

    async def main_invoke(messages):
        calls.append("main")
        return SimpleNamespace(content="main response")

    async def helper_invoke(messages):
        calls.append("helper")
        return TinySchema(value=9)

    llm._chat = SimpleNamespace(ainvoke=main_invoke)
    llm._structured_chat = SimpleNamespace(with_structured_output=lambda *args, **kwargs: SimpleNamespace(ainvoke=helper_invoke))
    assert (await llm.structured("system", "question", TinySchema)).value == 9
    assert await llm.text("system", "question") == "main response"
    assert calls == ["helper", "main"]
    assert llm.count_tokens("abc") == 3
    assert llm.count_structured_tokens("abc") == 9
    with pytest.raises(ProviderUnavailable, match="上下文预算"):
        await llm.structured("s", "x" * 700, TinySchema)
    assert calls == ["helper", "main"]
    assert not _HEAVY_MODEL_LOCK.locked()


def test_auxiliary_model_requires_explicit_tokenizer(tmp_path):
    settings = Settings(data_dir=tmp_path).model_copy(update={"structured_model": "helper", "structured_tokenizer_model": None})
    with pytest.raises(ValueError, match="structured_tokenizer_model"):
        OllamaLLM(settings)


@pytest.mark.asyncio
async def test_schema_whitelist_routes_and_budgets_each_selected_model(tmp_path):
    class SummarySchema(BaseModel):
        value: int

    settings = Settings(data_dir=tmp_path, context_window_tokens=512, output_reserve_tokens=128,
                        structured_retries=0).model_copy(update={
                            "structured_model": "helper", "structured_tokenizer_model": "helper-tokenizer",
                            "structured_model_schemas": " SummarySchema, EvidenceDigest "})
    llm = OllamaLLM(settings)
    llm._tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
    llm._structured_tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text) * 3)
    calls = []

    def selected(route, schema):
        async def invoke(messages):
            calls.append(route)
            return schema(value=7)
        return SimpleNamespace(ainvoke=invoke)

    llm._chat = SimpleNamespace(with_structured_output=lambda schema, **kwargs: selected("main", schema))
    llm._structured_chat = SimpleNamespace(with_structured_output=lambda schema, **kwargs: selected("helper", schema))
    # This fits the main tokenizer but would overflow the auxiliary tokenizer.
    assert (await llm.structured("s", "x" * 200, TinySchema)).value == 7
    assert (await llm.structured("s", "short", SummarySchema)).value == 7
    with pytest.raises(ProviderUnavailable, match="上下文预算"):
        await llm.structured("s", "x" * 200, SummarySchema)
    assert calls == ["main", "helper"]
    assert not _HEAVY_MODEL_LOCK.locked()
