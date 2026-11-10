"""Real local model adapters and explicitly labelled offline test doubles."""
from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
import zlib
from collections import Counter, OrderedDict
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, contextmanager, nullcontext
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .config import Settings
from .models import VectorEmbedding

T = TypeVar("T", bound=BaseModel)
_HEAVY_MODEL_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


@contextmanager
def _sync_model_slot(enabled: bool, timeout: float):
    acquired = False
    try:
        if enabled:
            acquired = _HEAVY_MODEL_LOCK.acquire(timeout=timeout)
            if not acquired:
                raise ProviderUnavailable("等待本地模型内存槽超时，请稍后重试。")
        yield
    finally:
        if acquired:
            _HEAVY_MODEL_LOCK.release()


@asynccontextmanager
async def _async_model_slot(enabled: bool):
    if enabled:
        while not _HEAVY_MODEL_LOCK.acquire(blocking=False):
            await asyncio.sleep(0.05)
    try:
        yield
    finally:
        if enabled:
            _HEAVY_MODEL_LOCK.release()


class ProviderUnavailable(RuntimeError):
    pass


class LLMClient(Protocol):
    token_count_estimated: bool

    async def structured(self, system: str, prompt: str, schema: type[T]) -> T: ...
    async def text(self, system: str, prompt: str) -> str: ...
    def astream(self, system: str, prompt: str) -> AsyncIterator[str]: ...
    def count_tokens(self, text: str) -> int: ...


def _model_cache(settings: Settings) -> str:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return str(settings.models_dir)


def _local_model_source(settings: Settings, model_name: str, *, tokenizer_only: bool = False) -> str:
    """Pin an already complete snapshot without network HEAD or cache refresh."""
    direct = Path(model_name).expanduser()
    if direct.is_dir():
        return str(direct.resolve())
    repository = settings.models_dir / ("models--" + model_name.replace("/", "--"))
    reference = repository / "refs" / "main"
    if not reference.is_file():
        return model_name
    revision = reference.read_text(encoding="utf-8").strip()
    # Ref content is data, never a path supplied to a shell or an arbitrary file.
    if not re.fullmatch(r"[a-fA-F0-9]{7,64}", revision):
        return model_name
    snapshot = repository / "snapshots" / revision
    tokenizer_ready = (snapshot / "tokenizer_config.json").is_file() and (
        (snapshot / "tokenizer.json").is_file() or (snapshot / "sentencepiece.bpe.model").is_file()
    )
    if tokenizer_only:
        return str(snapshot.resolve()) if tokenizer_ready else model_name
    weights = any((snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not weights and index_path.is_file():
            try:
                shards = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
                weights = bool(shards) and all((snapshot / name).is_file() for name in shards)
            except (KeyError, TypeError, ValueError):
                weights = False
    sparse_ready = "bge-m3" not in model_name.lower() or (snapshot / "sparse_linear.pt").is_file()
    if tokenizer_ready and weights and (snapshot / "config.json").is_file() and sparse_ready:
        return str(snapshot.resolve())
    return model_name


def _source_identity(source: str) -> str:
    path = Path(source)
    if not path.is_dir():
        return "repository:" + source
    if path.parent.name == "snapshots":
        return "snapshot:" + path.name
    # User-supplied local weights can change without changing the directory name.
    # File metadata avoids rereading gigabytes on every provider construction.
    files = [(item.name, item.stat().st_size, item.stat().st_mtime_ns)
             for item in sorted(path.iterdir()) if item.is_file()]
    return hashlib.sha256(json.dumps([str(path.resolve()), files]).encode()).hexdigest()


def _inference_key(fingerprint: str, *parts: str) -> str:
    payload = json.dumps([fingerprint, *parts], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _InferenceCache:
    """Bounded, model-keyed real outputs. No prompts or answers are cached.

    Cache keys are hashes, payloads are validated JSON, and failures degrade to
    actual inference. Counts are global per kind, not unbounded per model name.
    """
    MAX_PAYLOAD_BYTES = 128 * 1024

    def __init__(self, path: Path, kind: str, *, capacity: int, memory_capacity: int = 128):
        self.path, self.kind = path, kind
        self.capacity = max(0, capacity)
        self.memory_capacity = min(self.capacity, max(0, memory_capacity))
        self._memory: OrderedDict[str, bytes] = OrderedDict()
        self._connection = None
        self._lock = threading.RLock()

    def _database(self):
        if self._connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(self.path), timeout=10, check_same_thread=False)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            # Main database is capped at about 128MiB; also bound WAL growth.
            self._connection.execute("PRAGMA max_page_count=32768")
            self._connection.execute("PRAGMA journal_size_limit=8388608")
            self._connection.executescript("""
                CREATE TABLE IF NOT EXISTS inference_cache (
                    kind TEXT NOT NULL, cache_key TEXT NOT NULL,
                    payload BLOB NOT NULL, accessed INTEGER NOT NULL,
                    PRIMARY KEY(kind, cache_key)
                );
                CREATE INDEX IF NOT EXISTS inference_cache_lru ON inference_cache(kind, accessed);
            """)
            self._connection.commit()
        return self._connection

    def _remember(self, key: str, payload: bytes):
        if self.memory_capacity:
            self._memory[key] = payload
            self._memory.move_to_end(key)
            while len(self._memory) > self.memory_capacity:
                self._memory.popitem(last=False)

    def get(self, key: str) -> tuple[bytes | None, str | None]:
        if not self.capacity:
            return None, None
        with self._lock:
            if key in self._memory:
                self._memory.move_to_end(key)
                return self._memory[key], "memory"
            try:
                connection = self._database()
                row = connection.execute("SELECT payload FROM inference_cache WHERE kind=? AND cache_key=?", (self.kind, key)).fetchone()
                if row:
                    payload = bytes(row[0])
                    if len(payload) > self.MAX_PAYLOAD_BYTES:
                        return None, None
                    connection.execute("UPDATE inference_cache SET accessed=? WHERE kind=? AND cache_key=?", (time.time_ns(), self.kind, key))
                    connection.commit()
                    self._remember(key, payload)
                    return payload, "disk"
            except sqlite3.Error:
                logger.warning("inference_cache_read_failed kind=%s; running actual inference", self.kind, exc_info=True)
        return None, None

    def put(self, key: str, payload: bytes):
        if not self.capacity or len(payload) > self.MAX_PAYLOAD_BYTES:
            return
        with self._lock:
            self._remember(key, payload)
            try:
                connection = self._database()
                with connection:
                    connection.execute("INSERT OR REPLACE INTO inference_cache VALUES (?,?,?,?)",
                                       (self.kind, key, payload, time.time_ns()))
                    excess = connection.execute("SELECT COUNT(*) FROM inference_cache WHERE kind=?", (self.kind,)).fetchone()[0] - self.capacity
                    if excess > 0:
                        connection.execute("DELETE FROM inference_cache WHERE kind=? AND cache_key IN (SELECT cache_key FROM inference_cache WHERE kind=? ORDER BY accessed LIMIT ?)",
                                           (self.kind, self.kind, excess))
            except sqlite3.Error:
                logger.warning("inference_cache_write_failed kind=%s; output remains real inference", self.kind, exc_info=True)

    def discard(self, key: str):
        with self._lock:
            self._memory.pop(key, None)
            try:
                with self._database() as connection:
                    connection.execute("DELETE FROM inference_cache WHERE kind=? AND cache_key=?", (self.kind, key))
            except sqlite3.Error:
                logger.warning("inference_cache_invalid_entry_cleanup_failed", exc_info=True)

    def close(self):
        with self._lock:
            self._memory.clear()
            if self._connection is not None:
                self._connection.close()
                self._connection = None


def _call_stats(requested: int, unique: int, source: str) -> dict:
    return {"requested": requested, "unique_inputs": unique, "batch_duplicates": requested - unique,
            "cache_hits": 0, "cache_misses": 0, "memory_hits": 0, "disk_hits": 0,
            "computed": 0, "model_loads": 0, "model_source": source}


def _decompress_cache(payload: bytes) -> bytes:
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(payload, 2 * 1024 * 1024)
    if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError("缓存数据不完整或超出解压大小限制")
    return decoded


class OllamaLLM:
    token_count_estimated = False

    def __init__(self, settings: Settings):
        from langchain_ollama import ChatOllama

        self.settings = settings
        chat_options = dict(base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            num_ctx=settings.context_window_tokens,
            num_predict=settings.output_reserve_tokens,
            keep_alive=0 if settings.low_memory_mode else "5m",
            # Local Ollama traffic must not inherit system HTTP(S) proxies.
            client_kwargs={"timeout": settings.request_timeout_seconds, "trust_env": False})
        self._chat = ChatOllama(model=settings.chat_model, **chat_options)
        self._structured_model_name = getattr(settings, "structured_model", None)
        self._structured_tokenizer_name = getattr(settings, "structured_tokenizer_model", None)
        if self._structured_model_name and not self._structured_tokenizer_name:
            raise ValueError("启用结构化辅助模型时必须显式配置 structured_tokenizer_model")
        self._structured_chat = ChatOllama(model=self._structured_model_name, **chat_options) if self._structured_model_name else None
        self._tokenizer = None
        self._structured_tokenizer = None
        self._tokenizer_lock = threading.RLock()
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        with self._tokenizer_lock:
            if self._tokenizer is None:
                try:
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(
                        _local_model_source(self.settings, self.settings.tokenizer_model, tokenizer_only=True),
                        cache_dir=_model_cache(self.settings),
                        trust_remote_code=False,
                    )
                except Exception as exc:
                    raise ProviderUnavailable(
                        "无法加载生成模型的分词器。请安装 models 依赖并下载配置的 tokenizer；"
                        "项目不会把字符估算伪装为真实 token 统计。"
                    ) from exc
            return len(self._tokenizer.encode(text, add_special_tokens=False))

    def count_structured_tokens(self, text: str) -> int:
        if not self._structured_model_name:
            return self.count_tokens(text)
        with self._tokenizer_lock:
            if self._structured_tokenizer is None:
                try:
                    from transformers import AutoTokenizer
                    self._structured_tokenizer = AutoTokenizer.from_pretrained(
                        _local_model_source(self.settings, self._structured_tokenizer_name, tokenizer_only=True),
                        cache_dir=_model_cache(self.settings), trust_remote_code=False)
                except Exception as exc:
                    raise ProviderUnavailable("无法加载结构化辅助模型的真实分词器，请先准备 configured tokenizer。") from exc
            return len(self._structured_tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _messages(system: str, prompt: str):
        from langchain_core.messages import HumanMessage, SystemMessage

        return [SystemMessage(content=system), HumanMessage(content=prompt)]

    def _validate_prompt(self, system: str, prompt: str, *, structured: bool = False) -> None:
        counter = self.count_structured_tokens if structured else self.count_tokens
        if counter(system) + counter(prompt) + 64 > (
            self.settings.context_window_tokens - self.settings.output_reserve_tokens
        ):
            raise ProviderUnavailable("输入超过模型上下文预算，请缩短问题或调整上下文配置。")

    def _use_structured_helper(self, schema: type[BaseModel]) -> bool:
        if self._structured_chat is None:
            return False
        configured = str(getattr(self.settings, "structured_model_schemas", "") or "").strip()
        if not configured:
            return True  # Preserve the original all-structured routing when unset.
        allowed = {name.strip() for name in configured.split(",") if name.strip()}
        return schema.__name__ in allowed

    async def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        last_error = None
        base_prompt = prompt
        use_helper = self._use_structured_helper(schema)
        for attempt in range(self.settings.structured_retries + 1):
            try:
                prompt = base_prompt + ("\n请严格遵守JSON schema，不要添加解释。" if attempt else "")
                self._validate_prompt(system, prompt, structured=use_helper)
                self.calls += 1
                selected_chat = self._structured_chat if use_helper else self._chat
                model = selected_chat.with_structured_output(schema, method="json_schema")
                async with asyncio.timeout(self.settings.request_timeout_seconds):
                    async with _async_model_slot(self.settings.low_memory_mode):
                        result = await model.ainvoke(self._messages(system, prompt))
                return result if isinstance(result, schema) else schema.model_validate(result)
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                last_error = exc
        raise ProviderUnavailable(f"结构化模型调用失败（{schema.__name__}）：{last_error}") from last_error

    async def text(self, system: str, prompt: str) -> str:
        try:
            self._validate_prompt(system, prompt)
            self.calls += 1
            async with asyncio.timeout(self.settings.request_timeout_seconds):
                async with _async_model_slot(self.settings.low_memory_mode):
                    result = await self._chat.ainvoke(self._messages(system, prompt))
            return str(result.content)
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            raise ProviderUnavailable(f"Ollama 调用失败，请检查服务和模型：{exc}") from exc

    async def astream(self, system: str, prompt: str) -> AsyncIterator[str]:
        try:
            self._validate_prompt(system, prompt)
            self.calls += 1
            async with asyncio.timeout(self.settings.request_timeout_seconds):
                async with _async_model_slot(self.settings.low_memory_mode):
                    async for chunk in self._chat.astream(self._messages(system, prompt)):
                        if isinstance(chunk.content, str) and chunk.content:
                            yield chunk.content
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            raise ProviderUnavailable(f"Ollama 流式调用失败：{exc}") from exc


class BGEEmbedder:
    """One BGE-M3 pass produces both dense and learned sparse vectors."""
    dimensions = 1024

    def __init__(self, settings: Settings, *, cache_capacity: int = 2048, memory_cache_size: int = 128):
        self.settings = settings
        self.model_name = settings.embedding_model
        # Preserve the existing Qdrant identity. The stronger cache identity below
        # additionally separates revisions, precision and inference protocol.
        self.fingerprint = f"bge-m3:{self.model_name}:dense+sparse"
        self._model = None
        self._lock = threading.RLock()
        self._trace = threading.local()
        self._session_depth = 0
        self._source = _local_model_source(settings, self.model_name)
        self._cache_fingerprint = json.dumps({"adapter": self.fingerprint, "source": _source_identity(self._source),
            "dimensions": self.dimensions, "device": settings.device, "fp16": settings.device != "cpu",
            "max_length": 2048, "normalize_embeddings": True,
            "dense": True, "sparse": True, "colbert": False}, sort_keys=True)
        self._cache = _InferenceCache(Path(settings.data_dir) / "cache" / "inference.sqlite", "embedding",
                                      capacity=cache_capacity, memory_capacity=memory_cache_size)

    @property
    def last_call_stats(self) -> dict:
        """Thread-local call diagnostics, so concurrent queries do not share counts."""
        return dict(getattr(self._trace, "stats", {}))

    def _unload(self):
        self._model = None
        gc.collect()

    @contextmanager
    def batch_session(self):
        """Keep one model only during a synchronous bulk import, then unload.

        A session owns the provider lock and (in low-memory mode) the heavy slot.
        It must never span an await or another model call. Nested sessions from
        this same thread are safe; other threads cannot enter encode meanwhile.
        """
        with self._lock:
            outer = self._session_depth == 0
            slot = _sync_model_slot(self.settings.low_memory_mode, self.settings.request_timeout_seconds) if outer else nullcontext()
            with slot:
                self._session_depth += 1
                try:
                    yield self
                finally:
                    self._session_depth -= 1
                    if outer and self.settings.low_memory_mode:
                        self._unload()

    def _load(self):
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                self._source, use_fp16=self.settings.device != "cpu",
                devices=self.settings.device, cache_dir=_model_cache(self.settings), normalize_embeddings=True,
            )
        except Exception as exc:
            self._unload()
            raise ProviderUnavailable("BGE-M3 未就绪。请安装 models 依赖，并运行 agentic-kb prepare-models。") from exc

    def _validate(self, vector: VectorEmbedding):
        if (len(vector.dense) != self.dimensions or not all(math.isfinite(value) for value in vector.dense)
                or len(vector.sparse_indices) != len(vector.sparse_values)
                or len(set(vector.sparse_indices)) != len(vector.sparse_indices)
                or not all(index >= 0 for index in vector.sparse_indices)
                or not all(math.isfinite(value) for value in vector.sparse_values)):
            raise ValueError("BGE 返回的 Dense/Sparse 向量不合法")

    def encode(self, texts: Sequence[str]) -> list[VectorEmbedding]:
        texts = list(texts)
        keys = [_inference_key(self._cache_fingerprint, text) for text in texts]
        unique = dict(zip(keys, texts, strict=True))
        stats = _call_stats(len(texts), len(unique), self._source)
        started = time.perf_counter()
        with self._lock:
            resolved: dict[str, VectorEmbedding] = {}
            missing: dict[str, str] = {}
            try:
                for key, text in unique.items():
                    payload, location = self._cache.get(key)
                    if payload is not None:
                        try:
                            vector = VectorEmbedding.model_validate_json(_decompress_cache(payload))
                            self._validate(vector)
                        except (ValueError, zlib.error):
                            self._cache.discard(key)
                        else:
                            resolved[key] = vector
                            stats["cache_hits"] += 1
                            stats[location + "_hits"] += 1
                            continue
                    missing[key] = text
                stats["cache_misses"] = len(missing)
                if missing:
                    slot = nullcontext() if self._session_depth else _sync_model_slot(
                        self.settings.low_memory_mode, self.settings.request_timeout_seconds)
                    with slot:
                        try:
                            if self._model is None:
                                self._load()
                                stats["model_loads"] += 1
                            output = self._model.encode(
                                list(missing.values()), batch_size=4, max_length=2048,
                                return_dense=True, return_sparse=True, return_colbert_vecs=False,
                            )
                            vectors = []
                            for dense, weights in zip(output["dense_vecs"], output["lexical_weights"], strict=True):
                                sparse = sorted((int(index), float(value)) for index, value in weights.items())
                                vector = VectorEmbedding(dense=[float(value) for value in dense],
                                    sparse_indices=[item[0] for item in sparse], sparse_values=[item[1] for item in sparse])
                                self._validate(vector)
                                vectors.append(vector)
                            if len(vectors) != len(missing):
                                raise ValueError("BGE 向量数量与输入数量不一致")
                            for key, vector in zip(missing, vectors, strict=True):
                                resolved[key] = vector
                                self._cache.put(key, zlib.compress(vector.model_dump_json().encode("utf-8")))
                            stats["computed"] = len(vectors)
                        except BaseException:
                            self._unload()
                            raise
                        finally:
                            if self.settings.low_memory_mode and not self._session_depth:
                                self._unload()
                # Never expose the mutable object used by another caller.
                return [resolved[key].model_copy(deep=True) for key in keys]
            finally:
                stats["duration_seconds"] = round(time.perf_counter() - started, 6)
                self._trace.stats = stats
                logger.info("embedding_inference %s", json.dumps(stats, ensure_ascii=False))

    def close(self):
        with self._lock:
            self._unload()
            self._cache.close()


class CrossEncoderReranker:
    def __init__(self, settings: Settings, *, cache_capacity: int = 4096, memory_cache_size: int = 128):
        self.settings = settings
        self.model_name = settings.reranker_model
        self._model = None
        self._lock = threading.RLock()
        self._trace = threading.local()
        self._source = _local_model_source(settings, self.model_name)
        self._cache_fingerprint = json.dumps({"adapter": "cross-encoder-pair-v1", "model": self.model_name,
            "source": _source_identity(self._source), "device": settings.device, "fp16": settings.device != "cpu",
            "normalize": True, "max_length": 512}, sort_keys=True)
        self._cache = _InferenceCache(Path(settings.data_dir) / "cache" / "inference.sqlite", "rerank",
                                      capacity=cache_capacity, memory_capacity=memory_cache_size)

    @property
    def last_call_stats(self) -> dict:
        return dict(getattr(self._trace, "stats", {}))

    def _unload(self):
        self._model = None
        gc.collect()

    def _load(self):
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(self._source, use_fp16=self.settings.device != "cpu",
                devices=self.settings.device, cache_dir=_model_cache(self.settings))
        except Exception as exc:
            self._unload()
            raise ProviderUnavailable("Cross-Encoder 未就绪，请准备 models 依赖和重排模型。") from exc

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        passages = list(passages)
        keys = [_inference_key(self._cache_fingerprint, query, text) for text in passages]
        unique = dict(zip(keys, passages, strict=True))
        stats = _call_stats(len(passages), len(unique), self._source)
        started = time.perf_counter()
        with self._lock:
            resolved: dict[str, float] = {}
            missing: dict[str, str] = {}
            try:
                for key, text in unique.items():
                    payload, location = self._cache.get(key)
                    if payload is not None:
                        try:
                            value = float(_decompress_cache(payload))
                            if not math.isfinite(value):
                                raise ValueError("Non-finite cached score")
                        except (ValueError, zlib.error):
                            self._cache.discard(key)
                        else:
                            resolved[key] = value
                            stats["cache_hits"] += 1
                            stats[location + "_hits"] += 1
                            continue
                    missing[key] = text
                stats["cache_misses"] = len(missing)
                if missing:
                    with _sync_model_slot(self.settings.low_memory_mode, self.settings.request_timeout_seconds):
                        try:
                            if self._model is None:
                                self._load()
                                stats["model_loads"] += 1
                            scores = self._model.compute_score(
                                [[query, text] for text in missing.values()], normalize=True, max_length=512,
                                batch_size=2 if self.settings.low_memory_mode else 16,
                            )
                            if isinstance(scores, (float, int)):
                                scores = [scores]
                            scores = [float(value) for value in scores]
                            if len(scores) != len(missing) or not all(math.isfinite(value) for value in scores):
                                raise ValueError("Cross-Encoder 分数数量或数值不合法")
                            for key, value in zip(missing, scores, strict=True):
                                resolved[key] = value
                                self._cache.put(key, zlib.compress(repr(value).encode("ascii")))
                            stats["computed"] = len(scores)
                        except BaseException:
                            self._unload()
                            raise
                        finally:
                            if self.settings.low_memory_mode:
                                self._unload()
                return [resolved[key] for key in keys]
            finally:
                stats["duration_seconds"] = round(time.perf_counter() - started, 6)
                self._trace.stats = stats
                logger.info("reranker_inference %s", json.dumps(stats, ensure_ascii=False))

    def close(self):
        with self._lock:
            self._unload()
            self._cache.close()


def _terms(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_./:-]+|[\u4e00-\u9fff]", lowered)
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    return words + [chinese[i:i + 2] for i in range(max(0, len(chinese) - 1))]


class HashingTestEmbedder:
    """Deterministic plumbing test only. This is NOT a semantic embedding model."""
    dimensions = 256
    fingerprint = "OFFLINE-TEST:hashing-v1:not-for-quality-evaluation"
    model_name = "offline-test-hashing"

    def encode(self, texts: Sequence[str]) -> list[VectorEmbedding]:
        result = []
        for text in texts:
            counts = Counter(_terms(text))
            sparse: Counter[int] = Counter()
            dense = [0.0] * self.dimensions
            for term, value in counts.items():
                digest = hashlib.sha256(term.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big")
                sparse[index % 2_000_003] += math.log1p(value)
                dense[index % self.dimensions] += math.log1p(value)
            norm = math.sqrt(sum(x * x for x in dense)) or 1.0
            items = sorted(sparse.items())
            result.append(VectorEmbedding(
                dense=[x / norm for x in dense],
                sparse_indices=[x[0] for x in items], sparse_values=[x[1] for x in items],
            ))
        return result


class LexicalTestReranker:
    """Test double, deliberately named differently from a Cross-Encoder."""
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        query_terms = set(_terms(query))
        return [len(query_terms.intersection(_terms(p))) / max(1, len(query_terms)) for p in passages]


class OfflineTestLLM:
    """Rule-based workflow smoke test. Never used to report RAGAS quality."""
    token_count_estimated = True
    calls = 0

    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"[\u4e00-\u9fff]|\w+|[^\s\w]", text))

    async def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        self.calls += 1
        fields = schema.model_fields
        data = {}
        if "needs_clarification" in fields:
            ambiguous = any(q in prompt for q in ["这个版本怎么升级", "它怎么部署"])
            data.update(needs_clarification=ambiguous, question="请补充产品名和当前、目标版本。" if ambiguous else "",
                        clarification_question="请补充产品名和当前、目标版本。" if ambiguous else "",
                        options=["AtlasDesk v1 升级至 v2"], missing_fields=["版本"] if ambiguous else [])
        if "needs_decomposition" in fields:
            data["needs_decomposition"] = False
        if "tasks" in fields:
            # The test planner intentionally has one bounded task; real planning is exercised with Ollama.
            data["tasks"] = [{"task_id": "t1", "query": prompt[-1800:], "depends_on": [], "constraints": {}, "filters": {}}]
        for name in ("query", "rewritten_query", "rewritten_question"):
            if name in fields:
                data[name] = prompt[-1800:]
        for name in ("sufficient", "is_sufficient"):
            if name in fields:
                data[name] = True
        for name in ("missing", "missing_information"):
            if name in fields:
                data[name] = []
        if "supporting_quotes" in fields:
            originals = re.findall(r"<text>\n(.*?)\n</text>", prompt, flags=re.DOTALL)
            data["supporting_quotes"] = [originals[0][:80]] if originals else []
        if "answer" in fields:
            data["answer"] = "【离线流程测试，非模型回答】请查看引用原文。"
        if "extracts" in fields:
            data["extracts"] = []
        for name in ("goals", "constraints", "entities", "facts", "conclusions", "open_questions", "unresolved", "confirmed_facts"):
            if name in fields and name not in data:
                annotation = str(fields[name].annotation)
                data[name] = {} if "dict" in annotation else []
        for name in ("summary", "reason", "reasoning", "user_goal", "goal"):
            if name in fields:
                data[name] = "离线流程测试摘要。"
        try:
            return schema.model_validate(data)
        except Exception as exc:
            raise ProviderUnavailable(f"离线脚本未覆盖 schema {schema.__name__}，不能用于真实问答：{exc}") from exc

    async def text(self, system: str, prompt: str) -> str:
        return "【离线流程测试，非模型回答】\n" + prompt[-1200:]

    async def astream(self, system: str, prompt: str) -> AsyncIterator[str]:
        response = await self.text(system, prompt)
        for offset in range(0, len(response), 40):
            await asyncio.sleep(0)
            yield response[offset:offset + 40]


def build_providers(settings: Settings):
    if settings.provider == "offline-test":
        return OfflineTestLLM(), HashingTestEmbedder(), LexicalTestReranker()
    return OllamaLLM(settings), BGEEmbedder(settings), CrossEncoderReranker(settings)
