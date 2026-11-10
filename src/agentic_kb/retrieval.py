"""Persistent Qdrant dense+sparse retrieval, real reranking, and parent expansion."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .chunking import build_chunks, stable_id
from .ingestion import parse_document
from .models import DocumentRecord, Evidence
from .store import DocumentStore

logger = logging.getLogger(__name__)


def _has_substantive_text(text: str) -> bool:
    """Exclude heading/separator-only fragments without inventing relevance."""
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s{0,3}#{1,6}(?:\s|$)", line):
            continue
        if re.fullmatch(r"[\s*_=#~-]*", line):
            continue
        lines.append(line)
    return bool(re.search(r"[\w\u4e00-\u9fff]", "\n".join(lines)))


class KnowledgeBase:
    """Single-process KB; use one instance shared by all Gradio sessions.

    Documents have unique, case-insensitive filenames. Re-uploading a filename
    updates its version; identical file contents are deduplicated. Sources are
    copied into the managed data directory, independent of upload temp paths.
    """

    def __init__(self, settings, embedder, reranker, *, token_counter: Callable[[str], int] | None = None):
        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker
        self.token_counter = token_counter
        self._lock = threading.RLock()
        self._last_trace: dict[str, Any] = {}
        self.data_dir = Path(settings.data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = DocumentStore(self.data_dir / "documents.sqlite")
        self.collection = settings.qdrant_collection
        self.client = (
            QdrantClient(url=settings.qdrant_url, timeout=settings.request_timeout_seconds)
            if settings.qdrant_url
            else QdrantClient(path=str(self.data_dir / "qdrant"))
        )
        try:
            self._ensure_collection()
        except Exception:
            self.client.close()
            self.store.close()
            raise

    def _ensure_collection(self) -> None:
        fingerprint = json.dumps({
            "provider": self.settings.provider,
            "embedding_model": self.settings.embedding_model,
            "dimensions": self.embedder.dimensions,
            "adapter": getattr(self.embedder, "fingerprint", type(self.embedder).__qualname__),
        }, sort_keys=True)
        stored = self.store.metadata("embedding_fingerprint:" + self.collection)
        if stored is not None and stored != fingerprint:
            raise ValueError("嵌入模型或 provider 已改变，请使用新的 KB_DATA_DIR 重建索引，避免混用不同向量空间")
        if not self.client.collection_exists(self.collection):
            if self.store.list_documents():
                raise ValueError("文档数据库已有记录但 Qdrant 集合不存在，请检查数据目录或重建索引")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={"dense": qm.VectorParams(size=self.embedder.dimensions, distance=qm.Distance.COSINE)},
                sparse_vectors_config={"sparse": qm.SparseVectorParams()},
            )
        info = self.client.get_collection(self.collection)
        if stored is None and (info.points_count or 0) > 0:
            raise ValueError("Qdrant 集合已有向量，但当前目录没有对应模型元数据；请使用专用新集合，避免覆盖其他知识库")
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or "dense" not in vectors or vectors["dense"].size != self.embedder.dimensions:
            raise ValueError("Qdrant dense 向量维度不匹配，请使用新集合重建索引")
        sparse = info.config.params.sparse_vectors or {}
        if "sparse" not in sparse:
            raise ValueError("Qdrant 缺少 sparse 命名向量，不能执行混合检索")
        self.store.set_metadata("embedding_fingerprint:" + self.collection, fingerprint)

    @property
    def last_trace(self) -> dict[str, Any]:
        """Most recently completed search diagnostics; per-call logs are authoritative."""
        with self._lock:
            return self._last_trace.copy()

    def list_documents(self) -> list[DocumentRecord]:
        return self.store.list_documents()

    def ingest(self, paths: list[Path]) -> list[DocumentRecord]:
        paths = [Path(path).expanduser().resolve() for path in paths]
        names: dict[str, Path] = {}
        for path in paths:
            key = path.name.casefold()
            if key in names and names[key] != path:
                raise ValueError(f"同一批文件存在重复文件名 {path.name}，请重命名后导入")
            names[key] = path
        # One explicitly scoped import session amortizes weight loading across
        # documents; it cannot leak a resident model into subsequent chat calls.
        batch_session = getattr(self.embedder, "batch_session", None)
        with batch_session() if batch_session is not None else nullcontext():
            return [self._ingest_one(path) for path in paths]

    def _ingest_one(self, path: Path) -> DocumentRecord:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        doc_id = stable_id("document:" + path.name.casefold())
        # Serialize ingest so versions and file activation remain coherent.
        with self._lock:
            old = self.store.get_document(doc_id)
            if old is not None:
                if old.sha256 == digest:
                    return old
                # An existing logical document must activate its new content,
                # even when that content equals another document's bytes.
            else:
                for record in self.store.list_documents():
                    if record.sha256 == digest:
                        return record
            # Reserve a generation before any external writes. Failed cleanup
            # must never make orphan points visible in a later retry that reused
            # the same (doc_id, version) pair.
            reserved = int(self.store.metadata("last_reserved_version:" + doc_id) or "0")
            version = max(reserved, old.version if old else 0) + 1
            self.store.set_metadata("last_reserved_version:" + doc_id, str(version))
            managed_path = self.data_dir / "documents" / doc_id / f"v{version}" / path.name
            # Parse, hash and embed the same immutable byte snapshot. Reading the
            # original path again could mix versions when a user edits it during
            # an ingest operation.
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            managed_path.write_bytes(raw)
            sections = parse_document(managed_path)
            parents, children = build_chunks(
                sections, doc_id=doc_id, version=version, source_path=str(managed_path),
                parent_size=self.settings.parent_chunk_size,
                parent_overlap=self.settings.parent_chunk_overlap,
                parent_min_size=self.settings.parent_min_size,
                child_size=self.settings.child_chunk_size,
                child_overlap=self.settings.child_chunk_overlap,
            )
            if not children:
                raise ValueError(f"文档未生成可检索子块：{path.name}")
            embeddings = []
            for offset in range(0, len(children), 32):
                embeddings.extend(self.embedder.encode([child.text for child in children[offset:offset + 32]]))
            if len(embeddings) != len(children):
                raise ValueError("嵌入模型返回的向量数与子块数不一致")
            points = []
            for child, embedding in zip(children, embeddings, strict=True):
                self._validate_embedding(embedding)
                payload = child.model_dump()
                payload.update(document_name=path.name, source_path=str(managed_path))
                points.append(qm.PointStruct(
                    id=child.chunk_id,
                    vector={
                        "dense": embedding.dense,
                        "sparse": qm.SparseVector(indices=embedding.sparse_indices, values=embedding.sparse_values),
                    }, payload=payload,
                ))
            record = DocumentRecord(
                doc_id=doc_id, name=path.name, path=str(managed_path), sha256=digest,
                version=version, status="active", created_at=datetime.now(UTC).isoformat(),
                parent_count=len(parents), child_count=len(children),
            )
            try:
                for offset in range(0, len(points), 128):
                    self.client.upsert(self.collection, points[offset:offset + 128], wait=True)
                # SQLite activation is atomic; searches only see the active version.
                self.store.replace_document(record, parents)
            except Exception:
                try:
                    self.client.delete(self.collection, qm.PointIdsList(points=[point.id for point in points]), wait=True)
                except Exception:
                    logger.exception("索引回滚清理失败；未激活向量仍被版本过滤屏蔽")
                raise
            if old:
                try:
                    self.client.delete(
                        self.collection,
                        qm.FilterSelector(filter=qm.Filter(must=[
                            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
                            qm.FieldCondition(key="version", match=qm.MatchValue(value=old.version)),
                        ])), wait=True,
                    )
                except Exception:
                    logger.exception("旧向量清理失败；旧版本不会参与检索")
            logger.info("document_indexed doc_id=%s version=%d parents=%d children=%d", doc_id, version, len(parents), len(children))
            return record

    def _validate_embedding(self, embedding) -> None:
        if len(embedding.dense) != self.embedder.dimensions or not all(math.isfinite(v) for v in embedding.dense):
            raise ValueError("Dense 嵌入维度或数值不合法")
        if len(embedding.sparse_indices) != len(embedding.sparse_values):
            raise ValueError("Sparse indices 与 values 长度不一致")
        if len(set(embedding.sparse_indices)) != len(embedding.sparse_indices):
            raise ValueError("Sparse 索引包含重复维度")
        if not all(index >= 0 for index in embedding.sparse_indices) or not all(math.isfinite(v) for v in embedding.sparse_values):
            raise ValueError("Sparse 嵌入维度或数值不合法")

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            if self.store.get_document(doc_id) is None:
                return False
            # Make the source-of-truth removal visible first. A lost response
            # from Qdrant must never leave an active document with missing
            # vectors; any orphan points are excluded by _query_filter.
            removed = self.store.delete_document(doc_id)
            try:
                self.client.delete(
                    self.collection,
                    qm.FilterSelector(filter=qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])),
                    wait=True,
                )
            except Exception:
                logger.exception("document_logically_deleted_vector_cleanup_failed doc_id=%s; orphan points remain invisible", doc_id)
            # Source snapshots remain for historical answer citations; list/search
            # visibility is removed atomically from the source-of-truth database.
            return removed

    def _query_filter(self, filters: dict | None) -> qm.Filter | None:
        documents = [record for record in self.store.list_documents() if record.status == "active"]
        filters = filters or {}
        allowed = {"doc_id", "doc_ids", "document_name", "version"}
        unsupported = set(filters) - allowed
        if unsupported:
            raise ValueError(f"不支持的检索过滤项：{', '.join(sorted(unsupported))}")
        for key, value in filters.items():
            if value is None or value == "" or value == []:
                continue
            if key in {"doc_id", "doc_ids"}:
                identifiers = set(value if isinstance(value, (list, tuple, set)) else [value])
                documents = [record for record in documents if record.doc_id in identifiers]
            elif key == "document_name":
                documents = [record for record in documents if record.name == value]
            elif key == "version":
                documents = [record for record in documents if record.version == int(value)]
        if not documents:
            return None
        return qm.Filter(should=[qm.Filter(must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=record.doc_id)),
            qm.FieldCondition(key="version", match=qm.MatchValue(value=record.version)),
        ]) for record in documents])

    def search(self, query: str, *, filters: dict | None = None, mode: str = "hybrid", top_k: int | None = None) -> list[Evidence]:
        if mode not in {"dense", "hybrid", "rerank", "full"}:
            raise ValueError(f"不支持的检索模式：{mode}")
        if not query.strip():
            return []
        top_k = self.settings.rerank_top_k if top_k is None else top_k
        if top_k <= 0:
            return []
        started = time.perf_counter()
        with self._lock:
            query_filter = self._query_filter(filters)
        if query_filter is None:
            return []
        embeddings = self.embedder.encode([query])
        if len(embeddings) != 1:
            raise ValueError("查询嵌入必须恰好返回一个向量")
        embedding = embeddings[0]
        embedding_stats = dict(getattr(self.embedder, "last_call_stats", {}))
        self._validate_embedding(embedding)
        candidate_limit = max(self.settings.dense_top_k, self.settings.sparse_top_k, top_k)
        with self._lock:
            # Rebuild after embedding in case an ingest/delete happened meanwhile.
            query_filter = self._query_filter(filters)
            if query_filter is None:
                return []
            fetch_limit = candidate_limit
            filtered_title_ids: set[str] = set()
            fetch_rounds = 0
            while True:
                fetch_rounds += 1
                if mode == "dense" or not embedding.sparse_indices:
                    raw_hits = self.client.query_points(
                        self.collection, query=embedding.dense, using="dense", query_filter=query_filter,
                        limit=fetch_limit, with_payload=True,
                    ).points
                    used_sparse = False
                else:
                    raw_hits = self.client.query_points(
                        self.collection,
                        prefetch=[
                            qm.Prefetch(query=embedding.dense, using="dense", filter=query_filter, limit=max(self.settings.dense_top_k, fetch_limit)),
                            qm.Prefetch(query=qm.SparseVector(indices=embedding.sparse_indices, values=embedding.sparse_values),
                                        using="sparse", filter=query_filter, limit=max(self.settings.sparse_top_k, fetch_limit)),
                        ],
                        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                        query_filter=query_filter, limit=fetch_limit, with_payload=True,
                    ).points
                    used_sparse = True
                hits = []
                for hit in raw_hits:
                    if _has_substantive_text(hit.payload["text"]):
                        hits.append(hit)
                    else:
                        filtered_title_ids.add(str(hit.id))
                if len(hits) >= candidate_limit or len(raw_hits) < fetch_limit or fetch_limit >= candidate_limit * 4:
                    hits = hits[:candidate_limit]
                    break
                # Backfill filtered headings instead of letting them consume k.
                fetch_limit = min(candidate_limit * 4, fetch_limit * 2)
            # Capture parent snapshots while the active document version is locked.
            parent_snapshots = {hit.payload["parent_id"]: self.store.get_parent(hit.payload["parent_id"]) for hit in hits}
        trace: dict[str, Any] = {
            "query": query, "mode": mode, "used_sparse": used_sparse,
            "candidates": [{"child_id": str(hit.id), "score": hit.score, "doc_id": hit.payload["doc_id"]} for hit in hits],
            "reranker_fallback": False,
            "embedding": embedding_stats,
            "title_only_candidates_filtered": len(filtered_title_ids),
            "candidate_fetch_rounds": fetch_rounds,
            "candidate_fetch_limit": fetch_limit,
        }
        scores = [float(hit.score) for hit in hits]
        if mode in {"rerank", "full"} and hits:
            try:
                # Identical text may occur under several sources/versions. Score
                # it once, while preserving every hit and its provenance.
                passages = list(dict.fromkeys(hit.payload["text"] for hit in hits))
                unique_scores = list(self.reranker.score(query, passages))
                if len(unique_scores) != len(passages) or not all(math.isfinite(value) for value in unique_scores):
                    raise ValueError("重排模型返回的分数数量或数值不合法")
                score_by_text = dict(zip(passages, unique_scores, strict=True))
                scores = [score_by_text[hit.payload["text"]] for hit in hits]
                trace["reranking"] = {**dict(getattr(self.reranker, "last_call_stats", {})),
                    "candidate_count": len(hits), "unique_passages": len(passages),
                    "duplicate_candidates": len(hits) - len(passages)}
            except Exception as exc:  # noqa: BLE001 -- optional model boundary has a logged retrieval fallback
                trace["reranker_fallback"] = True
                trace["reranker_error"] = f"{type(exc).__name__}: {exc}"
                trace["reranking"] = {**dict(getattr(self.reranker, "last_call_stats", {})),
                    "candidate_count": len(hits), "unique_passages": len(passages),
                    "duplicate_candidates": len(hits) - len(passages)}
                scores = [float(hit.score) for hit in hits]
                logger.warning("reranker_failed_using_rrf error=%s", exc)
        ranked = sorted(zip(hits, scores, strict=True), key=lambda item: item[1], reverse=True)
        trace["ranked"] = [{"child_id": str(hit.id), "score": score} for hit, score in ranked]
        evidence: list[Evidence] = []
        if mode == "full":
            grouped: OrderedDict[str, list] = OrderedDict()
            for hit, score in ranked:
                grouped.setdefault(hit.payload["parent_id"], []).append((hit, score))
            counter = self.token_counter or len
            trace["context_budget"] = {"limit": self.settings.evidence_token_budget,
                "unit": "tokens" if self.token_counter is not None else "characters_fallback_not_tokens",
                "scope": "evidence_text_only; graph additionally counts metadata and prompts"}
            remaining = self.settings.evidence_token_budget
            for parent_id, children in grouped.items():
                if len(evidence) >= top_k or remaining <= 0:
                    break
                parent = parent_snapshots.get(parent_id)
                if parent is None:
                    continue
                hit, score = children[0]
                text = parent.text
                if counter(text) > remaining:
                    needle = hit.payload["text"]
                    position = max(0, text.find(needle))
                    # Keep the matched span near the start, then find a boundary
                    # with the actual generation-model tokenizer when provided.
                    start = max(0, position - len(needle) // 4)
                    candidate = text[start:]
                    lo, hi = 0, len(candidate)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if counter(candidate[:mid]) <= remaining:
                            lo = mid
                        else:
                            hi = mid - 1
                    text = candidate[:lo]
                if not text:
                    continue
                if not _has_substantive_text(text):
                    continue
                evidence.append(self._evidence(hit, score, text, parent.text, [str(item.id) for item, _ in children], parent_id))
                remaining -= counter(text)
            trace["context_budget"]["used"] = self.settings.evidence_token_budget - remaining
        else:
            for hit, score in ranked[:top_k]:
                evidence.append(self._evidence(hit, score, hit.payload["text"], hit.payload["text"], [str(hit.id)], str(hit.id)))
        trace["evidence_ids"] = [item.evidence_id for item in evidence]
        trace["duration_seconds"] = round(time.perf_counter() - started, 4)
        with self._lock:
            self._last_trace = trace
        logger.info("retrieval_trace %s", json.dumps(trace, ensure_ascii=False))
        return evidence

    @staticmethod
    def _evidence(hit, score: float, text: str, raw_text: str, child_ids: list[str], evidence_id: str) -> Evidence:
        payload = hit.payload
        return Evidence(
            evidence_id=evidence_id, parent_id=payload["parent_id"], doc_id=payload["doc_id"],
            document_name=payload["document_name"], text=text, raw_text=raw_text,
            child_ids=child_ids, heading_path=payload.get("heading_path", []),
            page_start=payload.get("page_start"), page_end=payload.get("page_end"),
            version=payload["version"], score=score, source_path=payload["source_path"],
        )

    def close(self) -> None:
        # Bulk ingestion owns the embedding lock before acquiring the KB lock.
        # Drain providers outside this lock to keep the same acquisition order.
        for provider in (self.embedder, self.reranker):
            close = getattr(provider, "close", None)
            if close is not None:
                close()
        with self._lock:
            try:
                self.client.close()
            finally:
                self.store.close()
