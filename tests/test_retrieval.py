"""Contract tests use explicit deterministic fixtures, never model-quality claims."""
from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agentic_kb.chunking import build_chunks
from agentic_kb.config import Settings
from agentic_kb.ingestion import parse_document
from agentic_kb.models import ParsedSection, VectorEmbedding
from agentic_kb.retrieval import KnowledgeBase


class FixtureEmbedder:
    dimensions = 4

    def encode(self, texts):
        result = []
        for text in texts:
            counts = collections.Counter(ord(char) for char in text.lower() if char.isalnum())
            indices = sorted(counts)
            result.append(VectorEmbedding(
                dense=[1.0, 0.0, 0.0, 0.0],
                sparse_indices=indices, sparse_values=[float(counts[index]) for index in indices],
            ))
        return result


class FixtureReranker:
    def score(self, query, passages):
        return [float(sum(char in text for char in set(query))) for text in passages]


def settings(tmp_path):
    return Settings(data_dir=tmp_path / "kb", provider="offline-test", parent_chunk_size=400,
                    parent_chunk_overlap=30, parent_min_size=80, child_chunk_size=100,
                    child_chunk_overlap=20)


def test_heading_parser_keeps_code_hash_inside_section(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# 部署\n\n首先备份。\n\n```python\n# 不是章节\nx = 1\n```\n\n## 回滚\n切回旧镜像。", encoding="utf-8")
    sections = parse_document(path)
    assert len(sections) == 2
    assert sections[0].heading_path == ["部署"]
    assert "# 不是章节" in sections[0].text
    assert sections[1].heading_path == ["部署", "回滚"]


def test_adaptive_chunks_do_not_cross_sections_and_keep_parent_mapping():
    sections = [
        ParsedSection(text="短段甲。\n\n短段乙。\n\n" + "部署步骤。" * 200, heading_path=["部署"]),
        ParsedSection(text="回滚第一步。\n\n回滚第二步。", heading_path=["回滚"]),
    ]
    parents, children = build_chunks(sections, doc_id="d", version=1, source_path="demo.md",
                                    parent_size=300, parent_overlap=30, parent_min_size=80,
                                    child_size=80, child_overlap=15)
    assert len(parents) > 2
    assert all(len(parent.text) <= 300 for parent in parents)
    assert all(len(child.text) <= 80 for child in children)
    by_id = {parent.parent_id: parent for parent in parents}
    assert "短段甲。\n\n短段乙。" in parents[0].text
    for child in children:
        parent = by_id[child.parent_id]
        assert child.text in parent.text
        assert child.heading_path == parent.heading_path
        assert not ("回滚" in child.text and "部署步骤" in child.text)
    assert parents[-1].heading_path == ["回滚"]


def test_pdf_and_docx_source_locations(tmp_path):
    from docx import Document
    from pypdf import PdfWriter

    docx_path = tmp_path / "guide.docx"
    document = Document()
    document.add_heading("接口说明", level=1)
    document.add_paragraph("返回业务状态码。")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "code"
    table.cell(0, 1).text = "含义"
    document.add_paragraph("表格之后。")
    document.save(docx_path)
    sections = parse_document(docx_path)
    assert sections[0].heading_path == ["接口说明"]
    assert sections[0].text.index("code") < sections[0].text.index("表格之后")
    assert sections[0].page_start is None  # Word page numbers must not be invented.

    empty_pdf = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(empty_pdf)
    with pytest.raises(ValueError, match="OCR"):
        parse_document(empty_pdf)

    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for message in ("Deployment guide", "Rollback procedure"):
        page = writer.add_blank_page(width=300, height=300)
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 30 250 Td ({message}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    text_pdf = tmp_path / "text.pdf"
    writer.write(text_pdf)
    pdf_sections = parse_document(text_pdf)
    assert len(pdf_sections) == 2
    assert pdf_sections[0].page_start == pdf_sections[0].page_end == 1
    assert pdf_sections[1].page_start == pdf_sections[1].page_end == 2
    assert "Rollback" in pdf_sections[1].text


def test_hybrid_rerank_update_delete_and_managed_source(tmp_path):
    kb = KnowledgeBase(settings(tmp_path), FixtureEmbedder(), FixtureReranker())
    first = tmp_path / "deployment.md"
    second = tmp_path / "billing.md"
    first.write_text("# 部署\n\n升级前备份配置。回滚使用旧镜像 old-image。", encoding="utf-8")
    second.write_text("# 账单\n\n账单每月结算，包含费用明细。", encoding="utf-8")
    try:
        records = kb.ingest([first, second])
        assert len(records) == 2
        assert len(kb.list_documents()) == 2
        assert Path(records[0].path).is_file()
        assert kb.ingest([first])[0].version == 1
        answer = kb.search("回滚", mode="full")
        assert answer[0].document_name == "deployment.md"
        assert "old-image" in answer[0].text
        assert kb.last_trace["used_sparse"]
        first.write_text("# 部署\n\n回滚改用 new-image，旧流程作废。", encoding="utf-8")
        updated = kb.ingest([first])[0]
        assert updated.version == 2
        hits = kb.search("回滚", mode="full", filters={"doc_id": updated.doc_id})
        assert hits and all(evidence.version == 2 for evidence in hits)
        assert all("old-image" not in evidence.text for evidence in hits)
        first.unlink()  # Upload-temp removal cannot break source links.
        assert Path(hits[0].source_path).is_file()
        assert kb.delete_document(updated.doc_id)
        assert kb.search("回滚", filters={"doc_id": updated.doc_id}) == []
        assert not kb.delete_document(updated.doc_id)
    finally:
        kb.close()


def test_restart_and_parallel_queries_are_isolated(tmp_path):
    configuration = settings(tmp_path)
    path = tmp_path / "handbook.txt"
    path.write_text("部署先备份。回滚要检查数据库版本。", encoding="utf-8")
    kb = KnowledgeBase(configuration, FixtureEmbedder(), FixtureReranker())
    record = kb.ingest([path])[0]
    kb.close()
    kb = KnowledgeBase(configuration, FixtureEmbedder(), FixtureReranker())
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda query: kb.search(query, mode="full"), ["部署", "回滚", "版本", "备份"]))
        assert all(result and result[0].doc_id == record.doc_id for result in results)
        assert kb.search("部署", filters={"document_name": "不存在"}) == []
        assert kb.search("部署", filters={"version": 999}) == []
    finally:
        kb.close()


def test_failed_embedding_preserves_active_document_and_rerank_falls_back(tmp_path):
    class BrokenReranker:
        def score(self, query, passages):
            raise RuntimeError("fixture outage")

    embedder = FixtureEmbedder()
    kb = KnowledgeBase(settings(tmp_path), embedder, BrokenReranker())
    path = tmp_path / "guide.txt"
    path.write_text("原版部署说明", encoding="utf-8")
    try:
        record = kb.ingest([path])[0]
        assert kb.search("部署", mode="full")
        assert kb.last_trace["reranker_fallback"]
        path.write_text("新版部署说明", encoding="utf-8")
        original_encode = embedder.encode
        embedder.encode = lambda texts: []
        with pytest.raises(ValueError, match="向量数"):
            kb.ingest([path])
        embedder.encode = original_encode
        assert kb.list_documents()[0].version == 1
        assert kb.search("部署", mode="full")[0].version == 1
        assert kb.list_documents()[0].doc_id == record.doc_id
    finally:
        kb.close()


def test_qdrant_partial_write_is_rolled_back_without_losing_old_version(tmp_path, monkeypatch):
    kb = KnowledgeBase(settings(tmp_path), FixtureEmbedder(), FixtureReranker())
    path = tmp_path / "guide.txt"
    path.write_text("旧版：部署使用蓝色镜像。", encoding="utf-8")
    try:
        record = kb.ingest([path])[0]
        original_upsert = kb.client.upsert

        def write_then_fail(*args, **kwargs):
            original_upsert(*args, **kwargs)
            raise RuntimeError("fixture network response lost after partial write")

        monkeypatch.setattr(kb.client, "upsert", write_then_fail)
        path.write_text("新版：部署改用绿色镜像。", encoding="utf-8")
        with pytest.raises(RuntimeError, match="partial write"):
            kb.ingest([path])
        hits = kb.search("镜像", mode="full")
        assert hits and hits[0].version == record.version
        assert "蓝色" in hits[0].text
        assert "绿色" not in hits[0].text
        assert kb.client.count(kb.collection, exact=True).count == record.child_count
    finally:
        kb.close()


def test_model_identity_change_requires_reindex(tmp_path):
    configuration = settings(tmp_path)
    kb = KnowledgeBase(configuration, FixtureEmbedder(), FixtureReranker())
    kb.close()
    configuration.embedding_model = "different-model"
    with pytest.raises(ValueError, match="重建索引"):
        KnowledgeBase(configuration, FixtureEmbedder(), FixtureReranker())


def test_sparse_branch_changes_ranking_when_dense_retrieval_is_wrong(tmp_path):
    class MisleadingDenseEmbedder(FixtureEmbedder):
        def encode(self, texts):
            vectors = super().encode(texts)
            for text, vector in zip(texts, vectors, strict=True):
                vector.dense = [0.0, 1.0, 0.0, 0.0] if "旧镜像" in text else [1.0, 0.0, 0.0, 0.0]
            return vectors

    kb = KnowledgeBase(settings(tmp_path), MisleadingDenseEmbedder(), FixtureReranker())
    unrelated = tmp_path / "billing.txt"
    target = tmp_path / "rollback.txt"
    unrelated.write_text("账单按月结算", encoding="utf-8")
    target.write_text("回滚需要切换旧镜像", encoding="utf-8")
    try:
        kb.ingest([unrelated, target])
        assert kb.search("回滚", mode="dense", top_k=1)[0].document_name == "billing.txt"
        assert kb.search("回滚", mode="hybrid", top_k=1)[0].document_name == "rollback.txt"
        assert kb.last_trace["used_sparse"]
    finally:
        kb.close()


def test_searches_observe_whole_versions_during_updates(tmp_path):
    kb = KnowledgeBase(settings(tmp_path), FixtureEmbedder(), FixtureReranker())
    path = tmp_path / "release.txt"
    path.write_text("部署版本 ALPHA", encoding="utf-8")
    try:
        kb.ingest([path])

        def update():
            for index in range(3):
                path.write_text(f"部署版本 BETA{index}", encoding="utf-8")
                kb.ingest([path])

        def searches():
            observations = []
            for _ in range(6):
                evidence = kb.search("部署", mode="full")[0]
                original = Path(evidence.source_path).read_text(encoding="utf-8")
                observations.append((evidence, original))
            return observations

        with ThreadPoolExecutor(max_workers=3) as pool:
            write_future = pool.submit(update)
            read_futures = [pool.submit(searches) for _ in range(2)]
            write_future.result()
            results = [future.result() for future in read_futures]
        for observations in results:
            for evidence, original in observations:
                assert evidence.text in original
                assert f"v{evidence.version}" in evidence.source_path
        assert kb.list_documents()[0].version == 4
    finally:
        kb.close()


def test_failed_cleanup_generation_is_never_reused(tmp_path, monkeypatch):
    kb = KnowledgeBase(settings(tmp_path), FixtureEmbedder(), FixtureReranker())
    path = tmp_path / "guide.txt"
    path.write_text("旧版正式内容", encoding="utf-8")
    try:
        kb.ingest([path])
        original_upsert, original_delete = kb.client.upsert, kb.client.delete

        def write_then_fail(*args, **kwargs):
            original_upsert(*args, **kwargs)
            raise RuntimeError("fixture post-write failure")

        def cannot_cleanup(*args, **kwargs):
            raise RuntimeError("fixture cleanup unavailable")

        monkeypatch.setattr(kb.client, "upsert", write_then_fail)
        monkeypatch.setattr(kb.client, "delete", cannot_cleanup)
        path.write_text("失败版本：不该返回的废弃内容", encoding="utf-8")
        with pytest.raises(RuntimeError, match="post-write"):
            kb.ingest([path])
        monkeypatch.setattr(kb.client, "upsert", original_upsert)
        monkeypatch.setattr(kb.client, "delete", original_delete)
        path.write_text("新版正式内容", encoding="utf-8")
        record = kb.ingest([path])[0]
        assert record.version == 3
        hits = kb.search("内容", mode="full")
        assert hits and all(evidence.version == 3 for evidence in hits)
        assert all("废弃" not in evidence.text for evidence in hits)
    finally:
        kb.close()


def test_updating_document_to_another_documents_content_replaces_old_version(tmp_path):
    kb = KnowledgeBase(settings(tmp_path), FixtureEmbedder(), FixtureReranker())
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("旧版本部署流程", encoding="utf-8")
    second.write_text("当前部署改用新流程", encoding="utf-8")
    try:
        first_record, second_record = kb.ingest([first, second])
        first.write_bytes(second.read_bytes())
        updated = kb.ingest([first])[0]
        assert updated.doc_id == first_record.doc_id
        assert updated.version == 2
        assert len(kb.list_documents()) == 2
        hits = kb.search("部署", mode="full", filters={"doc_id": first_record.doc_id})
        assert hits and all(item.version == 2 and item.document_name == "first.txt" for item in hits)
        assert "当前部署改用新流程" in hits[0].text
        assert "旧版本" not in hits[0].text
        assert kb.store.get_document(second_record.doc_id).version == 1
    finally:
        kb.close()


@pytest.mark.parametrize("delete_applied", [False, True])
def test_logical_delete_survives_qdrant_cleanup_failure(tmp_path, monkeypatch, caplog, delete_applied):
    kb = KnowledgeBase(settings(tmp_path), FixtureEmbedder(), FixtureReranker())
    path = tmp_path / "guide.txt"
    path.write_text("待删除版本的部署内容", encoding="utf-8")
    try:
        record = kb.ingest([path])[0]
        original_delete = kb.client.delete

        def fail_cleanup(*args, **kwargs):
            if delete_applied:
                original_delete(*args, **kwargs)
            raise RuntimeError("fixture deletion response lost" if delete_applied else "fixture cleanup unavailable")

        monkeypatch.setattr(kb.client, "delete", fail_cleanup)
        assert kb.delete_document(record.doc_id)
        assert kb.list_documents() == []
        assert kb.search("部署", mode="full") == []
        assert "document_logically_deleted_vector_cleanup_failed" in caplog.text
        expected_count = 0 if delete_applied else record.child_count
        assert kb.client.count(kb.collection, exact=True).count == expected_count
        assert not kb.delete_document(record.doc_id)

        # Reimporting identical bytes after the logical removal creates a fresh
        # generation; old orphan points cannot contaminate that generation.
        monkeypatch.setattr(kb.client, "delete", original_delete)
        reimported = kb.ingest([path])[0]
        assert reimported.version > record.version
        hits = kb.search("部署", mode="full")
        assert hits and all(item.version == reimported.version for item in hits)
    finally:
        kb.close()


def test_failed_sqlite_delete_does_not_remove_qdrant_vectors(tmp_path, monkeypatch):
    kb = KnowledgeBase(settings(tmp_path), FixtureEmbedder(), FixtureReranker())
    path = tmp_path / "guide.txt"
    path.write_text("仍然有效的部署内容", encoding="utf-8")
    try:
        record = kb.ingest([path])[0]

        def fail_source_removal(doc_id):
            raise RuntimeError("fixture sqlite transaction failure")

        monkeypatch.setattr(kb.store, "delete_document", fail_source_removal)
        with pytest.raises(RuntimeError, match="sqlite"):
            kb.delete_document(record.doc_id)
        assert kb.list_documents()[0].doc_id == record.doc_id
        assert kb.client.count(kb.collection, exact=True).count == record.child_count
        assert kb.search("部署", mode="full")[0].version == record.version
    finally:
        kb.close()


def test_duplicate_passages_are_reranked_once_without_losing_sources(tmp_path):
    class RecordingReranker(FixtureReranker):
        def __init__(self):
            self.passages = []

        def score(self, query, passages):
            self.passages = list(passages)
            return super().score(query, passages)

    reranker = RecordingReranker()
    kb = KnowledgeBase(settings(tmp_path), FixtureEmbedder(), reranker)
    first, second = tmp_path / "first.md", tmp_path / "second.md"
    first.write_text("# 共同部署\n部署之前先备份。\n\n# 独立内容\n甲。", encoding="utf-8")
    second.write_text("# 共同部署\n部署之前先备份。\n\n# 独立内容\n乙。", encoding="utf-8")
    try:
        kb.ingest([first, second])
        hits = kb.search("部署备份", mode="full")
        assert len(reranker.passages) == len(set(reranker.passages))
        assert kb.last_trace["reranking"]["duplicate_candidates"] == 1
        assert {item.document_name for item in hits} == {"first.md", "second.md"}
    finally:
        kb.close()


def test_parent_expansion_uses_injected_token_counter(tmp_path):
    configuration = settings(tmp_path)
    configuration.evidence_token_budget = 100
    def counter(text):
        return len(text) * 2
    kb = KnowledgeBase(configuration, FixtureEmbedder(), FixtureReranker(), token_counter=counter)
    path = tmp_path / "long.txt"
    path.write_text("部署备份步骤。" * 60, encoding="utf-8")
    try:
        kb.ingest([path])
        hits = kb.search("部署", mode="full")
        assert hits
        assert sum(counter(item.text) for item in hits) <= 100
        assert kb.last_trace["context_budget"]["unit"] == "tokens"
        assert kb.last_trace["context_budget"]["used"] == sum(counter(item.text) for item in hits)
    finally:
        kb.close()


def test_title_only_hits_are_backfilled_without_consuming_top_k(tmp_path):
    class HeadingFirstEmbedder(FixtureEmbedder):
        def encode(self, texts):
            vectors = super().encode(texts)
            for text, vector in zip(texts, vectors, strict=True):
                is_body = "正文" in text
                vector.dense = [0.0, 1.0, 0.0, 0.0] if is_body else [1.0, 0.0, 0.0, 0.0]
            return vectors

    configuration = settings(tmp_path)
    configuration.dense_top_k = configuration.sparse_top_k = configuration.rerank_top_k = 1
    kb = KnowledgeBase(configuration, HeadingFirstEmbedder(), FixtureReranker())
    first, second = tmp_path / "first.md", tmp_path / "second.md"
    first.write_text("# 部署甲\n\n## 正文\n部署前备份数据库。", encoding="utf-8")
    second.write_text("# 部署乙\n\n## 正文\n部署后检查就绪探针。", encoding="utf-8")
    try:
        kb.ingest([first, second])
        hits = kb.search("部署", mode="dense", top_k=1)
        assert len(hits) == 1
        assert "正文" in hits[0].text
        assert kb.last_trace["title_only_candidates_filtered"] == 2
        assert kb.last_trace["candidate_fetch_rounds"] > 1
    finally:
        kb.close()
