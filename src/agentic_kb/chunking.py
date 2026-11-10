"""Rule-based adaptive parent chunks and recursive-character child chunks."""
from __future__ import annotations

import uuid
from collections.abc import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import ChildChunk, ParentChunk, ParsedSection

_NAMESPACE = uuid.UUID("427ed077-cf59-4d98-a0d7-81b66d31adef")


def stable_id(value: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, value))


def _splitter(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    if size < 16 or overlap < 0 or overlap >= size:
        raise ValueError("切块长度必须至少16字符，重叠长度必须非负且小于块长度")
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", ";", "；", ". ", " ", ""],
        keep_separator=True,
        length_function=len,
        strip_whitespace=True,
    )


def build_chunks(
    sections: Sequence[ParsedSection],
    *,
    doc_id: str,
    version: int,
    source_path: str,
    parent_size: int = 1600,
    parent_overlap: int = 100,
    parent_min_size: int = 240,
    child_size: int = 420,
    child_overlap: int = 60,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Merge small neighbors only inside a section, then recursively split children.

    Sizes here deliberately mean characters, not tokens. Token budgets for model
    inputs are a separate concern handled by the context manager.
    """
    if parent_min_size < 0 or parent_min_size > parent_size:
        raise ValueError("父块合并阈值必须处于0和父块长度之间")
    parent_splitter = _splitter(parent_size, parent_overlap)
    child_splitter = _splitter(child_size, child_overlap)
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    for section_index, section in enumerate(sections):
        # Split into paragraph fragments, merge small neighboring fragments, and
        # split oversized ones without ever crossing a parsed section boundary.
        fragments = [part.strip() for part in section.text.split("\n\n") if part.strip()]
        merged: list[str] = []
        buffer = ""
        for fragment in fragments:
            candidate = f"{buffer}\n\n{fragment}" if buffer else fragment
            if buffer and len(candidate) > parent_size:
                merged.extend(parent_splitter.split_text(buffer))
                buffer = fragment
            else:
                buffer = candidate
            if len(buffer) > parent_size:
                pieces = parent_splitter.split_text(buffer)
                merged.extend(pieces[:-1])
                buffer = pieces[-1] if pieces else ""
        if buffer:
            if merged and len(buffer) < parent_min_size and len(merged[-1]) + len(buffer) + 2 <= parent_size:
                merged[-1] += "\n\n" + buffer
            else:
                merged.append(buffer)
        for parent_index, text in enumerate(merged):
            parent_id = stable_id(f"{doc_id}:{version}:{section_index}:{parent_index}:{text}")
            parent = ParentChunk(
                parent_id=parent_id,
                doc_id=doc_id,
                text=text,
                heading_path=section.heading_path,
                page_start=section.page_start,
                page_end=section.page_end,
                version=version,
                source_path=source_path,
            )
            parents.append(parent)
            for child_index, child_text in enumerate(child_splitter.split_text(text)):
                children.append(ChildChunk(
                    chunk_id=stable_id(f"{parent_id}:{child_index}:{child_text}"),
                    parent_id=parent_id,
                    doc_id=doc_id,
                    text=child_text,
                    heading_path=section.heading_path,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    version=version,
                ))
    return parents, children
