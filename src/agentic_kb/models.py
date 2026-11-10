"""Shared contracts; all external/LLM data is validated at the boundary."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VectorEmbedding(BaseModel):
    dense: list[float]
    sparse_indices: list[int] = Field(default_factory=list)
    sparse_values: list[float] = Field(default_factory=list)


class ParsedSection(BaseModel):
    text: str
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


class DocumentRecord(BaseModel):
    doc_id: str
    name: str
    path: str
    sha256: str
    version: int = 1
    status: str = "active"
    created_at: str = ""
    parent_count: int = 0
    child_count: int = 0


class ParentChunk(BaseModel):
    parent_id: str
    doc_id: str
    text: str
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    version: int = 1
    source_path: str = ""


class ChildChunk(BaseModel):
    chunk_id: str
    parent_id: str
    doc_id: str
    text: str
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    version: int = 1


class Evidence(BaseModel):
    evidence_id: str
    parent_id: str
    doc_id: str
    document_name: str
    text: str
    child_ids: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    version: int = 1
    score: float = 0.0
    source_path: str = ""
    raw_text: str = ""


class RetrievalTask(BaseModel):
    task_id: str
    query: str
    depends_on: list[str] = Field(default_factory=list)
    constraints: dict[str, str] = Field(default_factory=dict)
    filters: dict[str, str] = Field(default_factory=dict)


class TaskResult(BaseModel):
    task_id: str
    query: str
    status: Literal["succeeded", "partial", "failed"] = "succeeded"
    answer: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    attempts: int = 0
    error: str | None = None

