"""Validated, project-local configuration."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    data_dir: Path = Path("runtime")
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_executable: Path | None = None
    ollama_models_dir: Path | None = None
    chat_model: str = "qwen2.5:7b"
    knowledge_base_scope: str = ""
    tokenizer_model: str = "Qwen/Qwen2.5-7B-Instruct"
    structured_model: str | None = None
    structured_tokenizer_model: str | None = None
    structured_model_schemas: str = ""
    models_cache_dir: Path | None = None
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    low_memory_mode: bool = True
    provider: Literal["ollama", "offline-test"] = "ollama"
    qdrant_url: str | None = None
    qdrant_collection: str = "agentic_chunks"
    parent_chunk_size: int = Field(1600, ge=100)
    parent_chunk_overlap: int = Field(100, ge=0)
    parent_min_size: int = Field(240, ge=1)
    child_chunk_size: int = Field(420, ge=40)
    child_chunk_overlap: int = Field(60, ge=0)
    dense_top_k: int = Field(20, ge=1)
    sparse_top_k: int = Field(20, ge=1)
    rerank_top_k: int = Field(8, ge=1)
    evidence_token_budget: int = Field(3500, ge=100)
    context_window_tokens: int = Field(8192, ge=512)
    output_reserve_tokens: int = Field(1536, ge=128)
    max_parallel_tasks: int = Field(2, ge=1, le=16)
    max_tasks: int = Field(4, ge=1, le=16)
    max_waves: int = Field(2, ge=1, le=4)
    max_retrieval_attempts: int = Field(2, ge=1, le=5)
    evidence_review_policy: Literal["always", "complex"] = "complex"
    graph_recursion_limit: int = Field(80, ge=10)
    recent_messages: int = Field(4, ge=0)
    summary_trigger_ratio: float = Field(0.75, gt=0, lt=1)
    summary_target_ratio: float = Field(0.60, gt=0, lt=1)
    compression_enabled: bool = True
    evidence_compression_strategy: Literal["extractive", "llm"] = "extractive"
    summary_recent_fraction: float = Field(0.52, gt=0, lt=1)
    request_timeout_seconds: float = Field(180.0, gt=0)
    task_timeout_seconds: float = Field(240.0, gt=0)
    run_timeout_seconds: float = Field(600.0, gt=0)
    structured_retries: int = Field(2, ge=0, le=4)
    temperature: float = Field(0.0, ge=0, le=2)
    host: str = "127.0.0.1"
    port: int = Field(7860, ge=1024, le=65535)

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def models_dir(self) -> Path:
        return self.models_cache_dir if self.models_cache_dir is not None else self.data_dir / "models"

    @model_validator(mode="after")
    def validate_budgets(self):
        if self.parent_chunk_overlap >= self.parent_chunk_size:
            raise ValueError("parent overlap must be smaller than chunk size")
        if self.child_chunk_overlap >= self.child_chunk_size:
            raise ValueError("child overlap must be smaller than chunk size")
        if self.output_reserve_tokens >= self.context_window_tokens:
            raise ValueError("output reserve must be smaller than context window")
        if self.summary_target_ratio >= self.summary_trigger_ratio:
            raise ValueError("summary target must be below trigger")
        return self

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> Settings:
        load_dotenv(env_file, override=False)
        values = {}
        for name in cls.model_fields:
            value = os.getenv("KB_" + name.upper())
            if value is not None and value.strip():
                values[name] = value
        settings = cls(**values)
        settings.data_dir = settings.data_dir.resolve()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(settings.models_dir / "hf"))
        os.environ.setdefault("TORCH_HOME", str(settings.models_dir / "torch"))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        return settings

