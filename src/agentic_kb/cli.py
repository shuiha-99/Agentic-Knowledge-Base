"""Command-line entry points. Imports never start services or download models."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from rich.console import Console

from .config import Settings

app = typer.Typer(help="Agentic 知识库问答：文档、双层检索图与可复现评测", no_args_is_help=True)
console = Console()


def _readonly_settings() -> Settings:
    """Match dotenv/environment precedence without creating runtime directories."""
    from dotenv import dotenv_values
    source = {**dotenv_values(".env"), **os.environ}
    values = {}
    for name in Settings.model_fields:
        value = source.get("KB_" + name.upper())
        if value is not None and value.strip():
            values[name] = value
    settings = Settings(**values)
    settings.data_dir = settings.data_dir.resolve()
    return settings


def _isolated_ragas_metadata() -> dict:
    default = Path.cwd() / ".venv-ragas" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = Path(os.getenv("KB_RAGAS_PYTHON", str(default))).expanduser().resolve()
    result = {"python_path": str(python), "environment_exists": python.is_file(),
              "expected_ragas_version": "0.3.3", "runtime_import_checked": False,
              "note": "仅通过独立解释器查询包版本；未导入ragas、未加载模型或执行评分。"}
    if not python.is_file():
        return {**result, "status": "environment_missing", "packages": {},
                "action": "运行 scripts/setup.ps1 -WithEvaluation，或配置 KB_RAGAS_PYTHON。"}
    code = """import json,sys
from importlib.metadata import PackageNotFoundError,version
packages={}
for name in ('ragas','langchain-core','langchain-community','langchain-ollama'):
    try: packages[name]=version(name)
    except PackageNotFoundError: packages[name]=None
print(json.dumps({'python_version':sys.version.split()[0],'packages':packages}))
"""
    try:
        response = subprocess.run([str(python), "-I", "-c", code], capture_output=True, text=True,
            encoding="utf-8", timeout=10, check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if response.returncode:
            return {**result, "status": "metadata_query_failed", "packages": {},
                    "error": response.stderr.strip()[-1000:] or f"exit_code={response.returncode}"}
        metadata = json.loads(response.stdout)
        packages = metadata["packages"]
        if not isinstance(packages, dict):
            raise ValueError("独立环境版本查询未返回packages对象")
        return {**result, **metadata, "status": "metadata_available",
                "ragas_version_matches": packages.get("ragas") == "0.3.3"}
    except subprocess.TimeoutExpired:
        return {**result, "status": "metadata_query_timeout", "packages": {}, "timeout_seconds": 10}
    except (OSError, ValueError, KeyError) as exc:
        return {**result, "status": "metadata_query_failed", "packages": {}, "error": str(exc)}


def components(offline_test: bool = False, data_dir: Path | None = None):
    from .graph import GraphEngine
    from .providers import build_providers
    from .retrieval import KnowledgeBase

    settings = Settings.from_env()
    if data_dir is not None:
        settings.data_dir = data_dir.resolve()
    if offline_test:
        settings.provider = "offline-test"
        settings.data_dir = settings.data_dir / "offline-test"
        console.print("[yellow]离线流程测试：规则脚本和测试向量，不代表模型能力，禁止报告为质量评测。[/yellow]")
    llm, embedder, reranker = build_providers(settings)
    kb = KnowledgeBase(settings, embedder, reranker, token_counter=llm.count_tokens)
    return settings, kb, GraphEngine(settings, llm, kb)


@app.command()
def doctor():
    """检查配置、依赖和 Ollama；不下载模型。"""
    from .runtime import ollama_status
    settings = _readonly_settings()
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("langgraph", "gradio", "qdrant_client", "FlagEmbedding", "transformers", "torch")}
    result = {"data_dir": str(settings.data_dir),
              "main_environment": {"python_path": sys.executable, "dependencies": dependencies},
              "isolated_ragas_environment": _isolated_ragas_metadata(),
              "ollama": ollama_status(settings), "chat_model": settings.chat_model,
              "tokenizer_model": settings.tokenizer_model,
              "structured_model": settings.structured_model,
              "structured_tokenizer_model": settings.structured_tokenizer_model,
              "structured_model_schemas": settings.structured_model_schemas,
              "embedding_model": settings.embedding_model, "reranker_model": settings.reranker_model,
              "cached_model_directories": [p.name for p in settings.models_dir.glob("models--*")]}
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("start-ollama")
def start_local_ollama():
    """使用 .env 中配置的程序启动本地服务，日志保存在项目内。"""
    from .runtime import start_ollama
    console.print_json(json.dumps(start_ollama(Settings.from_env()), ensure_ascii=False))


@app.command("prepare-models")
def prepare_models():
    """下载并加载检索、重排和分词模型；首次运行需要网络。"""
    from .providers import build_providers
    settings = Settings.from_env()
    if settings.structured_model and not settings.structured_tokenizer_model:
        raise typer.BadParameter(
            "已配置 KB_STRUCTURED_MODEL，请同时填写对应的 KB_STRUCTURED_TOKENIZER_MODEL。",
            param_hint="KB_STRUCTURED_TOKENIZER_MODEL")
    llm, embedder, reranker = build_providers(settings)
    try:
        console.print("准备生成模型分词器……")
        llm.count_tokens("知识库问答")
        if settings.structured_model:
            console.print(f"准备辅助模型分词器：{settings.structured_tokenizer_model}……")
            llm.count_structured_tokens("辅助模型分词检查")
        console.print("准备 BGE-M3 Dense 与 Sparse 模型……")
        vectors = embedder.encode(["AtlasDesk v2 使用环境变量配置。"])
        console.print(f"Dense维度={len(vectors[0].dense)}，Sparse非零项={len(vectors[0].sparse_indices)}")
        console.print("准备 Cross-Encoder……")
        reranker.score("如何配置v2？", ["v2 使用环境变量配置。"])
        console.print("模型已准备完成。Ollama 的主、辅助生成模型需在服务中另行存在。")
    finally:
        for adapter in (embedder, reranker):
            close = getattr(adapter, "close", None)
            if close is not None:
                close()


@app.command()
def ingest(paths: Annotated[list[Path], typer.Argument(help="文件或目录")], offline_test: bool = False):
    """导入或更新文档；支持 md/txt/pdf/docx。"""
    _, kb, _ = components(offline_test)
    try:
        files = []
        for path in paths:
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in {".md", ".txt", ".pdf", ".docx"})
                         if path.is_dir() else [path])
        for record in kb.ingest(files):
            console.print(f"{record.name} · v{record.version} · {record.parent_count}父块 / {record.child_count}子块")
    finally:
        kb.close()


@app.command("list-documents")
def list_documents(offline_test: bool = False):
    _, kb, _ = components(offline_test)
    try:
        console.print_json(json.dumps([d.model_dump() for d in kb.list_documents()], ensure_ascii=False))
    finally:
        kb.close()


@app.command("delete-document")
def delete_document(doc_id: str, offline_test: bool = False):
    _, kb, _ = components(offline_test)
    try:
        kb.delete_document(doc_id)
        console.print("文档与索引已删除。")
    finally:
        kb.close()


@app.command()
def ask(question: str = "", thread_id: str = "", clarification: str | None = None,
        mode: str = "agent", offline_test: bool = False):
    """执行问答；使用相同thread-id及clarification继续暂停的会话。"""
    _, kb, engine = components(offline_test)
    thread_id = thread_id or uuid4().hex
    console.print(f"会话：{thread_id}")

    async def run():
        async for event in engine.astream(question or None, thread_id, clarification=clarification, mode=mode):
            if event["type"] == "token":
                console.print(event["text"], end="", markup=False)
            elif event["type"] == "done":
                console.print("\n")
                console.print_json(json.dumps({k: v for k, v in event.items() if k != "answer"}, ensure_ascii=False))
            elif event["type"] in {"stage", "error"}:
                console.print(event["text"], markup=False)
            elif event["type"] == "clarification":
                console.print(f"需要补充：{event['question']}")
            elif event["type"] == "sources":
                console.print("来源：" + ", ".join(s["document_name"] for s in event["sources"]))
        await engine.close()
    try:
        asyncio.run(run())
    finally:
        kb.close()


@app.command()
def evaluate(dataset: Path = Path("data/eval/demo_cases.jsonl"), output_dir: Path | None = None,
             mode: str = "agent", ragas: bool = False, split: str = "test", offline_test: bool = False):
    """运行固定数据集；质量评分需要 --ragas，离线脚本禁止评分。"""
    from .evaluation import evaluate_dataset
    settings, kb, engine = components(offline_test)
    output_dir = output_dir or settings.data_dir / "evaluations" / f"{mode}-{uuid4().hex[:12]}"
    try:
        report = asyncio.run(evaluate_dataset(settings, engine, dataset, output_dir,
                           mode=mode, use_ragas=ragas, split=None if split == "all" else split))
        console.print_json(json.dumps(report, ensure_ascii=False, default=str))
    finally:
        kb.close()


@app.command()
def serve(offline_test: bool = False, open_browser: bool = False):
    """启动本地中文界面。"""
    from .ui import create_app
    settings, kb, engine = components(offline_test)
    try:
        create_app(settings, kb, engine).queue(default_concurrency_limit=4).launch(
            server_name=settings.host, server_port=settings.port,
            share=False, inbrowser=open_browser,
            allowed_paths=[str(settings.data_dir / "documents")],
        )
    finally:
        kb.close()

