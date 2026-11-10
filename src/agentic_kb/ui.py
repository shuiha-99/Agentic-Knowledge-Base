"""Local Gradio interface, with persistent conversations and actual stream events."""
from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import gradio as gr

if TYPE_CHECKING:
    from .config import Settings
    from .graph import GraphEngine
    from .retrieval import KnowledgeBase


class _Conversations:
    """Display history is separate from LangGraph's durable execution checkpoints."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, updated TEXT NOT NULL, body TEXT NOT NULL)")

    def choices(self) -> list[tuple[str, str]]:
        with sqlite3.connect(self.path) as db:
            return [(f"{r[0]} · {r[1][:8]}", r[1]) for r in db.execute("SELECT title,id FROM conversations ORDER BY updated DESC")]

    def get(self, thread_id: str | None) -> dict[str, Any]:
        if thread_id:
            with sqlite3.connect(self.path) as db:
                row = db.execute("SELECT body FROM conversations WHERE id=?", (thread_id,)).fetchone()
            if row:
                return json.loads(row[0])
        return {"thread_id": uuid4().hex, "title": "新会话", "history": [], "pending": False, "clarification": "", "sources": "", "stages": "", "metrics": {}}

    def put(self, state: dict[str, Any]) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO conversations VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,updated=excluded.updated,body=excluded.body", (state["thread_id"], state["title"], datetime.now(UTC).isoformat(), json.dumps(state, ensure_ascii=False)))

    def normalize(self, value: Any) -> dict[str, Any]:
        """Accept only concrete JSON-like state; never invoke callable values."""
        state = self.get(None)
        if not isinstance(value, dict):
            return state
        thread_id = value.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id.strip():
            return state
        state["thread_id"] = thread_id
        for key in ("title", "clarification", "sources", "stages"):
            if isinstance(value.get(key), str):
                state[key] = value[key]
        if isinstance(value.get("pending"), bool):
            state["pending"] = value["pending"]
        history = value.get("history")
        if isinstance(history, list):
            state["history"] = [copy.deepcopy(item) for item in history
                                if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
                                and isinstance(item.get("content"), str)]
        metrics = value.get("metrics")
        if isinstance(metrics, dict):
            try:
                json.dumps(metrics, allow_nan=False)
                state["metrics"] = copy.deepcopy(metrics)
            except (TypeError, ValueError):
                pass
        return state


def _render_sources(sources: list[dict[str, Any]]) -> str:
    parts = []
    for item in sources:
        location = " / ".join(item.get("heading_path", []))
        if item.get("page_start") is not None:
            location += f" · 第{item['page_start']}页"
        text = item.get("raw_text") or item.get("text", "")
        if len(text) > 8000:
            text = text[:8000] + "\n（界面仅展示前8000字，完整证据保留在原文和执行记录中）"
        parts.append(f"[{item.get('evidence_id', '')}] {item.get('document_name', '')}\n版本：{item.get('version', '')} · {location}\n{text}")
    return "\n\n" + ("\n\n" + "─" * 40 + "\n\n").join(parts) if parts else "本轮暂无检索来源。"


def create_app(settings: Settings, knowledge_base: KnowledgeBase, engine: GraphEngine) -> gr.Blocks:
    conversations = _Conversations(Path(settings.data_dir) / "ui_sessions.sqlite")

    def document_rows():
        return [[d.doc_id, d.name, d.version, d.parent_count, d.child_count, d.status] for d in knowledge_base.list_documents()]

    def refresh_documents():
        docs = knowledge_base.list_documents()
        return document_rows(), gr.Dropdown(choices=[(f"{d.name} · v{d.version}", d.doc_id) for d in docs], value=None)

    def ingest_files(paths, progress=gr.Progress()):
        if not paths:
            yield *refresh_documents(), "请先选择文档。"
            return
        processed = []
        for i, path in enumerate(paths):
            progress(i / len(paths), desc=f"正在处理 {Path(path).name}")
            try:
                records = knowledge_base.ingest([Path(path)])
                processed.extend(f"{r.name}：{r.parent_count}个父块、{r.child_count}个子块" for r in records)
            except Exception as exc:
                processed.append(f"{Path(path).name}：处理失败，{exc}")
            yield *refresh_documents(), "\n".join(processed)
        progress(1.0, desc="处理完成")

    def delete_document(doc_id):
        if not doc_id:
            return *refresh_documents(), "请选择需要删除的文档。"
        try:
            knowledge_base.delete_document(doc_id)
            return *refresh_documents(), "已删除所选文档及其索引。"
        except Exception as exc:
            return *refresh_documents(), f"删除失败：{exc}"

    def state_outputs(state):
        state = conversations.normalize(state)
        return (state, state["history"], state["thread_id"], state["clarification"], state["sources"], state["stages"], state["metrics"], gr.Dropdown(choices=conversations.choices(), value=state["thread_id"]))

    def new_conversation():
        state = conversations.get(None)
        conversations.put(state)
        return state_outputs(state)

    async def reconcile_checkpoint(state):
        if hasattr(engine, "session_status"):
            try:
                status = await engine.session_status(state["thread_id"])
                state["pending"] = status.get("pending_clarification", False)
                if state["pending"]:
                    state["clarification"] = status.get("question", state.get("clarification", "请补充信息"))
                else:
                    state["clarification"] = ""
            except Exception:
                # A temporarily unavailable checkpoint must not discard a pending reply.
                pass
        return state

    async def initialize_conversation(state):
        state = conversations.normalize(state)
        state = await reconcile_checkpoint(state)
        conversations.put(state)
        return state_outputs(state)

    async def select_conversation(thread_id):
        thread_id = thread_id if isinstance(thread_id, str) else None
        state = await reconcile_checkpoint(conversations.normalize(conversations.get(thread_id)))
        conversations.put(state)
        return state_outputs(state)

    async def respond(message, state, mode):
        state = conversations.normalize(state)
        message = (message or "").strip()
        if not message:
            conversations.put(state)
            yield "", *state_outputs(state)
            return
        history = state["history"]
        resuming = state.get("pending", False)
        if not history:
            state["title"] = message[:32]
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        state.update(pending=resuming, clarification=state.get("clarification", "") if resuming else "", sources="", stages="", metrics={})
        stages: list[str] = []
        text = ""
        conversations.put(state)
        yield "", *state_outputs(state)
        try:
            async for event in engine.astream(None if resuming else message, state["thread_id"], clarification=message if resuming else None, mode=mode):
                kind = event.get("type")
                if kind == "token":
                    text += event.get("text", "")
                    history[-1]["content"] = text
                elif kind == "stage":
                    stages.append(event.get("text", ""))
                    state["stages"] = "\n".join(stages)
                elif kind == "sources":
                    state["sources"] = _render_sources(event.get("sources", []))
                elif kind == "clarification":
                    question = event.get("question", "请补充问题所需的信息。")
                    options = event.get("options", [])
                    if options:
                        question += "\n可选：" + "；".join(str(x) for x in options)
                    state.update(pending=True, clarification=question)
                    history[-1]["content"] = question
                elif kind == "done":
                    history[-1]["content"] = event.get("answer") or text or history[-1]["content"]
                    state["metrics"] = event.get("metrics", {})
                    state.update(pending=False, clarification="")
                elif kind == "error":
                    history[-1]["content"] = event.get("text", "本轮执行失败，请查看运行日志。")
                    stages.append(history[-1]["content"])
                    state["stages"] = "\n".join(stages)
                yield "", *state_outputs(state)
        except asyncio.CancelledError:
            if not history[-1]["content"]:
                history[-1]["content"] = "本轮已停止。"
            raise
        except Exception as exc:
            history[-1]["content"] = f"本轮执行失败：{exc}"
        finally:
            state = await reconcile_checkpoint(state)
            conversations.put(state)
        yield "", *state_outputs(state)

    with gr.Blocks(title="Agentic 知识库智能问答") as app:
        gr.Markdown("# Agentic 知识库智能问答\n上传技术文档，通过多任务检索获取带来源的回答。")
        if settings.provider == "offline-test":
            gr.Markdown("**当前为离线测试模式：规则式测试模型用于验证流程，不代表真实大模型效果，不能用于报告 RAGAS 成绩。**")
        # Use a concrete empty value. The load callback creates a fresh thread
        # for each actual Gradio SessionState; no callable can leak into state.
        session_state = gr.State(value=None)
        with gr.Tab("知识问答"):
            with gr.Row():
                sessions = gr.Dropdown(label="会话历史", choices=conversations.choices(), value=None, scale=4)
                new = gr.Button("新建会话", scale=1)
            session_id = gr.Textbox(label="会话编号", value="", interactive=False)
            chatbot = gr.Chatbot(label="对话", height=460)
            pending = gr.Textbox(label="待补充信息（填写在下方输入框后发送即可继续）", interactive=False)
            message = gr.Textbox(label="输入问题或补充信息", placeholder="例如：比较 v1 和 v2 的部署流程，并说明升级失败时如何回滚。", lines=2)
            with gr.Row():
                send = gr.Button("发送", variant="primary")
                mode = gr.Dropdown(label="运行模式", choices=[("Agent 自动规划", "agent"), ("Dense 基线", "dense"), ("混合检索", "hybrid"), ("混合检索＋重排", "rerank"), ("混合检索＋重排＋父块", "parent")], value="agent")
            with gr.Accordion("答案来源与原文", open=True):
                sources = gr.Textbox(label="本轮证据", lines=12, interactive=False)
            with gr.Accordion("执行过程与统计", open=False):
                stages = gr.Textbox(label="执行阶段", lines=8, interactive=False)
                metrics = gr.JSON(label="本轮统计")
            state_targets = [session_state, chatbot, session_id, pending, sources, stages, metrics, sessions]
            targets = [message, *state_targets]
            send.click(respond, [message, session_state, mode], targets, concurrency_id="conversation_generation", concurrency_limit=1)
            message.submit(respond, [message, session_state, mode], targets, concurrency_id="conversation_generation", concurrency_limit=1)
            new.click(new_conversation, outputs=state_targets, concurrency_id="conversation_generation", concurrency_limit=1)
            chatbot.clear(new_conversation, outputs=state_targets, concurrency_id="conversation_generation", concurrency_limit=1)
            sessions.input(select_conversation, [sessions], state_targets, concurrency_id="conversation_generation", concurrency_limit=1)
            app.load(initialize_conversation, inputs=[session_state], outputs=state_targets)
        with gr.Tab("文档管理"):
            files = gr.File(label="上传文档", file_count="multiple", file_types=[".md", ".txt", ".pdf", ".docx"], type="filepath")
            ingest = gr.Button("解析并建立索引", variant="primary")
            documents = gr.Dataframe(headers=["文档ID", "文件名", "版本", "父块数", "子块数", "状态"], value=[], interactive=False)
            document_select = gr.Dropdown(label="选择要删除的文档", choices=[])
            with gr.Row():
                refresh = gr.Button("刷新文档列表")
                delete = gr.Button("删除所选文档", variant="stop")
            index_status = gr.Textbox(label="索引进度", lines=6, interactive=False)
            ingest.click(ingest_files, [files], [documents, document_select, index_status], concurrency_id="index_operations", concurrency_limit=1)
            delete.click(delete_document, [document_select], [documents, document_select, index_status], concurrency_id="index_operations", concurrency_limit=1)
            refresh.click(refresh_documents, outputs=[documents, document_select])
            app.load(refresh_documents, outputs=[documents, document_select])
        gr.Markdown("本地演示应用。会话展示历史与图检查点分别持久化；评分结果请通过离线评测命令生成。")
    return app.queue(default_concurrency_limit=1, max_size=32)
