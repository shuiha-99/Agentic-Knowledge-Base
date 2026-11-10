"""SQLite source of truth for active document versions and parent passages."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .models import DocumentRecord, ParentChunk


class DocumentStore:
    def __init__(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS parents (
                parent_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS parent_document ON parents(doc_id, version);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        self.connection.commit()

    def list_documents(self) -> list[DocumentRecord]:
        with self.lock:
            rows = self.connection.execute("SELECT data FROM documents ORDER BY doc_id").fetchall()
        return [DocumentRecord.model_validate_json(row[0]) for row in rows]

    def metadata(self, key: str) -> str | None:
        with self.lock:
            row = self.connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        with self.lock, self.connection:
            self.connection.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, value))

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        with self.lock:
            row = self.connection.execute("SELECT data FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return DocumentRecord.model_validate_json(row[0]) if row else None

    def get_parent(self, parent_id: str) -> ParentChunk | None:
        with self.lock:
            row = self.connection.execute("SELECT data FROM parents WHERE parent_id = ?", (parent_id,)).fetchone()
        return ParentChunk.model_validate_json(row[0]) if row else None

    def replace_document(self, document: DocumentRecord, parents: list[ParentChunk]) -> None:
        with self.lock, self.connection:
            self.connection.execute("DELETE FROM parents WHERE doc_id = ?", (document.doc_id,))
            self.connection.executemany(
                "INSERT INTO parents(parent_id, doc_id, version, data) VALUES (?, ?, ?, ?)",
                [(p.parent_id, p.doc_id, p.version, p.model_dump_json()) for p in parents],
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO documents(doc_id, data) VALUES (?, ?)",
                (document.doc_id, document.model_dump_json()),
            )

    def delete_document(self, doc_id: str) -> bool:
        with self.lock, self.connection:
            self.connection.execute("DELETE FROM parents WHERE doc_id = ?", (doc_id,))
            cursor = self.connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cursor.rowcount > 0

    def close(self) -> None:
        with self.lock:
            self.connection.close()
