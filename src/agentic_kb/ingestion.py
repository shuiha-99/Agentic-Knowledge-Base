"""Parse supported documents while retaining headings and source locations."""
from __future__ import annotations

import re
from pathlib import Path

from .models import ParsedSection

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".docx"}


def _clean(text: str) -> str:
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    encodings = ("utf-16", "utf-8-sig", "gb18030") if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "gb18030")
    for encoding in encodings:
        try:
            return _clean(raw.decode(encoding))
        except UnicodeError:
            continue
    raise ValueError(f"无法识别文档编码：{path.name}，请另存为 UTF-8")


def _markdown_sections(text: str) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    headings: list[tuple[int, str]] = []
    current: list[str] = []
    current_path: list[str] = []
    fence: str | None = None

    def flush() -> None:
        body = _clean("\n".join(current))
        if body:
            sections.append(ParsedSection(text=body, heading_path=current_path.copy()))
        current.clear()

    for line in text.splitlines():
        stripped = line.lstrip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if marker:
            if fence is None:
                fence = marker.group(1)[0]
            elif stripped.startswith(fence * 3):
                fence = None
            current.append(line)
            continue
        match = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line) if fence is None else None
        if match:
            flush()
            level, heading = len(match.group(1)), match.group(2)
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, heading))
            current_path = [value for _, value in headings]
            current.append(line)
        else:
            current.append(line)
    flush()
    return sections


def _docx_sections(path: Path) -> list[ParsedSection]:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    result: list[ParsedSection] = []
    headings: list[tuple[int, str]] = []
    current: list[str] = []
    current_path: list[str] = []

    def flush() -> None:
        value = _clean("\n\n".join(current))
        if value:
            result.append(ParsedSection(text=value, heading_path=current_path.copy()))
        current.clear()

    # Walk XML children so tables remain in their original position.
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = paragraph.style.name if paragraph.style is not None else ""
            match = re.search(r"(?:Heading|标题)\s*(\d+)", style_name, re.IGNORECASE)
            if match:
                flush()
                level = int(match.group(1))
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, text))
                current_path = [value for _, value in headings]
            current.append(text)
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            rows = [" | ".join(cell.text.strip().replace("\n", "; ") for cell in row.cells) for row in table.rows]
            current.append("\n".join(rows))
    flush()
    return result


def parse_document(path: Path | str) -> list[ParsedSection]:
    """Parse text documents; scanned PDFs are rejected rather than silently indexed."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"文档不存在：{path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"暂不支持 {suffix or '无扩展名'}，支持 Markdown、TXT、文字型 PDF、DOCX")
    if suffix in {".md", ".markdown"}:
        sections = _markdown_sections(_read_text(path))
    elif suffix == ".txt":
        text = _read_text(path)
        sections = [ParsedSection(text=text, heading_path=[])] if text else []
    elif suffix == ".docx":
        sections = _docx_sections(path)
    else:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError(f"PDF 已加密，无法读取：{path.name}")
        sections = []
        for page_number, page in enumerate(reader.pages, 1):
            text = _clean(page.extract_text() or "")
            if text:
                sections.append(ParsedSection(text=text, page_start=page_number, page_end=page_number))
    if not sections:
        raise ValueError(f"未提取到可索引文本：{path.name}。扫描 PDF 需要先进行 OCR。")
    return sections
