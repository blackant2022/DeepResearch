"""
src/multimodal/attachments.py — 解析用户上传的图片与文档附件
"""
from __future__ import annotations

import base64
import mimetypes
import tempfile
from pathlib import Path
from typing import Any

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_DOC_EXT = {".pdf", ".docx", ".txt", ".md", ".csv"}


def _mime(name: str) -> str:
    mt, _ = mimetypes.guess_type(name)
    return mt or "application/octet-stream"


def _extract_doc_text(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext in {".txt", ".md", ".csv"}:
        for enc in ("utf-8", "gbk", "gb2312"):
            try:
                return data.decode(enc).strip()
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"无法解码文本文件 {name}")

    from src.rag.ingest import _read

    suffix = ext or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = Path(tmp.name)
    try:
        return (_read(path) or "").strip()
    finally:
        path.unlink(missing_ok=True)


def parse_upload(name: str, data: bytes, *, max_doc_chars: int = 12000) -> dict[str, Any]:
    """将单个上传文件转为可序列化附件 dict。"""
    ext = Path(name).suffix.lower()
    if ext in _IMAGE_EXT:
        return {
            "kind": "image",
            "name": Path(name).name,
            "mime": _mime(name),
            "b64": base64.b64encode(data).decode("ascii"),
        }
    if ext in _DOC_EXT:
        text = _extract_doc_text(name, data)
        if len(text) > max_doc_chars:
            text = text[:max_doc_chars] + "\n…（已截断）"
        return {
            "kind": "document",
            "name": Path(name).name,
            "mime": _mime(name),
            "text": text,
            "chars": len(text),
        }
    raise ValueError(f"不支持的文件类型：{ext}（支持图片与 PDF/Word/TXT/MD/CSV）")


def parse_uploads(files: list[tuple[str, bytes]], *, max_doc_chars: int = 12000) -> list[dict[str, Any]]:
    return [parse_upload(n, d, max_doc_chars=max_doc_chars) for n, d in files]


def has_images(attachments: list[dict[str, Any]] | None) -> bool:
    return any(a.get("kind") == "image" for a in (attachments or []))


def document_context(attachments: list[dict[str, Any]] | None) -> str:
    blocks: list[str] = []
    for att in attachments or []:
        if att.get("kind") != "document":
            continue
        text = (att.get("text") or "").strip()
        if text:
            blocks.append(f"### {att.get('name', '文档')}\n{text}")
    return "\n\n".join(blocks)


def image_data_url(att: dict[str, Any]) -> str:
    mime = att.get("mime") or "image/png"
    b64 = att.get("b64") or ""
    return f"data:{mime};base64,{b64}"


def build_enhanced_question(question: str, attachments: list[dict[str, Any]] | None) -> str:
    """把附件文档文本拼入问题（图片由视觉模型单独处理）。"""
    q = (question or "").strip()
    doc = document_context(attachments)
    if doc:
        q = f"{q}\n\n【用户本次上传的文档内容】\n{doc}" if q else f"【用户本次上传的文档内容】\n{doc}"
    return q.strip() or "请根据附件内容回答。"
