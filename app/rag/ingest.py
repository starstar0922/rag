"""索引阶段：加载文档 -> 切分 -> 向量化 -> 存入 pgvector。

全部使用 LangChain 组件：
  PyMuPDFLoader（PDF，可抽取表格）/ Docx2txtLoader（Word）/ TextLoader（Markdown、TXT）
  RecursiveCharacterTextSplitter（递归切分，分隔符来自 .env 的 CHUNK_SEPARATORS）
  PGVector.add_documents（自动完成向量化与入库）
"""

import time
import uuid
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings
from app.db import Document as DocumentRecord
from app.db import save_document, update_document
from app.rag.store import get_vectorstore

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def load_file(path: str | Path, settings: Settings | None = None) -> list[Document]:
    """按后缀选择 Loader，返回按页/按文档切好的 LangChain Document。"""
    s = settings or get_settings()
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        # PyMuPDF 按页输出文本，并可把表格转成 Markdown（开关在 .env：PDF_EXTRACT_TABLES）
        try:
            loader = PyMuPDFLoader(str(path), extract_tables="markdown" if s.pdf_extract_tables else None)
        except TypeError:  # 老版本没有 extract_tables 参数
            loader = PyMuPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix in {".md", ".markdown", ".txt"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件类型：{suffix}，支持 {sorted(SUPPORTED_SUFFIXES)}")
    return loader.load()


def split_documents(docs: list[Document], settings: Settings | None = None) -> list[Document]:
    """递归切分：优先按段落/句子切，切不动再按字符切（分隔符可在 .env 配置）。"""
    s = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=s.chunk_separators,
    )
    return splitter.split_documents(docs)


def ingest_file(
    path: str | Path,
    collection: str | None = None,
    original_name: str | None = None,
    settings: Settings | None = None,
) -> dict:
    """完整索引流程，返回处理结果字典（供 API / CLI 使用）。"""
    s = settings or get_settings()
    path = Path(path)
    collection = collection or s.default_collection
    name = original_name or path.name
    document_id = uuid.uuid4().hex
    started = time.time()

    save_document(
        DocumentRecord(
            id=document_id,
            name=name,
            doc_type=path.suffix.lower().lstrip("."),
            collection=collection,
            status="pending",
        )
    )

    try:
        raw_docs = load_file(path, s)
        # PDF 的页码从 0 开始，这里改成从 1 开始，和 PDF 阅读器显示一致
        for d in raw_docs:
            if d.metadata.get("page") is not None:
                d.metadata["page"] = int(d.metadata["page"]) + 1
        chunks = split_documents(raw_docs, s)

        # 给每个分片补充溯源信息：文档名、页码（PDF 有）、文档 ID、知识库
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "document_id": document_id,
                    "document_name": name,
                    "collection": collection,
                    "chunk_id": f"{document_id}_{i}",
                }
            )

        vectorstore = get_vectorstore(collection)
        vectorstore.add_documents(chunks)  # 内部自动调用 Embedding 并写入 pgvector

        update_document(document_id, status="ready", chunk_count=len(chunks))
        return {
            "document_id": document_id,
            "name": name,
            "collection": collection,
            "doc_type": path.suffix.lower().lstrip("."),
            "status": "ready",
            "chunk_count": len(chunks),
            "elapsed_seconds": round(time.time() - started, 2),
        }
    except Exception as exc:
        update_document(document_id, status="failed", error=str(exc)[:2000])
        return {
            "document_id": document_id,
            "name": name,
            "collection": collection,
            "doc_type": path.suffix.lower().lstrip("."),
            "status": "failed",
            "chunk_count": 0,
            "error": str(exc)[:2000],
            "elapsed_seconds": round(time.time() - started, 2),
        }
