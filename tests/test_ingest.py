"""基础单元测试（不需要数据库和大模型）。"""

from langchain_core.documents import Document

from app.config import Settings
from app.rag.ingest import SUPPORTED_SUFFIXES, load_file, split_documents
from app.rag.retrieval import chinese_tokenizer


def test_split_documents():
    s = Settings(chunk_size=120, chunk_overlap=20)
    doc = Document(page_content="第一章 报销制度。" * 30, metadata={"source": "x.md"})
    chunks = split_documents([doc], s)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= s.chunk_size + 50 for c in chunks)
    assert chunks[0].metadata["source"] == "x.md"


def test_load_text_file(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# 员工手册\n\n入职流程：提交材料、开通账号。", encoding="utf-8")
    docs = load_file(p)
    assert len(docs) == 1
    assert "员工手册" in docs[0].page_content


def test_supported_suffixes():
    assert {".pdf", ".docx", ".md", ".txt"} <= SUPPORTED_SUFFIXES


def test_chinese_tokenizer():
    tokens = chinese_tokenizer("差旅报销流程 reimbursement 2024")
    assert "reimbursement" in tokens
    assert any(t in ("差旅", "报销", "流程") for t in tokens)
