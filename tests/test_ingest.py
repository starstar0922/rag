"""基础单元测试（不需要数据库和大模型）。"""

from langchain_core.documents import Document

from app.config import Settings, load_prompt
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


def test_settings_from_env(monkeypatch, tmp_path):
    """配置必须能从环境变量/.env 读取（含 JSON 数组类型的分隔符）。"""
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_SEPARATORS", '["。", "\\n"]')
    monkeypatch.setenv("ENABLE_BM25", "false")
    s = Settings(upload_dir=tmp_path / "u", eval_dir=tmp_path / "e")
    assert s.chunk_size == 500
    assert s.chunk_separators == ["。", "\n"]
    assert s.enable_bm25 is False


def test_load_prompt_fallback(tmp_path):
    assert load_prompt(None, "默认提示词") == "默认提示词"
    f = tmp_path / "p.txt"
    f.write_text("文件里的提示词", encoding="utf-8")
    assert load_prompt(f, "默认提示词") == "文件里的提示词"
