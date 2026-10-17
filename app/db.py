"""数据库：一张 documents 表记录文档状态，向量本身由 LangChain 的 PGVector 管理。"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from sqlalchemy import DateTime, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


class Document(Base):
    """已上传并索引的文档。"""
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(512))
    doc_type: Mapped[str] = mapped_column(String(32))
    collection: Mapped[str] = mapped_column(String(64), default="default")
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending / ready / failed
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(String(2000), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


_engine = None
_session_factory = None


def get_engine(settings: Settings | None = None):
    global _engine, _session_factory
    if _engine is None:
        s = settings or get_settings()
        _engine = create_engine(s.database_url, pool_pre_ping=True)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db(settings: Settings | None = None) -> None:
    """建表 + 启用 pgvector 扩展。"""
    engine = get_engine(settings)
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    session: Session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_document(doc: Document) -> Document:
    with session_scope() as s:
        s.add(doc)
        return doc


def update_document(document_id: str, **fields: Any) -> None:
    with session_scope() as s:
        doc = s.get(Document, document_id)
        if doc is None:
            return
        for k, v in fields.items():
            setattr(doc, k, v)


def get_document(document_id: str) -> Document | None:
    with session_scope() as s:
        return s.get(Document, document_id)


def list_documents(collection: str | None = None, limit: int = 200) -> list[Document]:
    from sqlalchemy import select  # 局部导入，保持文件清爽

    with session_scope() as s:
        stmt = select(Document).order_by(Document.created_at.desc()).limit(limit)
        if collection:
            stmt = stmt.where(Document.collection == collection)
        return list(s.execute(stmt).scalars().all())


def delete_document(document_id: str) -> bool:
    """删除文档记录，同时删除它在向量库中的分片。"""
    delete_document_chunks(document_id)
    with session_scope() as s:
        doc = s.get(Document, document_id)
        if doc is None:
            return False
        s.delete(doc)
        return True


def delete_document_chunks(document_id: str) -> int:
    """按 metadata 中的 document_id 删除向量分片（LangChain PGVector 的表）。"""
    engine = get_engine()
    sql = text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'document_id' = :doc_id")
    try:
        with engine.begin() as conn:
            result = conn.execute(sql, {"doc_id": document_id})
        return result.rowcount
    except Exception:
        # 表还没创建（知识库为空）时直接忽略
        return 0
