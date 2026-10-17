"""模型（Embedding / Chat）与向量库（PGVector）的集中创建处：换模型只改这里。

LangChain 1.x 说明：
- 模型统一走 langchain-openai（OpenAI 兼容协议，可对接 DeepSeek / 通义 / vLLM 等）
- 向量库用 langchain-postgres 的 PGVector（Postgres + pgvector）
"""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sqlalchemy import create_engine, text

from app.config import Settings, get_settings


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """文本向量化模型。

    配置了 OPENAI_API_KEY 就走真实模型；没配置则用 LangChain 自带的「假向量」，
    保证系统能跑通（检索质量无意义，仅用于联调）。
    """
    s = settings or get_settings()
    if s.openai_api_key:
        return OpenAIEmbeddings(
            model=s.embedding_model,
            api_key=s.openai_api_key,
            base_url=s.openai_base_url,
        )
    from langchain_community.embeddings import DeterministicFakeEmbedding  # 局部导入，缺依赖也能启动

    return DeterministicFakeEmbedding(size=s.embedding_dim)


def get_llm(settings: Settings | None = None, streaming: bool = False) -> ChatOpenAI:
    """对话模型（OpenAI 兼容协议）。"""
    s = settings or get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        api_key=s.openai_api_key or "empty",
        base_url=s.openai_base_url,
        temperature=s.llm_temperature,
        timeout=s.llm_timeout,
        streaming=streaming,
    )


@lru_cache(maxsize=8)
def get_vectorstore(collection: str):
    """PGVector 向量库（底层就是 Postgres + pgvector）。"""
    from langchain_postgres import PGVector  # 局部导入，避免没装也能启动

    s = get_settings()
    vs = PGVector(
        embeddings=get_embeddings(s),
        collection_name=collection,
        connection=s.database_url,
        use_jsonb=True,
    )
    try:
        vs.create_collection()  # 幂等：集合已存在则跳过
    except Exception:
        pass
    return vs


def load_collection_documents(collection: str, settings: Settings | None = None) -> list[Document]:
    """把某个知识库的所有分片读出来（用于构建 BM25 稀疏检索索引）。"""
    s = settings or get_settings()
    engine = create_engine(s.database_url)
    sql = text(
        """
        SELECT e.document AS content, e.cmetadata AS metadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = :name
        """
    )
    docs: list[Document] = []
    try:
        with engine.connect() as conn:
            for row in conn.execute(sql, {"name": collection}):
                docs.append(Document(page_content=row.content or "", metadata=dict(row.metadata or {})))
    except Exception:
        # 表不存在（知识库为空）时返回空列表
        return []
    return docs
