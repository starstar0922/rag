"""检索阶段：混合检索（向量 + BM25） + 重排。

LangChain 1.x 组件位置：
- langchain_classic.retrievers.EnsembleRetriever（1.0 起经典链/检索器迁入 langchain-classic）
- langchain_community.retrievers.BM25Retriever（jieba 中文分词）
- langchain_community.document_compressors.flashrank_rerank.FlashrankRerank（本地 CPU 重排）
权重、条数、缓存时间全部来自 .env（见 app/config.py）。
"""

import time
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import Settings, get_settings
from app.rag.store import get_vectorstore, load_collection_documents

# BM25 索引缓存：{collection: (构建时间, retriever)}
_BM25_CACHE: dict[str, tuple[float, BM25Retriever | None]] = {}


def chinese_tokenizer(text: str) -> list[str]:
    """中文按词切分，英文按空格/标点切分。"""
    import jieba  # 局部导入，减少启动耗时

    jieba.setLogLevel(60)  # 关闭 jieba 的刷屏日志
    return [t for t in jieba.cut(text) if t.strip()]


def build_bm25_retriever(collection: str, k: int, settings: Settings | None = None) -> BM25Retriever | None:
    """基于知识库现有分片构建 BM25 索引（带缓存，新增文档后会自动重建）。"""
    s = settings or get_settings()
    if not s.enable_bm25:
        return None

    cached = _BM25_CACHE.get(collection)
    if cached and time.time() - cached[0] < s.bm25_cache_ttl:
        retriever = cached[1]
    else:
        docs = load_collection_documents(collection, s)
        retriever = BM25Retriever.from_documents(docs, preprocess_func=chinese_tokenizer) if docs else None
        _BM25_CACHE[collection] = (time.time(), retriever)
    if retriever is None:
        return None
    retriever.k = k
    return retriever


def rerank(query: str, docs: list[Document], settings: Settings | None = None) -> list[Document]:
    """用 Flashrank 轻量模型对候选分片重排，返回更相关的结果。"""
    s = settings or get_settings()
    if not s.enable_rerank or not docs:
        return docs[: s.rerank_top_k]
    try:
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

        compressor = FlashrankRerank(top_n=s.rerank_top_k)
        return compressor.compress_documents(docs, query)
    except Exception:
        # 没装 flashrank 或模型下载失败时，退化为「截断前 N 条」
        return docs[: s.rerank_top_k]


class RerankRetriever(BaseRetriever):
    """把「混合检索 + 重排」包装成一个标准 Retriever，方便直接接入 LangChain 链。"""

    base_retriever: Any
    top_k: int = 4
    collection: str = "default"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        docs = self.base_retriever.invoke(query)
        return rerank(query, docs)[: self.top_k]


def build_retriever(collection: str, settings: Settings | None = None) -> BaseRetriever:
    """向量检索 + BM25 混合（权重来自 .env），再接重排。"""
    s = settings or get_settings()
    vector_retriever = get_vectorstore(collection).as_retriever(search_kwargs={"k": s.top_k})

    bm25 = build_bm25_retriever(collection, s.top_k, s)
    if bm25 is not None:
        base: BaseRetriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25], weights=[s.vector_weight, s.bm25_weight]
        )
    else:
        base = vector_retriever

    return RerankRetriever(base_retriever=base, top_k=s.rerank_top_k, collection=collection)


def retrieve(query: str, collection: str, settings: Settings | None = None) -> list[Document]:
    """对外最简单的入口：给一句话，返回最相关的若干分片。"""
    return build_retriever(collection, settings).invoke(query)


def invalidate_bm25(collection: str | None = None) -> None:
    """新增/删除文档后调用，让 BM25 索引下次重新构建。"""
    if collection:
        _BM25_CACHE.pop(collection, None)
    else:
        _BM25_CACHE.clear()
