"""接口请求 / 响应模型（Pydantic）。"""

from typing import Any

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    name: str
    doc_type: str
    collection: str
    status: str
    chunk_count: int = 0
    error: str = ""
    created_at: str | None = None


class SearchRequest(BaseModel):
    query: str
    collection: str = "default"
    top_k: int = 5


class SearchHit(BaseModel):
    content: str
    document_name: str = ""
    page: Any = None
    source: str = ""


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str
    collection: str = "default"
    history: list[dict[str, str]] | None = None  # [{"role": "user"/"assistant", "content": "..."}]


class SourceOut(BaseModel):
    index: int
    document_name: str = ""
    page: Any = None
    source: str = ""
    snippet: str = ""


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)


class EvalRequest(BaseModel):
    collection: str = "default"
    size: int = 20
    use_ragas: bool = True
