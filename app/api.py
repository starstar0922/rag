"""HTTP 接口：文档上传、检索、问答（流式）、评测。"""

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.db import delete_document, get_document, list_documents
from app.rag.chain import answer, astream_answer
from app.rag.evaluate import get_run, list_runs, run_evaluation
from app.rag.ingest import SUPPORTED_SUFFIXES, ingest_file
from app.rag.retrieval import invalidate_bm25, retrieve
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentOut,
    EvalRequest,
    SearchRequest,
    SearchResponse,
    SearchHit,
    SourceOut,
)

settings = get_settings()


def check_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """可选鉴权：配置了 API_KEY 才需要传 X-API-Key。"""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="缺少或错误的 API Key")


# 所有接口统一走鉴权（未配置 API_KEY 时自动放行）
router = APIRouter(dependencies=[Depends(check_api_key)])


# ------------------------------------------------------------------ 系统
@router.get("/health")
def health() -> dict:
    try:
        from sqlalchemy import text  # 局部导入

        from app.db import get_engine  # 局部导入

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        db = "ok"
    except Exception as exc:
        db = f"error: {exc}"
    return {
        "status": "ok",
        "llm": settings.llm_model,
        "embedding": settings.embedding_model,
        "llm_configured": bool(settings.openai_api_key),
        "database": db,
        "collection": settings.default_collection,
    }


# ------------------------------------------------------------------ 文档
@router.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = Form("default"),
    sync: bool = Form(False),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"不支持的文件类型 {suffix}，支持 {sorted(SUPPORTED_SUFFIXES)}")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = (file.filename or "upload").replace("/", "_")
    target = settings.upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    target.write_bytes(await file.read())

    if sync:  # 小文件可以同步等待，方便调试
        result = ingest_file(target, collection=collection, original_name=safe_name)
        invalidate_bm25(collection)
        return result

    background_tasks.add_task(_ingest_task, target, collection, safe_name)
    return {"status": "processing", "name": safe_name, "collection": collection}


def _ingest_task(path: Path, collection: str, name: str) -> None:
    ingest_file(path, collection=collection, original_name=name)
    invalidate_bm25(collection)  # 让 BM25 索引下次重建


@router.get("/documents", response_model=list[DocumentOut])
def documents(collection: str | None = None) -> list[DocumentOut]:
    rows = list_documents(collection)
    return [
        DocumentOut(
            id=d.id,
            name=d.name,
            doc_type=d.doc_type,
            collection=d.collection,
            status=d.status,
            chunk_count=d.chunk_count,
            error=d.error,
            created_at=d.created_at.isoformat() if d.created_at else None,
        )
        for d in rows
    ]


@router.delete("/documents/{document_id}")
def remove_document(document_id: str) -> dict:
    if get_document(document_id) is None:
        raise HTTPException(404, "文档不存在")
    delete_document(document_id)
    invalidate_bm25()
    return {"ok": True}


# ------------------------------------------------------------------ 检索
@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    docs = retrieve(req.query, req.collection)[: req.top_k]
    return SearchResponse(
        query=req.query,
        hits=[
            SearchHit(
                content=d.page_content,
                document_name=d.metadata.get("document_name", ""),
                page=d.metadata.get("page"),
                source=d.metadata.get("source", ""),
            )
            for d in docs
        ],
    )


# ------------------------------------------------------------------ 问答
@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = answer(req.question, req.collection, req.history)
    return ChatResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[SourceOut(**s) for s in result["sources"]],
        contexts=result["contexts"],
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    async def gen():
        async for event in astream_answer(req.question, req.collection, req.history):
            yield f"data: {_json(event)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _json(payload: dict) -> str:
    import json  # 局部导入，保持文件简洁

    return json.dumps(payload, ensure_ascii=False)


# ------------------------------------------------------------------ 评测
@router.post("/eval/run")
def eval_run(req: EvalRequest, background_tasks: BackgroundTasks) -> dict:
    run_id = uuid.uuid4().hex[:12]
    background_tasks.add_task(run_evaluation, req.collection, req.size, req.use_ragas, None, run_id)
    return {"run_id": run_id, "status": "running", "message": "评测已启动，轮询 /eval/runs/{run_id}"}


@router.get("/eval/runs")
def eval_runs() -> list[dict]:
    return list_runs()


@router.get("/eval/runs/{run_id}")
def eval_run_detail(run_id: str) -> dict:
    data = get_run(run_id)
    if data is None:
        raise HTTPException(404, "评测记录不存在")
    return data
