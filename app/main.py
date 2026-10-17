"""应用入口：FastAPI + 静态页面。"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import router
from app.config import get_settings

settings = get_settings()
WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="LangChain + pgvector 的企业知识库问答系统：混合检索 + 重排 + RAGAS 评测",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

app.include_router(router, prefix=settings.api_prefix)


@app.on_event("startup")
def on_startup() -> None:
    try:
        from app.db import init_db

        init_db()
    except Exception as exc:  # 数据库没起来时也要能启动，方便排查
        print(f"[startup] 数据库初始化失败：{exc}")


# ---------- 页面 ----------
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
